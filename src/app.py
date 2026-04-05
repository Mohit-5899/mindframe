"""
app.py - Streamlit UI for MindFrame.

Clean, minimal interface to extract and display timestamped transcripts
from YouTube URLs or uploaded video files, with intelligent key frame extraction.

Run with: streamlit run src/app.py
"""

import logging
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

from src import setup_logging
from src.caption_fetcher import extract_video_id, fetch_captions
from src.audio_extractor import download_audio_from_youtube, extract_audio_from_file
from src.transcriber import transcribe
from src.formatter import format_for_display, format_timestamp, segments_to_plain_text
from src.frame_analyzer import analyze_video
from src.frame_extractor import extract_frames, pair_with_transcript

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MindFrame",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Layout */
    .block-container { max-width: 1100px; padding-top: 1.5rem; }

    /* Input row */
    .stRadio > div { flex-direction: row; gap: 1.5rem; }

    /* Scrollable transcript */
    .transcript-scroll {
        max-height: 520px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .transcript-scroll::-webkit-scrollbar { width: 4px; }
    .transcript-scroll::-webkit-scrollbar-thumb {
        background: rgba(128, 128, 128, 0.25);
        border-radius: 2px;
    }

    /* Transcript rows */
    .transcript-row {
        display: flex;
        gap: 0.75rem;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        align-items: flex-start;
    }
    .transcript-row:last-child { border-bottom: none; }
    .ts {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.72rem;
        color: #7c73ff;
        background: rgba(108, 99, 255, 0.1);
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        white-space: nowrap;
        min-width: 48px;
        text-align: center;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .tx {
        font-size: 0.85rem;
        line-height: 1.5;
        color: inherit;
    }

    /* Status steps */
    .step-done { color: #2ecc71; }
    .step-active { color: #7c73ff; }

    /* Summary stats */
    .stat-box {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        text-align: center;
    }
    .stat-num {
        font-size: 1.3rem;
        font-weight: 700;
        color: inherit;
    }
    .stat-label {
        font-size: 0.7rem;
        color: rgba(128, 128, 128, 0.65);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Frame grid */
    .frame-overlay {
        position: relative;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 0.3rem;
    }
    .frame-ts-badge {
        position: absolute;
        top: 6px;
        left: 6px;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.68rem;
        color: #fff;
        background: rgba(0, 0, 0, 0.6);
        padding: 0.15rem 0.4rem;
        border-radius: 3px;
        backdrop-filter: blur(4px);
    }
    .frame-caption {
        font-size: 0.75rem;
        color: inherit;
        opacity: 0.7;
        line-height: 1.4;
        margin-top: 0.2rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .frame-badge-row {
        display: flex;
        gap: 0.4rem;
        margin-top: 0.25rem;
        flex-wrap: wrap;
    }
    .frame-badge {
        font-size: 0.62rem;
        padding: 0.1rem 0.35rem;
        border-radius: 3px;
        background: rgba(108, 99, 255, 0.1);
        color: #7c73ff;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## MindFrame")
st.caption("Extract timestamped transcripts and key visual frames from videos.")
st.markdown("---")


# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([3, 1], vertical_alignment="bottom")

with col_input:
    input_mode = st.radio("Source", ["YouTube URL", "Upload video file"], label_visibility="collapsed")

source_ready = False
youtube_url = ""
uploaded_file = None

if input_mode == "YouTube URL":
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    source_ready = bool(youtube_url.strip())
else:
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mkv", "webm", "mov", "avi"],
        label_visibility="collapsed",
    )
    source_ready = uploaded_file is not None

go = st.button("Extract Transcript", disabled=not source_ready, width="stretch")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _render_step(label: str, state: str) -> str:
    """Return HTML for a pipeline step."""
    icons = {"done": "&#10003;", "active": "&#9654;"}
    return f'<span class="step-{state}">{icons[state]}</span> {label}'


def _run_youtube_pipeline(url: str) -> dict:
    """Full pipeline for a YouTube URL."""
    status = st.empty()

    logger.info("Pipeline start — YouTube URL: %s", url)
    status.markdown(_render_step("Checking for existing captions...", "active"), unsafe_allow_html=True)
    video_id = extract_video_id(url)
    result = fetch_captions(video_id)

    if result is not None:
        logger.info("Pipeline complete — used YouTube captions")
        status.markdown(_render_step("Captions found!", "done"), unsafe_allow_html=True)
        return result

    logger.info("No captions available, falling back to audio download")
    status.markdown(_render_step("No captions. Downloading audio...", "active"), unsafe_allow_html=True)
    wav_path = download_audio_from_youtube(video_id, output_dir="output")

    status.markdown(_render_step("Transcribing with Whisper...", "active"), unsafe_allow_html=True)
    result = transcribe(str(wav_path))
    result["video_id"] = video_id

    logger.info("Pipeline complete — used Whisper transcription")
    status.markdown(_render_step("Transcription complete", "done"), unsafe_allow_html=True)
    return result


def _run_file_pipeline(file) -> dict:
    """Full pipeline for an uploaded video file."""
    status = st.empty()

    logger.info("Pipeline start — uploaded file: %s", file.name)
    status.markdown(_render_step("Processing uploaded file...", "active"), unsafe_allow_html=True)
    suffix = Path(file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    logger.info("Saved upload to temp file: %s", tmp_path)

    status.markdown(_render_step("Extracting audio...", "active"), unsafe_allow_html=True)
    wav_path = extract_audio_from_file(tmp_path, output_dir="output")

    status.markdown(_render_step("Transcribing with Whisper...", "active"), unsafe_allow_html=True)
    result = transcribe(str(wav_path))
    result["source_file"] = file.name

    logger.info("Pipeline complete — transcribed uploaded file")
    status.markdown(_render_step("Transcription complete", "done"), unsafe_allow_html=True)
    return result


def _download_video_for_frames(video_id: str) -> str | None:
    """Download YouTube video for frame analysis. Returns path or None."""
    video_file = Path("output") / f"{video_id}.mp4"
    if video_file.exists():
        return str(video_file)

    status_dl = st.empty()
    status_dl.markdown(_render_step("Downloading video for frame analysis...", "active"), unsafe_allow_html=True)
    logger.info("Downloading video for frame analysis: %s", video_id)

    dl_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--output", str(video_file),
        "--no-playlist",
        "--quiet",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    dl_result = subprocess.run(dl_cmd, capture_output=True, text=True)

    if dl_result.returncode != 0:
        logger.error("Video download failed: %s", dl_result.stderr.strip())
        st.error(f"Failed to download video: {dl_result.stderr.strip()}")
        return None

    status_dl.markdown(_render_step("Video downloaded", "done"), unsafe_allow_html=True)
    return str(video_file)


def _run_frame_analysis(video_path: str, segments: list[dict]) -> list:
    """Run the frame analysis pipeline on a video file."""
    status = st.empty()

    status.markdown(_render_step("Analyzing video frames (SSIM + edge detection)...", "active"), unsafe_allow_html=True)
    logger.info("Starting frame analysis: %s", video_path)
    key_frames = analyze_video(video_path)

    if not key_frames:
        status.markdown(_render_step("No key frames detected", "done"), unsafe_allow_html=True)
        return []

    status.markdown(
        _render_step(f"Found {len(key_frames)} key frames. Extracting...", "active"),
        unsafe_allow_html=True,
    )

    timestamps = [kf.timestamp for kf in key_frames]
    video_stem = Path(video_path).stem
    image_paths = extract_frames(video_path, timestamps, output_dir=f"output/frames/{video_stem}")

    paired = pair_with_transcript(key_frames, segments)

    results = []
    for i, frame_data in enumerate(paired):
        image_path = image_paths[i] if i < len(image_paths) else ""
        results.append({
            "timestamp": frame_data.timestamp,
            "image_path": image_path,
            "transcript_text": frame_data.transcript_text,
            "reason": frame_data.reason,
            "accumulation": frame_data.accumulation_score,
            "edge_density": frame_data.edge_density,
        })

    status.markdown(_render_step(f"Extracted {len(results)} key frames", "done"), unsafe_allow_html=True)
    logger.info("Frame extraction complete: %d frames", len(results))
    return results


# ---------------------------------------------------------------------------
# Render: Stats bar
# ---------------------------------------------------------------------------

def _render_stats(segments: list[dict], method: str, num_frames: int = 0):
    """Render summary stats in a compact row."""
    display_segs = format_for_display(segments)
    full_text = segments_to_plain_text(segments)
    total_duration = segments[-1].get("end", 0) if segments else 0
    word_count = len(full_text.split())
    method_label = "YouTube Captions" if method == "youtube_captions" else "Groq Whisper"

    cols = st.columns(5)
    stats = [
        (format_timestamp(total_duration), "Duration"),
        (f"{len(display_segs)}", "Segments"),
        (f"{word_count:,}", "Words"),
        (f"{num_frames}", "Key Frames"),
        (method_label, "Source"),
    ]
    for col, (val, label) in zip(cols, stats):
        with col:
            st.markdown(
                f'<div class="stat-box"><div class="stat-num">{val}</div>'
                f'<div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Render: Transcript tab
# ---------------------------------------------------------------------------

def _render_transcript(segments: list[dict]):
    """Render transcript in a scrollable container."""
    display_segs = format_for_display(segments)
    if not display_segs:
        st.info("No transcript segments found.")
        return

    parts = []
    for seg in display_segs:
        parts.append(
            f'<div class="transcript-row">'
            f'  <span class="ts">{seg["timestamp"]}</span>'
            f'  <span class="tx">{seg["text"]}</span>'
            f'</div>'
        )

    html = f'<div class="transcript-scroll">{"".join(parts)}</div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render: Key Frames tab
# ---------------------------------------------------------------------------

def _render_key_frames(frames: list[dict]):
    """Render key frames in a 3-column grid layout."""
    if not frames:
        st.info("No key frames found in this video.")
        return

    # 3-column grid
    cols_per_row = 3
    for row_start in range(0, len(frames), cols_per_row):
        row_frames = frames[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, frame in zip(cols, row_frames):
            with col:
                ts_label = format_timestamp(frame["timestamp"])
                reason = "Pre-transition" if frame["reason"] == "pre_transition" else "Gap capture"

                # Image with timestamp overlay
                if frame["image_path"] and Path(frame["image_path"]).exists():
                    st.image(frame["image_path"], width="stretch")

                # Metadata below image
                badges = (
                    f'<div class="frame-badge-row">'
                    f'  <span class="ts">{ts_label}</span>'
                    f'  <span class="frame-badge">{reason}</span>'
                    f'  <span class="frame-badge">E:{frame["edge_density"]:.2f}</span>'
                    f'</div>'
                )
                st.markdown(badges, unsafe_allow_html=True)

                # Transcript context (truncated)
                if frame["transcript_text"]:
                    text = frame["transcript_text"][:120]
                    if len(frame["transcript_text"]) > 120:
                        text += "..."
                    st.markdown(
                        f'<div class="frame-caption">{text}</div>',
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if go:
    st.markdown("---")
    try:
        video_path_for_frames = None

        if input_mode == "YouTube URL":
            result = _run_youtube_pipeline(youtube_url.strip())
            video_id = extract_video_id(youtube_url.strip())
            video_path_for_frames = _download_video_for_frames(video_id)
        else:
            result = _run_file_pipeline(uploaded_file)
            suffix = Path(uploaded_file.name).suffix
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                video_path_for_frames = tmp.name

        segments = result.get("segments", [])
        key_frame_results = []

        if video_path_for_frames:
            key_frame_results = _run_frame_analysis(video_path_for_frames, segments)

        # --- Results ---
        st.markdown("---")

        _render_stats(segments, result.get("method", ""), len(key_frame_results))

        st.markdown("")

        tab_transcript, tab_frames = st.tabs(["Transcript", f"Key Frames ({len(key_frame_results)})"])

        with tab_transcript:
            _render_transcript(segments)

        with tab_frames:
            _render_key_frames(key_frame_results)

    except ValueError as e:
        st.error(str(e))
    except RuntimeError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")
