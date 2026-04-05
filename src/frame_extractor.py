"""
frame_extractor.py - Extract specific frames from video using ffmpeg.

Given a list of timestamps, extracts high-quality frames as images.
Pairs each frame with the corresponding transcript segment.

Functions:
    extract_frames(video_path, timestamps, output_dir) - timestamps -> image files
    pair_with_transcript(key_frames, segments) - match frames to transcript text
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.formatter import format_timestamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedFrame:
    """A frame extracted from video with metadata."""
    timestamp: float
    image_path: str
    transcript_text: str
    reason: str
    accumulation_score: float
    edge_density: float


def extract_frames(
    video_path: str,
    timestamps: list[float],
    output_dir: str = "output/frames",
) -> list[str]:
    """Extract frames at specific timestamps using ffmpeg.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps in seconds.
        output_dir: Directory to save extracted frames.

    Returns:
        List of paths to extracted frame images.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        RuntimeError: If ffmpeg fails.
    """
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %d frames from: %s", len(timestamps), source.name)

    image_paths = []

    for ts in timestamps:
        frame_name = f"frame_{format_timestamp(ts).replace(':', '-')}_{ts:.1f}.jpg"
        frame_path = out_path / frame_name

        if frame_path.exists():
            logger.debug("Frame already exists, skipping: %s", frame_name)
            image_paths.append(str(frame_path))
            continue

        command = [
            "ffmpeg",
            "-ss", str(ts),
            "-i", str(source),
            "-frames:v", "1",
            "-q:v", "2",
            "-y",
            "-loglevel", "error",
            str(frame_path),
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("ffmpeg failed for t=%.1f: %s", ts, result.stderr.strip())
            continue

        if frame_path.exists():
            image_paths.append(str(frame_path))
            logger.debug("Extracted frame at t=%.1f: %s", ts, frame_name)
        else:
            logger.warning("Frame not created for t=%.1f", ts)

    logger.info("Extracted %d/%d frames", len(image_paths), len(timestamps))
    return image_paths


def pair_with_transcript(
    key_frames: list,
    segments: list[dict],
) -> list[ExtractedFrame]:
    """Pair key frames with the nearest transcript text.

    For each key frame, finds the transcript segment that overlaps or is
    closest to the frame's timestamp.

    Args:
        key_frames: List of KeyFrame objects from frame_analyzer.
        segments: Transcript segments with start, end, text.

    Returns:
        List of ExtractedFrame objects with transcript context.
    """
    if not segments:
        return [
            ExtractedFrame(
                timestamp=kf.timestamp,
                image_path="",
                transcript_text="",
                reason=kf.reason,
                accumulation_score=kf.accumulation_score,
                edge_density=kf.edge_density,
            )
            for kf in key_frames
        ]

    paired = []

    for kf in key_frames:
        # Find segments within a window around the frame timestamp
        window_start = kf.timestamp - 10
        window_end = kf.timestamp + 5

        nearby = [
            seg for seg in segments
            if seg["start"] <= window_end and seg.get("end", seg["start"]) >= window_start
        ]

        if nearby:
            text = " ".join(seg["text"].strip() for seg in nearby if seg["text"].strip())
        else:
            # Fall back to closest segment
            closest = min(segments, key=lambda s: abs(s["start"] - kf.timestamp))
            text = closest["text"].strip()

        paired.append(ExtractedFrame(
            timestamp=kf.timestamp,
            image_path="",
            transcript_text=text,
            reason=kf.reason,
            accumulation_score=kf.accumulation_score,
            edge_density=kf.edge_density,
        ))

    logger.info("Paired %d frames with transcript context", len(paired))
    return paired
