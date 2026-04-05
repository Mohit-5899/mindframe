"""
audio_extractor.py - Download/extract audio from YouTube URLs or local video files.

This is used when YouTube captions are not available, so we need the actual
audio to run speech-to-text (faster-whisper).

Two paths:
    1. YouTube URL → yt-dlp downloads audio directly as WAV
    2. Local video file → ffmpeg extracts the audio track as WAV

Both produce a .wav file that faster-whisper can consume.

Functions:
    download_audio_from_youtube(video_id, output_dir) - YouTube → WAV
    extract_audio_from_file(video_path, output_dir) - Local video → WAV
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def download_audio_from_youtube(video_id: str, output_dir: str = "output") -> Path:
    """Download audio from a YouTube video using yt-dlp.

    Downloads the best available audio and converts to WAV format
    (16kHz mono - optimal for speech recognition).

    Args:
        video_id: The YouTube video ID.
        output_dir: Directory to save the audio file.

    Returns:
        Path to the downloaded WAV file.

    Raises:
        RuntimeError: If yt-dlp fails (video unavailable, network error, etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_file = output_path / f"{video_id}.wav"

    if wav_file.exists():
        logger.info("Audio already downloaded, skipping: %s", wav_file)
        return wav_file

    url = f"https://www.youtube.com/watch?v={video_id}"
    logger.info("Downloading audio from YouTube: %s", video_id)

    command = [
        "yt-dlp",
        "--extract-audio",               # Download audio only
        "--audio-format", "wav",          # Convert to WAV
        "--postprocessor-args",
        "ffmpeg:-ar 16000 -ac 1",         # 16kHz mono (optimal for Whisper)
        "--output", str(output_path / f"{video_id}.%(ext)s"),
        "--no-playlist",                  # Don't download entire playlist
        "--quiet",                        # Suppress progress output
        url,
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("yt-dlp failed for %s: %s", video_id, result.stderr.strip())
        raise RuntimeError(
            f"Failed to download audio from YouTube.\n"
            f"Video ID: {video_id}\n"
            f"Error: {result.stderr.strip()}"
        )

    if not wav_file.exists():
        logger.error("yt-dlp completed but WAV not found at: %s", wav_file)
        raise RuntimeError(
            f"yt-dlp completed but WAV file not found at: {wav_file}\n"
            f"Check if ffmpeg is installed."
        )

    logger.info("Audio downloaded: %s", wav_file)
    return wav_file


def extract_audio_from_file(video_path: str, output_dir: str = "output") -> Path:
    """Extract audio from a local video file using ffmpeg.

    Converts the audio track to WAV format (16kHz mono - optimal for
    speech recognition).

    Args:
        video_path: Path to the local video file.
        output_dir: Directory to save the extracted audio.

    Returns:
        Path to the extracted WAV file.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        RuntimeError: If ffmpeg fails.
    """
    source = Path(video_path)
    if not source.exists():
        logger.error("Video file not found: %s", video_path)
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_file = output_path / f"{source.stem}.wav"

    if wav_file.exists():
        logger.info("Audio already extracted, skipping: %s", wav_file)
        return wav_file

    logger.info("Extracting audio from file: %s", source.name)

    command = [
        "ffmpeg",
        "-i", str(source),          # Input video
        "-vn",                       # No video (audio only)
        "-ar", "16000",              # 16kHz sample rate
        "-ac", "1",                  # Mono channel
        "-y",                        # Overwrite if exists
        "-loglevel", "error",        # Only show errors
        str(wav_file),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg failed for %s: %s", source.name, result.stderr.strip())
        raise RuntimeError(
            f"Failed to extract audio from video.\n"
            f"File: {video_path}\n"
            f"Error: {result.stderr.strip()}"
        )

    logger.info("Audio extracted: %s", wav_file)
    return wav_file
