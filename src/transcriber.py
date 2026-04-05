"""
transcriber.py - Speech-to-text via Groq Whisper API.

Takes a WAV audio file and produces timestamped transcript segments.
Uses the same segment format as caption_fetcher for consistency:
    {"start": float, "end": float, "text": str}

Functions:
    transcribe(audio_path) - WAV file -> list of segments
"""

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3-turbo"
MAX_CHUNK_SIZE_MB = 25


def _get_api_key() -> str:
    """Load Groq API key from environment."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at https://console.groq.com"
        )
    return key


def _transcribe_chunk(file_path: Path, api_key: str) -> list[dict]:
    """Send a single audio file to Groq Whisper API and return segments."""
    logger.info("Sending to Groq API: %s (%.1f MB)", file_path.name, file_path.stat().st_size / 1_048_576)

    with open(file_path, "rb") as f:
        response = requests.post(
            GROQ_TRANSCRIPTION_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (file_path.name, f, "audio/wav")},
            data={
                "model": GROQ_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
                "language": "en",
                "temperature": "0",
            },
            timeout=300,
        )

    if response.status_code != 200:
        logger.error("Groq API error (%d): %s", response.status_code, response.text)
        raise RuntimeError(
            f"Groq API request failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    raw_segments = data.get("segments", [])

    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in raw_segments
        if seg.get("text", "").strip()
    ]


def transcribe(audio_path: str) -> dict:
    """Transcribe an audio file using Groq Whisper API.

    Args:
        audio_path: Path to a WAV audio file.

    Returns:
        A dict with method and segments list:
        {
            "method": "groq_whisper",
            "model": "whisper-large-v3-turbo",
            "segments": [
                {"start": 0.0, "end": 3.52, "text": "Hello everyone"},
                ...
            ]
        }

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        RuntimeError: If transcription fails.
    """
    path = Path(audio_path)
    if not path.exists():
        logger.error("Audio file not found: %s", audio_path)
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = _get_api_key()

    logger.info("Transcribing via Groq Whisper: %s", path.name)
    segments = _transcribe_chunk(path, api_key)
    logger.info("Transcription complete: %d segments", len(segments))

    return {
        "method": "groq_whisper",
        "model": GROQ_MODEL,
        "segments": segments,
    }
