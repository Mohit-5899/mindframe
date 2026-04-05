"""
caption_fetcher.py - Fetch existing YouTube captions/subtitles.

This is the "fast path". If YouTube already has captions for a video,
we grab them directly - no audio download or transcription needed.

Functions:
    extract_video_id(url) - Parse YouTube URL → video ID
    fetch_captions(video_id) - Fetch timestamped captions from YouTube
"""

import logging
import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL.

    Supports these formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - https://youtube.com/embed/VIDEO_ID

    Args:
        url: A YouTube video URL.

    Returns:
        The video ID string (typically 11 characters).

    Raises:
        ValueError: If the URL is not a valid YouTube URL or no video ID found.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # Strip "www." for consistent matching
    hostname = hostname.replace("www.", "")

    # youtube.com/watch?v=VIDEO_ID
    if hostname in ("youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id:
                return video_id

        # youtube.com/shorts/VIDEO_ID or youtube.com/embed/VIDEO_ID
        match = re.match(r"^/(shorts|embed)/([a-zA-Z0-9_-]+)", parsed.path)
        if match:
            return match.group(2)

    # youtu.be/VIDEO_ID
    if hostname == "youtu.be":
        video_id = parsed.path.lstrip("/")
        if video_id:
            return video_id

    logger.error("Failed to extract video ID from URL: %s", url)
    raise ValueError(
        f"Could not extract video ID from URL: {url}\n"
        "Supported formats:\n"
        "  - https://www.youtube.com/watch?v=VIDEO_ID\n"
        "  - https://youtu.be/VIDEO_ID\n"
        "  - https://www.youtube.com/shorts/VIDEO_ID"
    )


def fetch_captions(video_id: str) -> dict | None:
    """Fetch existing captions/subtitles from YouTube.

    Tries to get captions in this priority:
        1. Manually uploaded captions (most accurate)
        2. Auto-generated captions (YouTube's speech recognition)

    Args:
        video_id: The YouTube video ID.

    Returns:
        A dict with video_id, method, and segments list, e.g.:
        {
            "video_id": "abc123",
            "method": "youtube_captions",
            "segments": [
                {"start": 0.0, "end": 3.52, "text": "Hello everyone"},
                {"start": 3.52, "end": 7.1, "text": "Welcome"},
            ]
        }
        Returns None if no captions are available.
    """
    logger.info("Fetching captions for video: %s", video_id)

    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except Exception as e:
        logger.warning("No transcript list available for %s: %s", video_id, e)
        return None

    transcript = _find_best_transcript(transcript_list)
    if transcript is None:
        logger.info("No usable transcript found for %s", video_id)
        return None

    try:
        raw_segments = transcript.fetch()
    except Exception as e:
        logger.error("Failed to fetch transcript content for %s: %s", video_id, e)
        return None

    segments = [
        {
            "start": round(segment.start, 2),
            "end": round(segment.start + segment.duration, 2),
            "text": segment.text,
        }
        for segment in raw_segments
    ]

    logger.info("Fetched %d caption segments for %s", len(segments), video_id)

    return {
        "video_id": video_id,
        "method": "youtube_captions",
        "segments": segments,
    }


def _find_best_transcript(transcript_list):
    """Pick the best available transcript: manual first, then auto-generated.

    Args:
        transcript_list: A TranscriptList from youtube-transcript-api.

    Returns:
        A Transcript object, or None if nothing usable found.
    """
    # Priority 1: Manually created captions (English preferred)
    try:
        t = transcript_list.find_manually_created_transcript(["en"])
        logger.info("Found manual English transcript")
        return t
    except Exception:
        pass

    # Priority 2: Manually created in any language
    try:
        manual = [t for t in transcript_list if not t.is_generated]
        if manual:
            logger.info("Found manual transcript (language: %s)", manual[0].language)
            return manual[0]
    except Exception:
        pass

    # Priority 3: Auto-generated (English preferred)
    try:
        t = transcript_list.find_generated_transcript(["en"])
        logger.info("Found auto-generated English transcript")
        return t
    except Exception:
        pass

    # Priority 4: Auto-generated in any language
    try:
        generated = [t for t in transcript_list if t.is_generated]
        if generated:
            logger.info("Found auto-generated transcript (language: %s)", generated[0].language)
            return generated[0]
    except Exception:
        pass

    logger.warning("No transcript found in any language or type")
    return None
