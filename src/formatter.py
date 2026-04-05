"""
formatter.py - Format transcript segments for display.

Takes the standardized segment list from caption_fetcher or transcriber
and produces human-readable output.

Functions:
    format_timestamp(seconds) - float seconds -> "MM:SS" string
    format_for_display(segments) - segments -> list of display-ready dicts
    segments_to_plain_text(segments) - segments -> full text string
"""


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like "01:23" or "1:05:30" for hour+ content.
    """
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_for_display(segments: list[dict]) -> list[dict]:
    """Format segments into display-ready dicts with readable timestamps.

    Args:
        segments: List of {"start": float, "end": float, "text": str}.

    Returns:
        List of {"timestamp": "MM:SS", "start": float, "text": str}.
    """
    return [
        {
            "timestamp": format_timestamp(seg["start"]),
            "start": seg["start"],
            "text": seg["text"],
        }
        for seg in segments
        if seg["text"].strip()
    ]


def segments_to_plain_text(segments: list[dict]) -> str:
    """Join all segment text into a single string.

    Args:
        segments: List of segment dicts with "text" key.

    Returns:
        Full transcript as a single string.
    """
    return " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())
