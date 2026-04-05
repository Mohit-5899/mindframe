"""
frame_analyzer.py - Detect key frames using mathematical analysis.

Uses SSIM-based change detection to find "completed" visual states:
frames where content has fully accumulated before a major transition
(slide change, board erase, camera switch).

Algorithm:
    1. Sample frames at 1fps
    2. Compute inter-frame difference: d(t) = 1 - SSIM(F(t), F(t-1))
    3. Find transitions: d(t) > μ(d) + k·σ(d)
    4. Capture frame just before each transition (completed state)
    5. Filter by edge density (keep frames with actual content)
    6. Force capture after long gaps (stability rule)

Functions:
    analyze_video(video_path) - full pipeline, returns list of key timestamps
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters for frame analysis."""
    sample_fps: float = 1.0
    sensitivity_k: float = 2.5
    capture_delta_sec: float = 2.0
    min_edge_density: float = 0.05
    max_gap_sec: float = 60.0
    min_accumulation: float = 0.1


@dataclass(frozen=True)
class KeyFrame:
    """A selected key frame with metadata."""
    timestamp: float
    accumulation_score: float
    edge_density: float
    reason: str


def _sample_frames(video_path: str, fps: float) -> list[tuple[float, np.ndarray]]:
    """Sample grayscale frames from video at given fps.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to sample.

    Returns:
        List of (timestamp_seconds, grayscale_frame) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    logger.info(
        "Sampling video: %.1f fps source, %d total frames, sampling every %d frames",
        video_fps, total_frames, frame_interval,
    )

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = frame_idx / video_fps
            frames.append((timestamp, gray))

        frame_idx += 1

    cap.release()
    logger.info("Sampled %d frames from video", len(frames))
    return frames


def _compute_differences(frames: list[tuple[float, np.ndarray]]) -> list[tuple[float, float]]:
    """Compute SSIM-based difference signal between consecutive frames.

    Args:
        frames: List of (timestamp, grayscale_frame) tuples.

    Returns:
        List of (timestamp, difference) tuples where difference = 1 - SSIM.
    """
    differences = []

    for i in range(1, len(frames)):
        timestamp = frames[i][0]
        score = ssim(frames[i - 1][1], frames[i][1])
        diff = 1.0 - score
        differences.append((timestamp, diff))

    logger.info("Computed %d frame differences", len(differences))
    return differences


def _find_transitions(
    differences: list[tuple[float, float]],
    k: float,
) -> list[float]:
    """Find major visual transitions using adaptive thresholding.

    Transition at time T when: d(T) > μ(d) + k·σ(d)

    Args:
        differences: List of (timestamp, difference) tuples.
        k: Sensitivity multiplier for threshold.

    Returns:
        List of transition timestamps.
    """
    if not differences:
        return []

    diffs = np.array([d for _, d in differences])
    mean_d = np.mean(diffs)
    std_d = np.std(diffs)
    threshold = mean_d + k * std_d

    transitions = [t for t, d in differences if d > threshold]

    logger.info(
        "Transition detection: μ=%.4f, σ=%.4f, θ=%.4f → %d transitions found",
        mean_d, std_d, threshold, len(transitions),
    )
    return transitions


def _compute_edge_density(frame: np.ndarray) -> float:
    """Compute edge density of a frame using Sobel operator.

    I(t) = |Sobel(F(t))| / num_pixels

    Args:
        frame: Grayscale frame.

    Returns:
        Edge density value (0 to 1 range, typically 0.01-0.20).
    """
    sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return float(np.mean(magnitude) / 255.0)


def _compute_accumulation(
    differences: list[tuple[float, float]],
    start_time: float,
    end_time: float,
) -> float:
    """Compute cumulative visual change between two timestamps.

    C(t) = Σ d(i) for i from start to end

    Args:
        differences: Full difference signal.
        start_time: Start of accumulation window.
        end_time: End of accumulation window.

    Returns:
        Accumulation score.
    """
    return sum(d for t, d in differences if start_time <= t < end_time)


def _get_frame_at_timestamp(
    frames: list[tuple[float, np.ndarray]],
    target_time: float,
) -> tuple[float, np.ndarray] | None:
    """Find the closest sampled frame to a target timestamp.

    Args:
        frames: Sampled frames list.
        target_time: Target timestamp in seconds.

    Returns:
        (actual_timestamp, frame) or None if no frames.
    """
    if not frames:
        return None

    closest = min(frames, key=lambda f: abs(f[0] - target_time))
    return closest


def analyze_video(
    video_path: str,
    config: AnalysisConfig | None = None,
) -> list[KeyFrame]:
    """Analyze a video to find key frames with completed visual content.

    Args:
        video_path: Path to the video file.
        config: Analysis parameters. Uses defaults if None.

    Returns:
        List of KeyFrame objects sorted by timestamp.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        RuntimeError: If video cannot be processed.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cfg = config or AnalysisConfig()
    logger.info("Starting frame analysis: %s", path.name)

    # Step 1: Sample frames
    frames = _sample_frames(str(path), cfg.sample_fps)
    if len(frames) < 2:
        logger.warning("Not enough frames to analyze")
        return []

    # Step 2: Compute difference signal
    differences = _compute_differences(frames)
    if not differences:
        return []

    # Step 3: Find transitions
    transitions = _find_transitions(differences, cfg.sensitivity_k)

    # Step 4: Build capture candidates
    video_start = frames[0][0]
    video_end = frames[-1][0]

    # Add video boundaries
    boundaries = [video_start] + transitions + [video_end]

    key_frames = []

    for i in range(1, len(boundaries)):
        segment_start = boundaries[i - 1]
        segment_end = boundaries[i]

        # Capture frame δ seconds before transition
        capture_time = max(segment_start, segment_end - cfg.capture_delta_sec)

        # Get the actual frame
        result = _get_frame_at_timestamp(frames, capture_time)
        if result is None:
            continue

        actual_time, frame = result

        # Compute scores
        accumulation = _compute_accumulation(differences, segment_start, segment_end)
        edge_density = _compute_edge_density(frame)

        # Filter: must have enough content
        if accumulation < cfg.min_accumulation and edge_density < cfg.min_edge_density:
            logger.debug(
                "Skipping t=%.1f: low accumulation (%.3f) and edge density (%.3f)",
                actual_time, accumulation, edge_density,
            )
            continue

        if edge_density < cfg.min_edge_density:
            logger.debug("Skipping t=%.1f: low edge density (%.3f)", actual_time, edge_density)
            continue

        key_frames.append(KeyFrame(
            timestamp=actual_time,
            accumulation_score=round(accumulation, 4),
            edge_density=round(edge_density, 4),
            reason="pre_transition",
        ))

    # Step 5: Stability rule — force capture after long gaps
    if key_frames:
        gap_frames = _fill_gaps(frames, key_frames, differences, cfg)
        key_frames.extend(gap_frames)

    # Sort and deduplicate (remove frames within 3s of each other)
    key_frames = _deduplicate(sorted(key_frames, key=lambda kf: kf.timestamp), min_gap=3.0)

    logger.info("Frame analysis complete: %d key frames selected", len(key_frames))
    return key_frames


def _fill_gaps(
    frames: list[tuple[float, np.ndarray]],
    existing: list[KeyFrame],
    differences: list[tuple[float, float]],
    cfg: AnalysisConfig,
) -> list[KeyFrame]:
    """Add frames in long gaps where no transition was detected.

    Args:
        frames: All sampled frames.
        existing: Already selected key frames.
        differences: Difference signal.
        cfg: Analysis config.

    Returns:
        Additional KeyFrame objects for gap regions.
    """
    gap_frames = []
    sorted_existing = sorted(existing, key=lambda kf: kf.timestamp)

    all_times = [frames[0][0]] + [kf.timestamp for kf in sorted_existing] + [frames[-1][0]]

    for i in range(1, len(all_times)):
        gap = all_times[i] - all_times[i - 1]
        if gap <= cfg.max_gap_sec:
            continue

        # Place captures every max_gap_sec within the gap
        num_captures = int(gap / cfg.max_gap_sec)
        for j in range(1, num_captures + 1):
            capture_time = all_times[i - 1] + j * cfg.max_gap_sec
            result = _get_frame_at_timestamp(frames, capture_time)
            if result is None:
                continue

            actual_time, frame = result
            edge_density = _compute_edge_density(frame)

            if edge_density >= cfg.min_edge_density:
                gap_frames.append(KeyFrame(
                    timestamp=actual_time,
                    accumulation_score=0.0,
                    edge_density=round(edge_density, 4),
                    reason="stability_gap",
                ))
                logger.debug("Gap capture at t=%.1f (edge=%.3f)", actual_time, edge_density)

    return gap_frames


def _deduplicate(key_frames: list[KeyFrame], min_gap: float) -> list[KeyFrame]:
    """Remove key frames that are too close together.

    Keeps the frame with higher accumulation score when two are within min_gap.

    Args:
        key_frames: Sorted list of key frames.
        min_gap: Minimum seconds between kept frames.

    Returns:
        Deduplicated list.
    """
    if not key_frames:
        return []

    result = [key_frames[0]]

    for kf in key_frames[1:]:
        if kf.timestamp - result[-1].timestamp >= min_gap:
            result.append(kf)
        elif kf.accumulation_score > result[-1].accumulation_score:
            result[-1] = kf

    return result
