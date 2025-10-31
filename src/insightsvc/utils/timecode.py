"""Timestamp and timecode utilities.

Provides functions for working with audio timestamps, converting between formats,
and formatting for human-readable output.
"""

from datetime import timedelta


def seconds_to_timecode(seconds: float, include_ms: bool = True) -> str:
    """Convert seconds to MM:SS or HH:MM:SS timecode.

    Args:
        seconds: Time in seconds.
        include_ms: Whether to include milliseconds.

    Returns:
        Formatted timecode string.

    Examples:
        >>> seconds_to_timecode(65.5)
        '01:05.500'
        >>> seconds_to_timecode(3665.123)
        '1:01:05.123'
    """
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    ms = td.microseconds // 1000

    if hours > 0:
        base = f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        base = f"{minutes:02d}:{secs:02d}"

    if include_ms:
        return f"{base}.{ms:03d}"
    return base


def timecode_to_seconds(timecode: str) -> float:
    """Convert timecode to seconds.

    Args:
        timecode: Timecode string (MM:SS, HH:MM:SS, or with .ms).

    Returns:
        Time in seconds.

    Examples:
        >>> timecode_to_seconds("01:05.500")
        65.5
        >>> timecode_to_seconds("1:01:05")
        3665.0
    """
    # Split milliseconds if present
    if "." in timecode:
        time_part, ms_part = timecode.split(".")
        ms = float(f"0.{ms_part}")
    else:
        time_part = timecode
        ms = 0.0

    # Split time components
    parts = time_part.split(":")
    parts = [int(p) for p in parts]

    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid timecode format: {timecode}")

    total = hours * 3600 + minutes * 60 + seconds + ms
    return total


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.

    Examples:
        >>> format_duration(65.5)
        '1m 5s'
        >>> format_duration(3665)
        '1h 1m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def time_overlap(start1: float, end1: float, start2: float, end2: float) -> tuple[float, float]:
    """Calculate overlap between two time intervals.

    Args:
        start1: Start of first interval.
        end1: End of first interval.
        start2: Start of second interval.
        end2: End of second interval.

    Returns:
        Tuple of (overlap_duration, overlap_fraction) where fraction is relative to first interval.

    Examples:
        >>> time_overlap(0, 10, 5, 15)
        (5.0, 0.5)
        >>> time_overlap(0, 10, 20, 30)
        (0.0, 0.0)
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = max(0.0, overlap_end - overlap_start)

    interval1_duration = end1 - start1
    if interval1_duration <= 0:
        return 0.0, 0.0

    overlap_fraction = overlap_duration / interval1_duration
    return overlap_duration, overlap_fraction


def merge_overlapping_segments(
    segments: list[tuple[float, float]], min_gap: float = 0.0
) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent time segments.

    Args:
        segments: List of (start, end) tuples.
        min_gap: Minimum gap to keep segments separate.

    Returns:
        List of merged (start, end) tuples.

    Examples:
        >>> merge_overlapping_segments([(0, 5), (3, 8), (10, 15)])
        [(0, 8), (10, 15)]
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segs[0]]

    for start, end in sorted_segs[1:]:
        last_start, last_end = merged[-1]

        # Check if segments overlap or are within min_gap
        if start <= last_end + min_gap:
            # Merge by extending the last segment
            merged[-1] = (last_start, max(last_end, end))
        else:
            # Add as new segment
            merged.append((start, end))

    return merged


def split_into_chunks(
    duration: float, chunk_size: float, overlap: float = 0.0
) -> list[tuple[float, float]]:
    """Split duration into overlapping chunks for processing.

    Args:
        duration: Total duration in seconds.
        chunk_size: Size of each chunk in seconds.
        overlap: Overlap between chunks in seconds.

    Returns:
        List of (start, end) tuples for each chunk.

    Examples:
        >>> split_into_chunks(100, 30, 5)
        [(0, 30), (25, 55), (50, 80), (75, 100)]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    start = 0.0
    step = chunk_size - overlap

    while start < duration:
        end = min(start + chunk_size, duration)
        chunks.append((start, end))

        if end >= duration:
            break

        start += step

    return chunks
