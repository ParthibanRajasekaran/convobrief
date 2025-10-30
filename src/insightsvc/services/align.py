"""Word-to-speaker alignment using interval trees.

Assigns each transcribed word to a speaker based on diarization segments
using interval tree for efficient overlap computation.
"""

from intervaltree import Interval, IntervalTree

from insightsvc.logging import get_logger
from insightsvc.models.base import ASRWord, DiarizationSegment
from insightsvc.schemas import Word

logger = get_logger(__name__)


def overlap_duration(interval: Interval, word_start: float, word_end: float) -> float:
    """Calculate overlap duration between interval and word.

    Args:
        interval: Interval tree interval.
        word_start: Word start time.
        word_end: Word end time.

    Returns:
        Duration of overlap in seconds.
    """
    overlap_start = max(interval.begin, word_start)
    overlap_end = min(interval.end, word_end)
    return max(0.0, overlap_end - overlap_start)


def assign_speakers(
    words: list[ASRWord],
    diarization_segments: list[DiarizationSegment],
    enable_overlaps: bool = True,
    overlap_threshold: float = 0.4,
) -> list[Word]:
    """Assign speakers to words using interval tree alignment.

    Args:
        words: List of ASR words with timestamps.
        diarization_segments: List of diarization segments.
        enable_overlaps: Whether to detect and mark overlapping speech.
        overlap_threshold: Threshold for marking word as overlap (0-1).

    Returns:
        List of Word objects with speaker assignments.

    Examples:
        >>> words = [ASRWord("Hello", 0.0, 0.5, 0.95)]
        >>> segments = [DiarizationSegment("S1", 0.0, 1.0, 0.9)]
        >>> result = assign_speakers(words, segments)
        >>> result[0].speaker
        'S1'
    """
    logger.info(
        "Starting speaker alignment",
        word_count=len(words),
        segment_count=len(diarization_segments),
    )

    # Build interval tree over diarization segments
    tree = IntervalTree()
    for seg in diarization_segments:
        tree.addi(seg.start, seg.end, seg)

    aligned_words: list[Word] = []

    for word in words:
        # Find all overlapping segments
        overlaps = tree.overlap(word.start, word.end)

        if not overlaps:
            # No speaker found - assign None
            aligned_words.append(
                Word(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    speaker=None,
                    conf=word.confidence,
                    overlap=False,
                )
            )
            continue

        # Calculate overlap durations for each candidate
        candidates = []
        for interval in overlaps:
            seg: DiarizationSegment = interval.data
            duration = overlap_duration(interval, word.start, word.end)
            candidates.append((duration, seg.score, seg.speaker, seg))

        # Sort by overlap duration (desc), then by segment score (desc)
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Assign to speaker with maximum overlap
        best_duration, _, best_speaker, best_seg = candidates[0]
        word_duration = word.end - word.start

        # Check if this is overlapping speech
        is_overlap = False
        if enable_overlaps and len(candidates) > 1:
            second_duration = candidates[1][0]
            # If second speaker overlaps >threshold of word duration, mark as overlap
            if second_duration / word_duration > overlap_threshold:
                is_overlap = True

        aligned_words.append(
            Word(
                text=word.text,
                start=word.start,
                end=word.end,
                speaker=best_speaker,
                conf=word.confidence,
                overlap=is_overlap,
            )
        )

    # Log alignment statistics
    assigned_count = sum(1 for w in aligned_words if w.speaker is not None)
    overlap_count = sum(1 for w in aligned_words if w.overlap)

    logger.info(
        "Speaker alignment complete",
        total_words=len(aligned_words),
        assigned_words=assigned_count,
        overlap_words=overlap_count,
        assignment_rate=assigned_count / len(aligned_words) if aligned_words else 0.0,
    )

    return aligned_words


def group_into_utterances(
    words: list[Word],
    max_gap_sec: float = 1.0,
) -> list[dict]:
    """Group words into continuous utterances by speaker.

    Args:
        words: List of speaker-aligned words.
        max_gap_sec: Maximum gap between words in same utterance.

    Returns:
        List of utterance dictionaries.
    """
    if not words:
        return []

    utterances = []
    current_utterance: dict | None = None

    for word in words:
        # Skip words without speaker
        if word.speaker is None:
            continue

        # Start new utterance if:
        # 1. First word
        # 2. Different speaker
        # 3. Gap too large
        if current_utterance is None:
            current_utterance = {
                "speaker": word.speaker,
                "words": [word],
                "start": word.start,
                "end": word.end,
            }
        elif (
            word.speaker != current_utterance["speaker"]
            or word.start - current_utterance["end"] > max_gap_sec
        ):
            # Finalize current utterance
            utterances.append(current_utterance)

            # Start new utterance
            current_utterance = {
                "speaker": word.speaker,
                "words": [word],
                "start": word.start,
                "end": word.end,
            }
        else:
            # Continue current utterance
            current_utterance["words"].append(word)
            current_utterance["end"] = word.end

    # Add final utterance
    if current_utterance is not None:
        utterances.append(current_utterance)

    logger.info("Grouped words into utterances", utterance_count=len(utterances))

    return utterances
