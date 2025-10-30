"""Unit tests for speaker alignment."""

import pytest

from insightsvc.models.base import ASRWord, DiarizationSegment
from insightsvc.services.align import assign_speakers, group_into_utterances


def test_assign_speakers_basic():
    """Test basic speaker assignment."""
    words = [
        ASRWord(text="Hello", start=0.0, end=0.5, confidence=0.95),
        ASRWord(text="world", start=0.6, end=1.0, confidence=0.92),
    ]

    segments = [
        DiarizationSegment(speaker="S1", start=0.0, end=1.5, score=0.9),
    ]

    result = assign_speakers(words, segments, enable_overlaps=False)

    assert len(result) == 2
    assert result[0].speaker == "S1"
    assert result[0].text == "Hello"
    assert result[1].speaker == "S1"


def test_assign_speakers_multiple():
    """Test assignment with multiple speakers."""
    words = [
        ASRWord(text="Hello", start=0.0, end=0.5, confidence=0.95),
        ASRWord(text="Hi", start=1.0, end=1.3, confidence=0.93),
    ]

    segments = [
        DiarizationSegment(speaker="S1", start=0.0, end=0.8, score=0.9),
        DiarizationSegment(speaker="S2", start=0.9, end=1.5, score=0.85),
    ]

    result = assign_speakers(words, segments)

    assert result[0].speaker == "S1"
    assert result[1].speaker == "S2"


def test_assign_speakers_overlap():
    """Test overlap detection."""
    words = [
        ASRWord(text="Test", start=0.5, end=1.0, confidence=0.95),
    ]

    segments = [
        DiarizationSegment(speaker="S1", start=0.0, end=0.8, score=0.9),
        DiarizationSegment(speaker="S2", start=0.7, end=1.5, score=0.85),
    ]

    result = assign_speakers(words, segments, enable_overlaps=True, overlap_threshold=0.3)

    # Word overlaps both speakers significantly
    assert result[0].overlap is True


def test_assign_speakers_no_match():
    """Test handling of words without matching segments."""
    words = [
        ASRWord(text="Orphan", start=5.0, end=5.5, confidence=0.95),
    ]

    segments = [
        DiarizationSegment(speaker="S1", start=0.0, end=1.0, score=0.9),
    ]

    result = assign_speakers(words, segments)

    assert result[0].speaker is None


def test_group_into_utterances():
    """Test grouping words into utterances."""
    from insightsvc.schemas import Word

    words = [
        Word(text="Hello", start=0.0, end=0.5, speaker="S1", conf=0.95, overlap=False),
        Word(text="world", start=0.6, end=1.0, speaker="S1", conf=0.92, overlap=False),
        Word(text="Hi", start=2.0, end=2.3, speaker="S2", conf=0.93, overlap=False),
        Word(text="there", start=2.4, end=2.8, speaker="S2", conf=0.91, overlap=False),
    ]

    utterances = group_into_utterances(words, max_gap_sec=1.0)

    assert len(utterances) == 2
    assert utterances[0]["speaker"] == "S1"
    assert len(utterances[0]["words"]) == 2
    assert utterances[1]["speaker"] == "S2"
    assert len(utterances[1]["words"]) == 2


def test_group_into_utterances_large_gap():
    """Test utterance splitting on large gaps."""
    from insightsvc.schemas import Word

    words = [
        Word(text="Hello", start=0.0, end=0.5, speaker="S1", conf=0.95, overlap=False),
        Word(text="world", start=5.0, end=5.5, speaker="S1", conf=0.92, overlap=False),
    ]

    utterances = group_into_utterances(words, max_gap_sec=1.0)

    # Should split into 2 utterances due to large gap
    assert len(utterances) == 2
