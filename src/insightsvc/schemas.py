"""Pydantic schemas for API requests and responses.

All schemas use Pydantic v2 with strict type validation and comprehensive
documentation for API clients.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class SummarizerConfig(BaseModel):
    """Configuration for meeting summarization."""

    max_words: int = Field(
        default=250,
        ge=50,
        le=1000,
        description="Maximum words for summary",
    )
    style: str = Field(
        default="concise",
        pattern="^(concise|detailed|bullet)$",
        description="Summary style",
    )


class AnalyzeRequest(BaseModel):
    """Request schema for /analyze endpoint.

    Supports either audio_uri (S3/HTTP) or file upload via multipart form.
    """

    audio_uri: str | None = Field(
        default=None,
        description="URI to audio file (s3://, http://, https://)",
    )
    expected_speakers: int | None = Field(
        default=None,
        ge=2,
        le=5,
        description="Expected number of speakers (auto-detect if None)",
    )
    language_hint: str | None = Field(
        default=None,
        pattern="^[a-z]{2}$",
        description="ISO 639-1 language code hint",
    )
    enable_overlaps: bool = Field(
        default=True,
        description="Enable overlap detection for crosstalk",
    )
    return_word_confidence: bool = Field(
        default=True,
        description="Include confidence scores for each word",
    )
    summarizer: SummarizerConfig = Field(
        default_factory=SummarizerConfig,
        description="Summarization configuration",
    )
    sarcasm_sensitivity: str = Field(
        default="balanced",
        pattern="^(low|balanced|high)$",
        description="Sarcasm detection sensitivity",
    )

    @field_validator("audio_uri")
    @classmethod
    def validate_audio_uri(cls, v: str | None) -> str | None:
        """Validate audio URI format."""
        if v is None:
            return v
        # Allow local file paths (for uploaded files) or remote URIs
        if v.startswith(("s3://", "http://", "https://", "/", ".", "artifacts/")):
            return v
        raise ValueError("audio_uri must be a valid file path or URI (s3://, http://, https://)")


class ModelInfo(BaseModel):
    """Model metadata for reproducibility."""

    name: str = Field(description="Model name/identifier")
    version: str | None = Field(default=None, description="Model version or commit SHA")
    config: dict[str, Any] = Field(default_factory=dict, description="Model configuration")


class Word(BaseModel):
    """Single word with speaker attribution and timing."""

    text: str = Field(description="Word text")
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    speaker: str | None = Field(default=None, description="Speaker ID (e.g., S1, S2)")
    conf: float = Field(ge=0.0, le=1.0, description="Confidence score")
    overlap: bool = Field(default=False, description="Part of overlapping speech")

    @field_validator("end")
    @classmethod
    def validate_end_time(cls, v: float, info: ValidationInfo) -> float:
        """Ensure end >= start."""
        start = info.data.get("start", 0.0)
        if v < start:
            raise ValueError(f"end ({v}) must be >= start ({start})")
        return v


class Utterance(BaseModel):
    """Continuous utterance from a single speaker."""

    speaker: str = Field(description="Speaker ID")
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    text: str = Field(description="Utterance text")
    overlap: bool = Field(default=False, description="Overlaps with other speaker")
    confidence: float = Field(ge=0.0, le=1.0, description="Average word confidence")


class SpeakerStats(BaseModel):
    """Per-speaker statistics."""

    id: str = Field(description="Speaker ID")
    talk_time_sec: float = Field(ge=0.0, description="Total talk time in seconds")
    utterance_count: int = Field(ge=0, description="Number of utterances")
    avg_confidence: float = Field(ge=0.0, le=1.0, description="Average confidence")


class Transcript(BaseModel):
    """Complete transcript with speaker attribution."""

    words: list[Word] = Field(description="All words with timestamps and speakers")
    utterances: list[Utterance] = Field(description="Utterances grouped by speaker")
    speakers: list[SpeakerStats] = Field(description="Per-speaker statistics")
    duration_sec: float = Field(ge=0.0, description="Total audio duration")
    detected_language: str | None = Field(default=None, description="Detected language code")


class SarcasmInfo(BaseModel):
    """Sarcasm detection result."""

    is_sarcastic: bool = Field(description="Whether utterance is sarcastic")
    prob: float = Field(ge=0.0, le=1.0, description="Sarcasm probability")
    rationale: str | None = Field(default=None, description="Brief explanation")


class MoodTimepoint(BaseModel):
    """Mood snapshot at a specific time."""

    t: float = Field(ge=0.0, description="Timestamp in seconds")
    valence: float = Field(ge=0.0, le=1.0, description="Valence (negative to positive)")
    arousal: float = Field(ge=0.0, le=1.0, description="Arousal (calm to excited)")
    sentiment: str = Field(description="Text sentiment label")
    sarcasm: SarcasmInfo = Field(description="Sarcasm detection")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence spans")


class FinalMoodRating(BaseModel):
    """Final aggregated mood assessment."""

    valence: float = Field(ge=0.0, le=1.0, description="Overall valence")
    label: str = Field(description="Human-readable mood label")
    confidence: float = Field(ge=0.0, le=1.0, description="Rating confidence")


class SpeakerMood(BaseModel):
    """Complete mood profile for a speaker."""

    speaker: str = Field(description="Speaker ID")
    timeline: list[MoodTimepoint] = Field(description="Mood over time")
    final_rating: FinalMoodRating = Field(description="Overall mood assessment")


class Mood(BaseModel):
    """Mood analysis for all speakers."""

    per_speaker: list[SpeakerMood] = Field(description="Mood profile per speaker")


class Decision(BaseModel):
    """Meeting decision with context."""

    text: str = Field(description="Decision text")
    timestamp: float = Field(ge=0.0, description="When decision was made (seconds)")
    speakers: list[str] = Field(description="Speakers involved")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")


class ActionItem(BaseModel):
    """Action item with owner and deadline."""

    text: str = Field(description="Action item description")
    owner: str | None = Field(default=None, description="Responsible speaker")
    due_date: str | None = Field(default=None, description="Due date if mentioned")
    timestamp: float = Field(ge=0.0, description="When item was mentioned (seconds)")


class Disagreement(BaseModel):
    """Detected disagreement between speakers."""

    text: str = Field(description="Disagreement description")
    speakers: list[str] = Field(description="Speakers in disagreement")
    timestamp: float = Field(ge=0.0, description="When disagreement occurred (seconds)")
    intensity: float = Field(ge=0.0, le=1.0, description="Disagreement intensity")


class Risk(BaseModel):
    """Identified risk or blocker."""

    text: str = Field(description="Risk description")
    category: str = Field(description="Risk category")
    timestamp: float = Field(ge=0.0, description="When risk was mentioned (seconds)")


class Summary(BaseModel):
    """Meeting summary with key points."""

    summary: str = Field(description="Concise meeting summary")
    decisions: list[Decision] = Field(default_factory=list, description="Decisions made")
    action_items: list[ActionItem] = Field(default_factory=list, description="Action items")
    disagreements: list[Disagreement] = Field(default_factory=list, description="Disagreements")
    risks: list[Risk] = Field(default_factory=list, description="Identified risks")
    open_questions: list[str] = Field(default_factory=list, description="Unresolved questions")


class Artifacts(BaseModel):
    """Paths to generated artifact files."""

    transcript_json: str = Field(description="Path to transcript.json")
    summary_json: str = Field(description="Path to summary.json")
    mood_json: str = Field(description="Path to mood.json")
    report_md: str = Field(description="Path to report.md")


class Metrics(BaseModel):
    """Analysis quality metrics."""

    wer: float | None = Field(
        default=None, ge=0.0, description="Word error rate (if ref available)"
    )
    der: float | None = Field(default=None, ge=0.0, description="Diarization error rate")
    processing_time_sec: float = Field(ge=0.0, description="Total processing time")
    rtf: float = Field(ge=0.0, description="Real-time factor (processing / audio duration)")


class AnalyzeResponse(BaseModel):
    """Response schema for /analyze endpoint."""

    job_id: UUID = Field(description="Unique job identifier")
    models: dict[str, ModelInfo] = Field(description="Model versions used")
    transcript: Transcript = Field(description="Speaker-attributed transcript")
    summary: Summary = Field(description="Meeting summary")
    mood: Mood = Field(description="Mood analysis")
    artifacts: Artifacts = Field(description="Artifact file paths")
    metrics: Metrics = Field(description="Analysis metrics")
    created_at: datetime = Field(description="Analysis timestamp")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    version: str = Field(description="Service version")
    models_loaded: bool = Field(description="Whether models are loaded")


class ErrorCode(str, Enum):
    """Standard error codes."""

    INVALID_INPUT = "invalid_input"
    AUDIO_LOAD_ERROR = "audio_load_error"
    PROCESSING_ERROR = "processing_error"
    MODEL_ERROR = "model_error"
    STORAGE_ERROR = "storage_error"
    INTERNAL_ERROR = "internal_error"


class ErrorResponse(BaseModel):
    """Error response schema."""

    code: ErrorCode = Field(description="Error code")
    message: str = Field(description="Error message")
    hint: str | None = Field(default=None, description="Suggestion for resolution")
    recoverable: bool = Field(default=False, description="Whether error is recoverable")
    job_id: UUID | None = Field(default=None, description="Job ID if available")
