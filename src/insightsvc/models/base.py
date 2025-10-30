"""Base interfaces for all ML models.

Defines abstract base classes that all model implementations must follow,
enabling dependency injection and testability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ModelMetadata:
    """Metadata about a loaded model."""

    name: str
    version: str | None
    config: dict[str, Any]
    device: str


@dataclass
class DiarizationSegment:
    """Single diarization segment with speaker label."""

    speaker: str
    start: float
    end: float
    score: float
    overlap_ok: bool = False


@dataclass
class ASRWord:
    """Single word from ASR with timing and confidence."""

    text: str
    start: float
    end: float
    confidence: float


@dataclass
class ASRResult:
    """Complete ASR output."""

    words: list[ASRWord]
    language: str | None
    language_prob: float | None


@dataclass
class EmotionResult:
    """Audio emotion analysis result."""

    valence: float  # 0-1, negative to positive
    arousal: float  # 0-1, calm to excited
    dominance: float | None = None  # 0-1, submissive to dominant
    confidence: float = 1.0


@dataclass
class SentimentResult:
    """Text sentiment analysis result."""

    label: str  # negative, neutral, positive
    scores: dict[str, float]  # label -> probability
    confidence: float


@dataclass
class SarcasmResult:
    """Sarcasm detection result."""

    is_sarcastic: bool
    probability: float
    rationale: str | None = None


class AudioModel(ABC):
    """Base class for audio processing models."""

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata.

        Returns:
            Model metadata including name, version, and config.
        """
        pass

    @abstractmethod
    def to(self, device: str) -> "AudioModel":
        """Move model to device.

        Args:
            device: Target device (cuda or cpu).

        Returns:
            Self for chaining.
        """
        pass


class DiarizationModel(AudioModel):
    """Interface for speaker diarization models."""

    @abstractmethod
    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: int | None = None,
        min_speakers: int = 2,
        max_speakers: int = 5,
    ) -> list[DiarizationSegment]:
        """Perform speaker diarization on audio.

        Args:
            audio: Audio samples (mono).
            sample_rate: Audio sample rate in Hz.
            num_speakers: Expected number of speakers (None for auto-detect).
            min_speakers: Minimum speakers to detect.
            max_speakers: Maximum speakers to detect.

        Returns:
            List of diarization segments with speaker labels and timestamps.
        """
        pass


class ASRModel(AudioModel):
    """Interface for automatic speech recognition models."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        beam_size: int = 5,
        temperatures: list[float] | None = None,
    ) -> ASRResult:
        """Transcribe audio to text with word-level timestamps.

        Args:
            audio: Audio samples (mono).
            sample_rate: Audio sample rate in Hz.
            language: Language hint (ISO 639-1 code).
            beam_size: Beam size for decoding.
            temperatures: Temperature fallback sequence.

        Returns:
            ASR result with words, timestamps, and language detection.
        """
        pass


class EmotionModel(AudioModel):
    """Interface for audio emotion recognition models."""

    @abstractmethod
    def predict_emotion(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> EmotionResult:
        """Predict emotion from audio.

        Args:
            audio: Audio samples (mono).
            sample_rate: Audio sample rate in Hz.

        Returns:
            Emotion result with valence, arousal, and confidence.
        """
        pass


class TextModel(ABC):
    """Base class for text processing models."""

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata.

        Returns:
            Model metadata including name, version, and config.
        """
        pass

    @abstractmethod
    def to(self, device: str) -> "TextModel":
        """Move model to device.

        Args:
            device: Target device (cuda or cpu).

        Returns:
            Self for chaining.
        """
        pass


class SentimentModel(TextModel):
    """Interface for text sentiment analysis models."""

    @abstractmethod
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text.

        Args:
            text: Input text.

        Returns:
            Sentiment result with label, scores, and confidence.
        """
        pass


class SarcasmModel(TextModel):
    """Interface for sarcasm detection models."""

    @abstractmethod
    def detect_sarcasm(
        self,
        text: str,
        context: list[str] | None = None,
    ) -> SarcasmResult:
        """Detect sarcasm in text.

        Args:
            text: Input text to analyze.
            context: Optional conversation context (previous utterances).

        Returns:
            Sarcasm result with prediction, probability, and rationale.
        """
        pass


class SummarizerModel(TextModel):
    """Interface for text summarization models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Stop sequences for generation.

        Returns:
            Generated text.
        """
        pass
