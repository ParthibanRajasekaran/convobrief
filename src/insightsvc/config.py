"""Configuration management using Pydantic Settings.

Loads configuration from environment variables with validation and type safety.
All settings can be overridden via .env file or environment variables.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation and type safety.

    All settings can be configured via environment variables or .env file.
    Follows the 12-factor app methodology for configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core Configuration
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="Device for model inference (cuda or cpu)",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    artifacts_dir: Path = Field(
        default=Path("./artifacts"),
        description="Directory for storing output artifacts",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    # Hugging Face
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face API token for accessing gated models",
    )

    # Model Configuration
    diarization_backend: Literal["pyannote", "hybrid"] = Field(
        default="pyannote",
        description="Diarization backend (pyannote or hybrid fallback)",
    )
    asr_model_name: str = Field(
        default="openai/whisper-large-v3",
        description="ASR model name from Hugging Face",
    )
    asr_beam_size: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Beam size for ASR decoding",
    )
    asr_temperatures: list[float] = Field(
        default=[0.0, 0.2, 0.4],
        description="Temperature fallback sequence for ASR",
    )

    emotion_model_name: str = Field(
        default="audeering/wav2vec2-large-robust-24-ft-emotion-msp-dim",
        description="Audio emotion model",
    )
    sentiment_model_name: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="Text sentiment model",
    )
    sarcasm_model_name: str = Field(
        default="cardiffnlp/twitter-roberta-base-irony",
        description="Sarcasm detection model",
    )
    summarizer_model_name: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        description="LLM for summarization",
    )

    # NLP Configuration
    emotion_window_sec: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Window size for emotion analysis (seconds)",
    )
    sarcasm_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for sarcasm detection",
    )
    sarcasm_sensitivity: Literal["low", "balanced", "high"] = Field(
        default="balanced",
        description="Sarcasm detection sensitivity",
    )
    fusion_weights: list[float] = Field(
        default=[0.4, 0.3, 0.3],
        description="Fusion weights [audio_valence, text_sentiment, sarcasm]",
    )

    # Audio Processing
    min_audio_duration_sec: int = Field(
        default=10,
        ge=1,
        description="Minimum audio duration (seconds)",
    )
    max_audio_duration_sec: int = Field(
        default=10800,
        ge=60,
        description="Maximum audio duration (seconds, 3 hours)",
    )
    target_sample_rate: int = Field(
        default=16000,
        description="Target sample rate for audio processing (Hz)",
    )

    # VAD Configuration
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice activity detection threshold",
    )
    min_speech_duration_ms: int = Field(
        default=250,
        ge=100,
        description="Minimum speech segment duration (ms)",
    )
    min_silence_duration_ms: int = Field(
        default=100,
        ge=50,
        description="Minimum silence duration for segmentation (ms)",
    )

    # Diarization Configuration
    min_speakers: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum number of speakers",
    )
    max_speakers: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of speakers",
    )

    # S3 Configuration (optional)
    s3_bucket: str | None = Field(
        default=None,
        description="S3 bucket for remote storage",
    )
    s3_prefix: str = Field(
        default="insights/",
        description="S3 key prefix",
    )
    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS access key ID",
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS secret access key",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region",
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host binding",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API port",
    )
    max_upload_size_mb: int = Field(
        default=500,
        ge=1,
        le=2000,
        description="Maximum upload size (MB)",
    )

    # Security & PII
    redact_pii: bool = Field(
        default=True,
        description="Redact PII from logs",
    )
    log_transcript_text: bool = Field(
        default=False,
        description="Log transcript text (may contain PII)",
    )
    hash_audio_uris: bool = Field(
        default=True,
        description="Hash audio URIs in logs",
    )

    @field_validator("asr_temperatures", "fusion_weights", mode="before")
    @classmethod
    def parse_float_list(cls, v):
        """Parse comma-separated string or list to list of floats."""
        if isinstance(v, str):
            return [float(x.strip()) for x in v.split(",")]
        return v

    @field_validator("fusion_weights")
    @classmethod
    def validate_fusion_weights(cls, v: list[float]) -> list[float]:
        """Validate fusion weights sum to approximately 1.0."""
        if len(v) != 3:
            raise ValueError("fusion_weights must have exactly 3 elements")
        total = sum(v)
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"fusion_weights should sum to ~1.0, got {total}")
        return v

    @field_validator("max_speakers")
    @classmethod
    def validate_speaker_range(cls, v: int, info: ValidationInfo) -> int:
        """Ensure max_speakers >= min_speakers."""
        min_speakers = info.data.get("min_speakers", 2)
        if v < min_speakers:
            raise ValueError(f"max_speakers ({v}) must be >= min_speakers ({min_speakers})")
        return v

    def get_sarcasm_threshold_adjusted(self) -> float:
        """Get sarcasm threshold adjusted for sensitivity.

        Returns:
            Adjusted threshold based on sensitivity setting.
        """
        adjustments = {
            "low": 0.7,
            "balanced": 0.5,
            "high": 0.3,
        }
        return adjustments[self.sarcasm_sensitivity]


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings singleton.

    Returns:
        Settings instance with current configuration.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        # Ensure artifacts directory exists
        _settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return _settings
