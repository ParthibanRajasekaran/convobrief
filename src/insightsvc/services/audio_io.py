"""Audio I/O operations: loading, resampling, and normalization.

Handles various audio formats (wav, mp3, m4a) and prepares audio
for processing pipeline.
"""

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment

from insightsvc.api.app import AnalysisError
from insightsvc.logging import get_logger
from insightsvc.schemas import ErrorCode

logger = get_logger(__name__)


def load_audio(
    audio_uri: str,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file from URI and preprocess.

    Args:
        audio_uri: Path or URI to audio file.
        target_sr: Target sample rate (Hz).
        mono: Convert to mono.
        normalize: Normalize audio to [-1, 1].

    Returns:
        Tuple of (audio_samples, sample_rate).

    Raises:
        AnalysisError: If audio loading fails.
    """
    logger.info("Loading audio", audio_uri=audio_uri, target_sr=target_sr, mono=mono)

    try:
        # Handle different URI schemes
        if audio_uri.startswith("s3://"):
            # TODO: Implement S3 download
            raise NotImplementedError("S3 URIs not yet supported")
        elif audio_uri.startswith(("http://", "https://")):
            # TODO: Implement HTTP download
            raise NotImplementedError("HTTP URIs not yet supported")
        else:
            # Local file
            audio_path = Path(audio_uri)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_uri}")

            # Load based on extension
            if audio_path.suffix.lower() in [".wav", ".flac"]:
                audio, sr = _load_with_soundfile(audio_path, target_sr, mono, normalize)
            elif audio_path.suffix.lower() in [".mp3", ".m4a", ".aac", ".ogg"]:
                audio, sr = _load_with_pydub(audio_path, target_sr, mono, normalize)
            else:
                # Fallback to librosa
                audio, sr = _load_with_librosa(audio_path, target_sr, mono, normalize)

        duration = len(audio) / sr
        logger.info(
            "Audio loaded",
            duration_sec=duration,
            sample_rate=sr,
            channels="mono" if mono else "stereo",
            shape=audio.shape,
        )

        return audio, sr

    except Exception as e:
        logger.exception("Failed to load audio", audio_uri=audio_uri, error=str(e))
        raise AnalysisError(
            code=ErrorCode.AUDIO_LOAD_ERROR,
            message=f"Failed to load audio: {str(e)}",
            hint="Ensure file exists and format is supported (wav, mp3, m4a, flac)",
            recoverable=False,
        )


def _load_with_soundfile(
    path: Path,
    target_sr: int,
    mono: bool,
    normalize: bool,
) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile."""
    audio, sr = sf.read(str(path), dtype="float32")

    # Convert to mono if needed
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normalize
    if normalize:
        audio = _normalize_audio(audio)

    return audio, sr


def _load_with_pydub(
    path: Path,
    target_sr: int,
    mono: bool,
    normalize: bool,
) -> Tuple[np.ndarray, int]:
    """Load audio using pydub (for mp3, m4a, etc.)."""
    # Load with pydub
    audio_segment = AudioSegment.from_file(str(path))

    # Convert to mono if needed
    if mono:
        audio_segment = audio_segment.set_channels(1)

    # Resample
    audio_segment = audio_segment.set_frame_rate(target_sr)

    # Convert to numpy array
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    # Normalize based on bit depth
    bit_depth = audio_segment.sample_width * 8
    max_value = 2 ** (bit_depth - 1)
    audio = samples / max_value

    # Normalize
    if normalize:
        audio = _normalize_audio(audio)

    return audio, target_sr


def _load_with_librosa(
    path: Path,
    target_sr: int,
    mono: bool,
    normalize: bool,
) -> Tuple[np.ndarray, int]:
    """Load audio using librosa (fallback)."""
    audio, sr = librosa.load(str(path), sr=target_sr, mono=mono)

    if normalize:
        audio = _normalize_audio(audio)

    return audio, sr


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1].

    Args:
        audio: Audio samples.

    Returns:
        Normalized audio.
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def save_audio(
    audio: np.ndarray,
    path: Path,
    sample_rate: int = 16000,
) -> None:
    """Save audio to file.

    Args:
        audio: Audio samples.
        path: Output path.
        sample_rate: Sample rate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    logger.info("Audio saved", path=str(path))


def validate_audio_duration(
    audio: np.ndarray,
    sample_rate: int,
    min_duration: float,
    max_duration: float,
) -> None:
    """Validate audio duration is within bounds.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate.
        min_duration: Minimum duration (seconds).
        max_duration: Maximum duration (seconds).

    Raises:
        AnalysisError: If duration out of bounds.
    """
    duration = len(audio) / sample_rate

    if duration < min_duration:
        raise AnalysisError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Audio too short: {duration:.1f}s (minimum: {min_duration}s)",
            hint="Provide longer audio file",
            recoverable=True,
        )

    if duration > max_duration:
        raise AnalysisError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Audio too long: {duration:.1f}s (maximum: {max_duration}s)",
            hint="Split audio into shorter segments",
            recoverable=True,
        )
