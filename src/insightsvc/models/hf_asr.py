"""Whisper-based ASR implementation.

Implements automatic speech recognition using OpenAI Whisper with word-level timestamps.
"""

from typing import Any

import numpy as np
import torch
import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from insightsvc.logging import get_logger
from insightsvc.models.base import ASRModel, ASRResult, ASRWord, ModelMetadata

logger = get_logger(__name__)


class WhisperASR(ASRModel):
    """Whisper ASR implementation with word-level timestamps."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: str = "cuda",
        use_transformers: bool = True,
    ):
        """Initialize Whisper ASR model.

        Args:
            model_name: Hugging Face model name or whisper model size.
            device: Device for inference (cuda or cpu).
            use_transformers: Use HF transformers (True) or openai-whisper (False).
        """
        self.model_name = model_name
        self.device_name = device
        self.use_transformers = use_transformers

        logger.info(
            "Loading ASR model",
            model_name=model_name,
            device=device,
            backend="transformers" if use_transformers else "whisper",
        )

        if use_transformers:
            self._load_transformers_model()
        else:
            self._load_whisper_model()

        logger.info("ASR model loaded", model_name=model_name)

    def _load_transformers_model(self) -> None:
        """Load model using HF transformers."""
        torch_dtype = torch.float16 if self.device_name == "cuda" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device_name)

        processor = AutoProcessor.from_pretrained(self.model_name)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device_name,
            return_timestamps="word",
        )
        self.model = model
        self.processor = processor

    def _load_whisper_model(self) -> None:
        """Load model using openai-whisper."""
        model_size = self.model_name.split("/")[-1].replace("whisper-", "")
        self.model = whisper.load_model(model_size, device=self.device_name)
        self.pipe = None
        self.processor = None

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        config: dict[str, Any] = {}
        version = None

        if self.use_transformers and hasattr(self.model, "config"):
            config = self.model.config.to_dict()
            if hasattr(self.model.config, "_commit_hash"):
                version = self.model.config._commit_hash

        return ModelMetadata(
            name=self.model_name,
            version=version,
            config=config,
            device=self.device_name,
        )

    def to(self, device: str) -> "WhisperASR":
        """Move model to device."""
        self.device_name = device
        if self.model is not None:
            self.model.to(device)
        return self

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        beam_size: int = 5,
        temperatures: list[float] | None = None,
    ) -> ASRResult:
        """Transcribe audio with word-level timestamps.

        Args:
            audio: Audio samples (mono).
            sample_rate: Audio sample rate.
            language: Language hint.
            beam_size: Beam size for decoding.
            temperatures: Temperature fallback sequence.

        Returns:
            ASR result with words and timestamps.
        """
        if temperatures is None:
            temperatures = [0.0, 0.2, 0.4]

        logger.info(
            "Starting ASR transcription",
            duration_sec=len(audio) / sample_rate,
            language=language,
            beam_size=beam_size,
        )

        if self.use_transformers:
            result = self._transcribe_transformers(audio, sample_rate, language)
        else:
            result = self._transcribe_whisper(audio, sample_rate, language, beam_size, temperatures)

        logger.info(
            "ASR transcription complete",
            word_count=len(result.words),
            detected_language=result.language,
        )

        return result

    def _transcribe_transformers(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None,
    ) -> ASRResult:
        """Transcribe using HF transformers pipeline."""
        generate_kwargs = {"task": "transcribe", "return_timestamps": "word"}
        if language:
            generate_kwargs["language"] = language

        result = self.pipe(
            audio.astype(np.float32),
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )

        # Extract words with timestamps
        words = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                if "timestamp" in chunk and chunk["timestamp"]:
                    start, end = chunk["timestamp"]
                    # Handle None timestamps
                    if start is None:
                        start = words[-1].end if words else 0.0
                    if end is None:
                        end = start + 0.5

                    words.append(
                        ASRWord(
                            text=chunk["text"].strip(),
                            start=float(start),
                            end=float(end),
                            confidence=1.0,  # TODO: Extract confidence from logits
                        )
                    )

        return ASRResult(
            words=words,
            language=generate_kwargs.get("language"),
            language_prob=None,
        )

    def _transcribe_whisper(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None,
        beam_size: int,
        temperatures: list[float],
    ) -> ASRResult:
        """Transcribe using openai-whisper with temperature fallback."""
        # Normalize audio to [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Resample if needed
        if sample_rate != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Try each temperature until we get good results
        for temp in temperatures:
            try:
                result = self.model.transcribe(
                    audio,
                    language=language,
                    beam_size=beam_size,
                    temperature=temp,
                    word_timestamps=True,
                    verbose=False,
                )

                # Extract words
                words = []
                if "segments" in result:
                    for segment in result["segments"]:
                        if "words" in segment:
                            for word_info in segment["words"]:
                                words.append(
                                    ASRWord(
                                        text=word_info["word"].strip(),
                                        start=word_info["start"],
                                        end=word_info["end"],
                                        confidence=word_info.get("probability", 1.0),
                                    )
                                )

                return ASRResult(
                    words=words,
                    language=result.get("language"),
                    language_prob=None,
                )

            except Exception as e:
                logger.warning("ASR failed at temperature", temperature=temp, error=str(e))
                if temp == temperatures[-1]:
                    raise

        return ASRResult(words=[], language=None, language_prob=None)
