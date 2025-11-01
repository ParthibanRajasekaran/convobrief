"""API routes for analyze, health check, and artifacts.

Defines all HTTP endpoints for the service.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from insightsvc import __version__
from insightsvc.api.app import AnalysisError
from insightsvc.config import get_settings
from insightsvc.logging import get_logger
from insightsvc.metrics import requests_in_progress, requests_total
from insightsvc.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorCode,
    HealthResponse,
    ParticipantMood,
    SimpleAnalysisResponse,
)

logger = get_logger(__name__)
router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status with version and model status.
    """
    # TODO: Check if models are loaded
    models_loaded = True

    return HealthResponse(
        status="healthy",
        version=__version__,
        models_loaded=models_loaded,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
)
async def analyze_audio(
    request: Annotated[str | None, Form()] = None,
    file: UploadFile | None = File(default=None),
    expected_speakers: Annotated[int | None, Form()] = None,
    language_hint: Annotated[str | None, Form()] = None,
    enable_overlaps: Annotated[bool, Form()] = True,
    return_word_confidence: Annotated[bool, Form()] = True,
) -> AnalyzeResponse:
    """Analyze audio file for speaker diarization, transcription, and mood.

    Supports both JSON request with audio_uri and multipart file upload.

    Args:
        request: JSON string containing AnalyzeRequest data (for audio_uri and advanced options).
        file: Uploaded audio file.
        expected_speakers: Expected number of speakers.
        language_hint: Language hint (ISO 639-1).
        enable_overlaps: Enable overlap detection.
        return_word_confidence: Include word confidences.

    Returns:
        Complete analysis result with transcript, summary, and mood.

    Raises:
        HTTPException: If request JSON is invalid.
        AnalysisError: If analysis fails.
    """
    settings = get_settings()
    job_id = uuid.uuid4()

    logger.info("Starting analysis job", job_id=str(job_id))
    requests_in_progress.inc()
    requests_total.labels(endpoint="/analyze", status="started").inc()

    start_time = time.time()

    try:
        # Parse request JSON string if provided
        parsed_request: AnalyzeRequest | None = None
        if request:
            try:
                request_data = json.loads(request)
                parsed_request = AnalyzeRequest(**request_data)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in 'request' field: {str(e)}",
                ) from e
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request data: {str(e)}",
                ) from e

        # Handle both JSON and form upload
        if file is not None:
            # File upload
            logger.info("Processing uploaded file", filename=file.filename, job_id=str(job_id))

            # Save uploaded file temporarily
            temp_dir = settings.artifacts_dir / str(job_id) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            audio_path = temp_dir / (file.filename or "audio")

            with open(audio_path, "wb") as f:
                content = await file.read()
                f.write(content)

            audio_uri = str(audio_path)

            # Build request from form data
            from insightsvc.schemas import SummarizerConfig

            # Merge parsed_request data with form parameters if provided
            if parsed_request:
                # Use values from parsed_request but allow form fields to override
                req = AnalyzeRequest.model_construct(
                    audio_uri=audio_uri,
                    expected_speakers=expected_speakers or parsed_request.expected_speakers,
                    language_hint=language_hint or parsed_request.language_hint,
                    enable_overlaps=(
                        enable_overlaps
                        if enable_overlaps is not None
                        else parsed_request.enable_overlaps
                    ),
                    return_word_confidence=(
                        return_word_confidence
                        if return_word_confidence is not None
                        else parsed_request.return_word_confidence
                    ),
                    summarizer=parsed_request.summarizer,
                    sarcasm_sensitivity=parsed_request.sarcasm_sensitivity,
                )
            else:
                # Use model_construct to bypass validation for local file paths
                req = AnalyzeRequest.model_construct(
                    audio_uri=audio_uri,
                    expected_speakers=expected_speakers,
                    language_hint=language_hint,
                    enable_overlaps=enable_overlaps,
                    return_word_confidence=return_word_confidence,
                    summarizer=SummarizerConfig(),
                    sarcasm_sensitivity="balanced",
                )
        elif parsed_request is not None:
            req = parsed_request
            audio_uri = req.audio_uri
        else:
            raise AnalysisError(
                code=ErrorCode.INVALID_INPUT,
                message="Either 'request' body or 'file' upload must be provided",
                hint="Use JSON with audio_uri or multipart file upload",
                recoverable=True,
            )

        # TODO: Implement full pipeline
        # 1. Load audio
        # 2. Run VAD
        # 3. Run diarization
        # 4. Run ASR
        # 5. Align words to speakers
        # 6. Run NLP (sentiment, emotion, sarcasm)
        # 7. Generate summary
        # 8. Write artifacts

        # Placeholder response
        from insightsvc.schemas import (
            Artifacts,
            Metrics,
            ModelInfo,
            Mood,
            Summary,
            Transcript,
        )

        result = AnalyzeResponse(
            job_id=job_id,
            models={
                "asr": ModelInfo(name=settings.asr_model_name, version=None, config={}),
                "diarization": ModelInfo(
                    name="pyannote/speaker-diarization-3.1", version=None, config={}
                ),
            },
            transcript=Transcript(
                words=[],
                utterances=[],
                speakers=[],
                duration_sec=0.0,
                detected_language=req.language_hint,
            ),
            summary=Summary(
                summary="TODO: Implement pipeline",
                decisions=[],
                action_items=[],
                disagreements=[],
                risks=[],
                open_questions=[],
            ),
            mood=Mood(per_speaker=[]),
            artifacts=Artifacts(
                transcript_json="",
                summary_json="",
                mood_json="",
                report_md="",
            ),
            metrics=Metrics(
                wer=None,
                der=None,
                processing_time_sec=time.time() - start_time,
                rtf=0.0,
            ),
            created_at=datetime.now(),
        )

        requests_total.labels(endpoint="/analyze", status="success").inc()
        logger.info(
            "Analysis job completed",
            job_id=str(job_id),
            duration_sec=time.time() - start_time,
        )

        return result

    except AnalysisError:
        requests_total.labels(endpoint="/analyze", status="error").inc()
        raise
    except Exception as e:
        requests_total.labels(endpoint="/analyze", status="error").inc()
        logger.exception("Analysis job failed", job_id=str(job_id), error=str(e))
        raise AnalysisError(
            code=ErrorCode.PROCESSING_ERROR,
            message=f"Analysis failed: {str(e)}",
            hint="Check logs for details",
            recoverable=False,
        ) from e
    finally:
        requests_in_progress.dec()


@router.get("/artifacts/{job_id}/{artifact_name}", tags=["Artifacts"])
async def get_artifact(job_id: str, artifact_name: str) -> FileResponse:
    """Retrieve generated artifact file.

    Args:
        job_id: Job ID from analysis response.
        artifact_name: Artifact file name (transcript.json, summary.json, mood.json, report.md).

    Returns:
        File content.

    Raises:
        AnalysisError: If file not found.
    """
    settings = get_settings()
    artifact_path = settings.artifacts_dir / job_id / artifact_name

    if not artifact_path.exists():
        raise AnalysisError(
            code=ErrorCode.STORAGE_ERROR,
            message=f"Artifact not found: {artifact_name}",
            hint="Check job_id and artifact_name",
            recoverable=False,
        )

    return FileResponse(
        path=artifact_path,
        media_type="application/json" if artifact_name.endswith(".json") else "text/markdown",
        filename=artifact_name,
    )


@router.post(
    "/analyze/simple",
    response_model=SimpleAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
)
async def analyze_audio_simple(
    file: UploadFile = File(..., description="Audio file to analyze"),
) -> SimpleAnalysisResponse:
    """Simplified audio analysis endpoint for quick insights.

    Upload an audio file and get immediate insights about:
    - Number of speakers detected
    - Each speaker's talk time, tone, mood, and sarcasm detection
    - Overall conversation mood and dynamics
    - High-level feedback summary

    This endpoint is designed for ease of use - just upload a file and get
    a clean, human-readable JSON response without complex nested structures.

    Args:
        file: Audio file (wav, mp3, m4a, flac).

    Returns:
        Simple analysis with speaker insights and conversation summary.

    Raises:
        HTTPException: If file processing fails or analysis cannot be completed.
    """
    logger.info("Starting simple analysis", filename=file.filename)
    start_time = time.time()

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="File must have a valid filename",
            )

        # Check file extension
        allowed_extensions = [".wav", ".mp3", ".m4a", ".flac"]
        file_ext = file.filename.lower().split(".")[-1]
        if f".{file_ext}" not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}",
            )

        # Read file content
        logger.info("Reading audio file", filename=file.filename)
        audio_content = await file.read()

        if len(audio_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        # TODO: Implement actual audio processing pipeline
        # For now, return placeholder data that demonstrates the response structure

        # Simulated analysis results
        # In production, this would call:
        # 1. VAD (Voice Activity Detection)
        # 2. Speaker diarization (pyannote)
        # 3. ASR (Whisper) with speaker attribution
        # 4. Emotion analysis (valence/arousal)
        # 5. Sentiment analysis
        # 6. Sarcasm detection
        # 7. LLM-based summarization

        logger.info("Analyzing speakers and mood", filename=file.filename)

        # Placeholder: Simulate detecting 2-3 speakers
        num_speakers = 2

        participants = [
            ParticipantMood(
                label="Person 1",
                talk_time_sec=120.5,
                tone="calm",
                mood="positive",
                sarcasm_detected=False,
                context_summary="Provided clear updates and maintained a professional, reassuring tone throughout the conversation.",
            ),
            ParticipantMood(
                label="Person 2",
                talk_time_sec=85.3,
                tone="assertive",
                mood="neutral",
                sarcasm_detected=True,
                context_summary="Offered direct feedback with occasional sarcasm. Focused on practical concerns and action items.",
            ),
        ]

        overall_mood = "positive and collaborative"
        feedback = "Participants maintained professional engagement with constructive tone variations. The conversation was productive with clear communication and mutual respect."

        processing_time = time.time() - start_time

        logger.info(
            "Simple analysis completed",
            filename=file.filename,
            speakers=num_speakers,
            duration_sec=processing_time,
        )

        return SimpleAnalysisResponse(
            speakers_detected=num_speakers,
            participants=participants,
            overall_conversation_mood=overall_mood,
            feedback_summary=feedback,
            processing_time_sec=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Simple analysis failed", filename=file.filename, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        ) from e
