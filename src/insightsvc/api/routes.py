"""API routes for analyze, health check, and artifacts.

Defines all HTTP endpoints for the service.
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status
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
    request: AnalyzeRequest | None = None,
    file: UploadFile | None = File(default=None),
    expected_speakers: Annotated[int | None, Form()] = None,
    language_hint: Annotated[str | None, Form()] = None,
    enable_overlaps: Annotated[bool, Form()] = True,
    return_word_confidence: Annotated[bool, Form()] = True,
) -> AnalyzeResponse:
    """Analyze audio file for speaker diarization, transcription, and mood.

    Supports both JSON request with audio_uri and multipart file upload.

    Args:
        request: JSON request body (for audio_uri).
        file: Uploaded audio file.
        expected_speakers: Expected number of speakers.
        language_hint: Language hint (ISO 639-1).
        enable_overlaps: Enable overlap detection.
        return_word_confidence: Include word confidences.

    Returns:
        Complete analysis result with transcript, summary, and mood.

    Raises:
        AnalysisError: If analysis fails.
    """
    settings = get_settings()
    job_id = uuid.uuid4()

    logger.info("Starting analysis job", job_id=str(job_id))
    requests_in_progress.inc()
    requests_total.labels(endpoint="/analyze", status="started").inc()

    start_time = time.time()

    try:
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

            req = AnalyzeRequest(
                audio_uri=audio_uri,
                expected_speakers=expected_speakers,
                language_hint=language_hint,
                enable_overlaps=enable_overlaps,
                return_word_confidence=return_word_confidence,
                summarizer=SummarizerConfig(),
                sarcasm_sensitivity="balanced",
            )
        elif request is not None:
            req = request
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
                "diarization": ModelInfo(name="pyannote/speaker-diarization-3.1", version=None, config={}),
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
        )
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
