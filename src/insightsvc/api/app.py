"""FastAPI application factory.

Creates and configures the FastAPI application with middleware,
error handlers, and lifecycle management.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from insightsvc import __version__
from insightsvc.config import get_settings
from insightsvc.logging import get_logger, setup_logging
from insightsvc.schemas import ErrorCode, ErrorResponse

# Setup logging first
setup_logging()
logger = get_logger(__name__)


class AnalysisError(Exception):
    """Custom exception for analysis errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        hint: str | None = None,
        recoverable: bool = False,
    ):
        """Initialize analysis error.

        Args:
            code: Error code.
            message: Error message.
            hint: Hint for resolution.
            recoverable: Whether error is recoverable.
        """
        self.code = code
        self.message = message
        self.hint = hint
        self.recoverable = recoverable
        super().__init__(message)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle.

    Args:
        app: FastAPI application instance.

    Yields:
        None during application runtime.
    """
    settings = get_settings()
    logger.info(
        "Starting AI Conversation Insights Service",
        version=__version__,
        device=settings.device,
        artifacts_dir=str(settings.artifacts_dir),
    )

    # TODO: Pre-load models if needed
    # This can speed up first request at the cost of longer startup

    yield

    logger.info("Shutting down AI Conversation Insights Service")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    # Settings loaded via lifespan context manager
    app = FastAPI(
        title="AI Conversation Insights Service",
        description="Production-ready speaker diarization, ASR, and mood analysis service",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Register routes
    from insightsvc.api.routes import router

    app.include_router(router)

    # Mount Prometheus metrics at /metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Error handlers
    @app.exception_handler(AnalysisError)
    async def analysis_error_handler(request: Request, exc: AnalysisError) -> JSONResponse:
        """Handle analysis errors."""
        logger.error(
            "Analysis error",
            code=exc.code.value,
            message=exc.message,
            hint=exc.hint,
            recoverable=exc.recoverable,
        )

        return JSONResponse(
            status_code=(
                status.HTTP_400_BAD_REQUEST
                if exc.code == ErrorCode.INVALID_INPUT
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content=ErrorResponse(
                code=exc.code,
                message=exc.message,
                hint=exc.hint,
                recoverable=exc.recoverable,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors."""
        logger.warning("Validation error", errors=exc.errors())

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                code=ErrorCode.INVALID_INPUT,
                message="Invalid request parameters",
                hint=str(exc.errors()[0]) if exc.errors() else None,
                recoverable=True,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        logger.exception("Unhandled exception", error=str(exc), error_type=type(exc).__name__)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                code=ErrorCode.INTERNAL_ERROR,
                message="Internal server error",
                hint="Please contact support if this persists",
                recoverable=False,
            ).model_dump(),
        )

    # Add middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests."""
        logger.info(
            "Request received",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )

        response = await call_next(request)

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        return response

    return app


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "insightsvc.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
