# Contributing to AI Conversation Insights Service

This guide describes the development workflow, testing strategies, and best practices for contributing to this project.

---

## 🏗️ Initial Setup (One Time)

```bash
# 1. Install dependencies
poetry install

# 2. Configure environment
cp .env.example .env
# Edit .env and add your HF_TOKEN from https://huggingface.co/settings/tokens

# 3. Generate test samples
python tests/data/generate_samples.py

# 4. Verify setup
python scripts/demo.py
```

---

## 🔄 Daily Development Loop

### 1. Choose a Task

```bash
# Check what needs to be done
cat CHECKLIST.md | grep -E "(Week 1|🔴)"

# Read implementation details
cat IMPLEMENTATION_GUIDE.md | less
```

### 2. Create a Branch (Optional)

```bash
git checkout -b feature/diarization-model
```

### 3. Implement

Follow the SOLID principles and type hints. Example workflow for adding a model:

```bash
# Open the file
code src/insightsvc/models/hf_diarize.py

# Check the interface
cat src/insightsvc/models/base.py | grep -A 20 "class DiarizationModel"
```

**Implementation Template:**
```python
"""Diarization model implementation."""

from pathlib import Path
from typing import Optional

import torch
from pyannote.audio import Pipeline

from insightsvc.models.base import DiarizationModel, DiarizationResult, Segment
from insightsvc.config import Settings
from insightsvc.logging import get_logger

log = get_logger()


class HuggingFaceDiarizer(DiarizationModel):
    """Pyannote-based speaker diarization."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = Pipeline.from_pretrained(
            settings.diarization_model_name,
            use_auth_token=settings.hf_token
        )
        if settings.device == "cuda":
            self.pipeline.to(torch.device("cuda"))
        log.info(
            "diarization_model_loaded",
            model=settings.diarization_model_name,
            device=settings.device
        )

    async def diarize(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None
    ) -> DiarizationResult:
        """Perform speaker diarization."""
        log.info("diarization_started", audio_path=str(audio_path))
        
        # Your implementation here
        # ...
        
        log.info("diarization_complete", num_speakers=len(speakers))
        return DiarizationResult(speakers=speakers, segments=segments)
```

### 4. Run Type Checking

```bash
# Check types
make lint

# Or manually
poetry run mypy src/insightsvc
```

### 5. Write Tests

```bash
# Create test file
code tests/unit/test_diarize.py
```

**Test Template:**
```python
"""Tests for diarization model."""

import pytest
from pathlib import Path

from insightsvc.models.hf_diarize import HuggingFaceDiarizer
from insightsvc.config import Settings


@pytest.mark.asyncio
async def test_diarize_two_speakers(settings: Settings) -> None:
    """Test diarization with two speakers."""
    model = HuggingFaceDiarizer(settings)
    audio_path = Path("tests/data/sample_2speakers_clean.wav")
    
    result = await model.diarize(audio_path, num_speakers=2)
    
    assert len(result.speakers) == 2
    assert len(result.segments) > 0
    assert all(s.speaker in ["S1", "S2"] for s in result.segments)
```

### 6. Run Tests

```bash
# Run specific test
pytest tests/unit/test_diarize.py -v

# Run all tests
make test

# Run with coverage
make coverage
```

### 7. Test with Sample Audio

```bash
# Update test_with_samples.py to use your model
code tests/test_with_samples.py

# Run sample tests
python tests/test_with_samples.py
```

### 8. Commit

```bash
git add .
git commit -m "feat: implement pyannote diarization model"
```

---

## 🧪 Testing Strategies

### Unit Tests (Fast, Isolated)

Test individual components in isolation:

```bash
# Test a single function
pytest tests/unit/test_align.py::test_assign_speakers_basic -v

# Test a single file
pytest tests/unit/test_align.py -v

# Test with specific markers
pytest -m "not slow" -v
```

### Integration Tests (Slower, End-to-End)

Test the full pipeline:

```bash
# Run integration tests
pytest tests/integration/ -v

# Test with real audio
pytest tests/integration/test_pipeline.py -v
```

### Manual Testing (API)

Test the HTTP API:

```bash
# Terminal 1: Start server
make dev

# Terminal 2: Test endpoints
curl http://localhost:8000/healthz

# Upload a file
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  -F "expected_speakers=2"

# Check metrics
curl http://localhost:8000/metrics
```

---

## 🐛 Debugging

### Enable Debug Logging

```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or environment variable
LOG_LEVEL=DEBUG python scripts/demo.py
```

### Use Python Debugger

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or Python 3.7+
breakpoint()
```

### Check Logs

```bash
# View structured logs
tail -f logs/app.log | jq .

# Filter by level
tail -f logs/app.log | jq 'select(.level == "error")'

# Filter by event
tail -f logs/app.log | jq 'select(.event == "diarization_started")'
```

### Profile Performance

```python
import time
start = time.perf_counter()
# ... code ...
elapsed = time.perf_counter() - start
log.info("operation_timed", elapsed_sec=elapsed)
```

---

## 📊 Metrics & Monitoring

### View Metrics

```bash
# Start server
make dev

# In another terminal
curl http://localhost:8000/metrics
```

### Key Metrics

- `analyze_requests_total` - Total requests
- `analyze_duration_seconds` - Request duration
- `model_inference_duration_seconds{model="diarization"}` - Model timing
- `audio_duration_seconds` - Input audio duration
- `process_info` - Python version, platform

### Prometheus + Grafana (Optional)

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up

# Access Grafana
open http://localhost:3000
```

---

## 🚀 Deployment

### Local Development

```bash
# Run with hot reload
make dev

# Or manually
poetry run uvicorn insightsvc.api.app:create_app --factory --reload
```

### Docker

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or with docker-compose
docker-compose up --build
```

### Production

```bash
# Build production image
docker build -t insightsvc:latest .

# Run with resource limits
docker run -d \
  --name insightsvc \
  -p 8000:8000 \
  --memory=4g \
  --cpus=2 \
  -e HF_TOKEN=$HF_TOKEN \
  insightsvc:latest
```

---

## 📦 Adding Dependencies

### Runtime Dependency

```bash
# Add to main dependencies
poetry add package-name

# Or with version constraint
poetry add "package-name>=1.0,<2.0"
```

### Development Dependency

```bash
# Add to dev dependencies
poetry add --group dev package-name
```

### Update Lock File

```bash
# Update all dependencies
poetry update

# Update specific package
poetry update package-name
```

---

## 🎯 Common Tasks

### Add a New Model

1. Check `src/insightsvc/models/base.py` for interface
2. Create `src/insightsvc/models/hf_<model>.py`
3. Implement interface methods
4. Add configuration to `src/insightsvc/config.py`
5. Write tests in `tests/unit/test_<model>.py`
6. Wire up in `src/insightsvc/pipelines/analyze_meeting.py`

### Add a New API Endpoint

1. Add route in `src/insightsvc/api/routes.py`
2. Add schemas in `src/insightsvc/schemas.py` if needed
3. Add tests in `tests/integration/test_api.py`
4. Update API documentation

### Add a New Service

1. Create `src/insightsvc/services/<service>.py`
2. Define interface (class or functions)
3. Add configuration if needed
4. Write unit tests
5. Use in pipeline

### Update Configuration

1. Add field to `src/insightsvc/config.py` Settings class
2. Add to `.env.example` with description
3. Add validator if needed
4. Update documentation

---

## 📋 Code Review Checklist

Before submitting code:

- [ ] Type hints on all functions
- [ ] Docstrings on all public functions
- [ ] Unit tests with >80% coverage
- [ ] No linting errors (`make lint`)
- [ ] No type errors (`make typecheck`)
- [ ] Tests pass (`make test`)
- [ ] Structured logging (not print statements)
- [ ] Error handling with custom exceptions
- [ ] Configuration via Settings (not hardcoded)
- [ ] PII redaction where applicable
- [ ] Metrics for key operations

---

## 🆘 Getting Help

### Documentation

- `README.md` - Architecture overview
- `IMPLEMENTATION_GUIDE.md` - Detailed implementation instructions
- `QUICK_START.md` - Getting started guide
- `CHECKLIST.md` - Task tracking
- `PROJECT_STATUS.md` - Current status

### Example Code

- `src/insightsvc/services/align.py` - Complete service
- `src/insightsvc/services/audio_io.py` - Complete service
- `tests/unit/test_align.py` - Test examples

### External Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/latest/)
- [Pyannote Docs](https://github.com/pyannote/pyannote-audio)
- [Whisper Docs](https://github.com/openai/whisper)
- [Transformers Docs](https://huggingface.co/docs/transformers/)

---

**Happy coding!** 🚀
