# AI Conversation Insights Service

Production-ready Python 3.11 service for multi-speaker audio analysis with diarization, ASR, sentiment analysis, emotion detection, and meeting summarization.

## Features

- **Speaker Diarization**: 2-5 speakers with automatic detection (pyannote or hybrid fallback)
- **Speech Recognition**: Whisper-based ASR with word-level timestamps and confidences
- **Mood Analysis**: Audio emotion (valence/arousal) + text sentiment + sarcasm detection with fusion
- **Meeting Summarization**: LLM-powered summaries with decisions, action items, and disagreements
- **Robust**: Handles crosstalk, background noise, accents, and filler words
- **Observable**: Structured logging (JSON), Prometheus metrics, calibrated confidences
- **SOLID Architecture**: Clean boundaries, dependency injection, comprehensive type hints

## Architecture

```
VAD → Diarization → ASR (word timestamps) → Alignment → NLP (sentiment/emotion/sarcasm) → Summarization
```

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, CPU fallback available)
- Hugging Face token for pyannote models (optional)

### Installation

```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Or with S3 support
poetry install --extras s3

# Copy and configure environment
cp .env.example .env
# Edit .env with your HF_TOKEN and other settings
```

### Configuration

Create `.env` file:

```bash
# Core
DEVICE=cuda  # or cpu
LOG_LEVEL=INFO
ARTIFACTS_DIR=./artifacts

# Hugging Face (required for pyannote diarization)
HF_TOKEN=your_token_here

# Models
DIARIZATION_BACKEND=pyannote  # or hybrid
ASR_MODEL_NAME=openai/whisper-large-v3
ASR_BEAM_SIZE=5
ASR_TEMPERATURES=[0.0,0.2,0.4]

EMOTION_MODEL_NAME=audeering/wav2vec2-large-robust-24-ft-emotion-msp-dim
SENTIMENT_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
SARCASM_MODEL_NAME=cardiffnlp/twitter-roberta-base-irony
SUMMARIZER_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3

# NLP Config
EMOTION_WINDOW_SEC=10.0
SARCASM_THRESHOLD=0.5
FUSION_WEIGHTS=[0.4,0.3,0.3]  # audio_valence, text_sentiment, sarcasm

# S3 (optional)
# S3_BUCKET=my-bucket
# S3_PREFIX=insights/
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
```

### Run Service

```bash
# Development
make run

# Or directly
poetry run uvicorn insightsvc.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

# Docker
docker build -t insightsvc:latest .
docker run -p 8000:8000 --gpus all -v $(pwd)/.env:/app/.env insightsvc:latest
```

## Interactive Dashboard

**NEW**: We now provide a beautiful Streamlit-based dashboard for easy visualization!

```bash
# Install dashboard dependencies
poetry install --with dashboard

# Start backend (Terminal 1)
make run

# Start dashboard (Terminal 2)
poetry run streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser and enjoy:
- 📁 Drag-and-drop file upload
- 📋 Interactive summary with decisions, action items, and risks
- 💬 Timeline visualization of speaker diarization
- 🎭 Mood analysis with valence/arousal charts
- 📊 Quality metrics and performance stats
- 💾 Export results as JSON or Markdown

See [DASHBOARD.md](DASHBOARD.md) for detailed documentation.

## API Usage

### Analyze Meeting

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "audio_uri": "s3://bucket/meeting.m4a",
    "expected_speakers": null,
    "language_hint": "en",
    "enable_overlaps": true,
    "return_word_confidence": true,
    "summarizer": {"max_words": 250, "style": "concise"},
    "sarcasm_sensitivity": "balanced"
  }'
```

Or with file upload:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@meeting.wav" \
  -F "expected_speakers=3"
```

### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "models": {
    "asr": {"name": "openai/whisper-large-v3", "version": "..."},
    "diarization": {"name": "pyannote/speaker-diarization-3.1", "version": "..."}
  },
  "transcript": {
    "words": [...],
    "utterances": [...],
    "speakers": [{"id": "S1", "talk_time_sec": 321.4}, ...]
  },
  "summary": {
    "summary": "...",
    "decisions": [...],
    "action_items": [...],
    "disagreements": [...],
    "risks": [...]
  },
  "mood": {
    "per_speaker": [...]
  },
  "artifacts": {
    "transcript_json": "/artifacts/550e8400.../transcript.json",
    "summary_json": "/artifacts/550e8400.../summary.json",
    "mood_json": "/artifacts/550e8400.../mood.json",
    "report_md": "/artifacts/550e8400.../report.md"
  },
  "metrics": {"wer": 0.08, "der": 0.12}
}
```

## Outputs

1. **transcript.json**: Speaker-attributed words and utterances with timestamps/confidences
2. **summary.json**: Meeting summary, decisions, action items, questions, disagreements
3. **mood.json**: Per-speaker mood timeline + final rating (emotion, sentiment, sarcasm)
4. **report.md**: Human-readable report with timestamp links

## Development

```bash
# Setup
make setup

# Lint
make lint

# Type check
make typecheck

# Test
make test

# All checks
make check

# Benchmark
make bench
```

## Testing

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest --cov=src/insightsvc --cov-report=html
```

## Models & Licenses

All models use permissive licenses:

- **Whisper**: MIT (OpenAI)
- **Pyannote**: MIT (with HF token required)
- **Cardiff NLP**: MIT
- **Audeering Emotion**: CC-BY-4.0
- **Mistral/Llama**: Apache 2.0

## Performance

Target benchmarks:
- 60-min meeting on A10G: < 1.5× real-time
- 60-min meeting on CPU: < 5× real-time (batched)

## Observability

- **Logs**: Structured JSON via structlog
- **Metrics**: `/metrics` endpoint (Prometheus format)
- **Health**: `/healthz` endpoint

## Obtaining Pyannote Token

1. Create account at https://huggingface.co
2. Accept pyannote model licenses:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Generate token at https://huggingface.co/settings/tokens
4. Set `HF_TOKEN` in `.env`

If no token available, service falls back to hybrid diarization (VAD + speaker embeddings + clustering).

## Architecture Details

### SOLID Principles

- **Single Responsibility**: Each service handles one concern
- **Open/Closed**: Model interfaces allow extension without modification
- **Liskov Substitution**: All model implementations are interchangeable
- **Interface Segregation**: Clean boundaries between components
- **Dependency Inversion**: Depends on abstractions (ABCs), not implementations

### Project Structure

```
src/insightsvc/
├── config.py           # Pydantic Settings
├── schemas.py          # API request/response models
├── logging.py          # Structured logging
├── metrics.py          # Prometheus metrics
├── utils/
│   └── timecode.py     # Timestamp utilities
├── api/
│   ├── app.py          # FastAPI factory
│   └── routes.py       # Endpoints
├── services/
│   ├── audio_io.py     # Audio loading/resampling
│   ├── vad.py          # Voice activity detection
│   ├── diarization.py  # Speaker diarization
│   ├── asr.py          # Speech recognition
│   ├── align.py        # Word-speaker alignment
│   ├── nlp.py          # Sentiment/emotion/sarcasm
│   ├── summarize.py    # LLM summarization
│   ├── orchestrator.py # Pipeline coordinator
│   ├── calibration.py  # Probability calibration
│   └── storage.py      # Artifact persistence
├── models/
│   ├── base.py         # ABC interfaces
│   ├── hf_asr.py       # Whisper implementation
│   ├── hf_diarize.py   # Pyannote/hybrid
│   ├── hf_emotion.py   # Audio emotion
│   ├── hf_sentiment.py # Text sentiment
│   └── hf_sarcasm.py   # Sarcasm detection
└── pipelines/
    └── analyze_meeting.py  # End-to-end pipeline
```

## License

MIT
