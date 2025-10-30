# Implementation Guide

This document provides a roadmap for completing the AI Conversation Insights Service implementation.

## Current Status

### ✅ Complete
- Project structure and configuration
- Pydantic schemas for all I/O
- Configuration management with environment variables
- Structured logging (structlog + JSON)
- Prometheus metrics definitions
- FastAPI application factory with error handling
- API routes scaffolding (`/analyze`, `/healthz`, `/metrics`, `/artifacts`)
- Alignment service (interval tree-based word-speaker assignment)
- Audio I/O service (loading, resampling, normalization)
- Timecode utilities
- Base model interfaces (ABCs for all model types)
- Whisper ASR implementation (partial)
- Unit tests for alignment
- Docker configuration
- Makefile for development workflow
- Comprehensive documentation

### 🚧 To Complete

The following files need implementation. Each TODO is marked with priority and estimated complexity.

## Priority 1: Core Pipeline (Critical Path)

### 1. Model Implementations

#### `src/insightsvc/models/hf_diarize.py` ⭐⭐⭐
**Priority: HIGH | Complexity: HIGH**

Implement two diarization backends:

```python
class PyannoteDiarizer(DiarizationModel):
    """Pyannote-based diarization (requires HF token)."""
    
    def __init__(self, model_name: str, hf_token: str, device: str):
        # Load pyannote.audio pipeline
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=hf_token
        )
        self.pipeline.to(torch.device(device))
    
    def diarize(self, audio, sample_rate, num_speakers, ...):
        # Convert numpy to torch
        # Run pipeline
        # Convert output to DiarizationSegment list
        pass

class HybridDiarizer(DiarizationModel):
    """Fallback: VAD + embeddings + clustering."""
    
    def __init__(self, device: str):
        # Load pyannote VAD or silero-vad
        # Load speechbrain speaker embeddings
        pass
    
    def diarize(self, audio, sample_rate, ...):
        # 1. Run VAD to get speech segments
        # 2. Extract speaker embeddings for each segment
        # 3. Cluster embeddings (agglomerative/spectral)
        # 4. Assign speaker labels
        pass
```

**Resources:**
- pyannote.audio docs: https://github.com/pyannote/pyannote-audio
- speechbrain embeddings: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

---

#### `src/insightsvc/models/hf_emotion.py` ⭐⭐
**Priority: HIGH | Complexity: MEDIUM**

```python
class Wav2Vec2Emotion(EmotionModel):
    """Audio emotion using wav2vec2."""
    
    def __init__(self, model_name: str, device: str):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
    
    def predict_emotion(self, audio, sample_rate):
        # Preprocess audio
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        # Run inference
        outputs = self.model(**inputs)
        # Extract valence/arousal from logits
        # Return EmotionResult
        pass
```

**Models to use:**
- `audeering/wav2vec2-large-robust-24-ft-emotion-msp-dim` (dimensional)
- Or `superb/hubert-large-superb-er` (categorical, map to dimensional)

---

#### `src/insightsvc/models/hf_sentiment.py` ⭐⭐
**Priority: HIGH | Complexity: LOW**

```python
class RoBERTaSentiment(SentimentModel):
    """Text sentiment using RoBERTa."""
    
    def __init__(self, model_name: str, device: str):
        from transformers import pipeline
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if device == "cuda" else -1
        )
    
    def analyze_sentiment(self, text):
        result = self.pipe(text)[0]
        # Return SentimentResult with label, scores, confidence
        pass
```

---

#### `src/insightsvc/models/hf_sarcasm.py` ⭐⭐
**Priority: MEDIUM | Complexity: LOW**

Similar to sentiment, use `cardiffnlp/twitter-roberta-base-irony`.

---

#### `src/insightsvc/models/hf_summarize.py` ⭐⭐⭐
**Priority: MEDIUM | Complexity: HIGH**

```python
class MistralSummarizer(SummarizerModel):
    """LLM-based summarization."""
    
    def __init__(self, model_name: str, device: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
    
    def generate(self, prompt, max_tokens, temperature, stop_sequences):
        # Format prompt with instruction template (Mistral/Llama format)
        # Run generation with stopping criteria
        # Parse JSON output
        pass
```

**Critical:** Implement the prompts from spec:
```python
SYSTEM_PROMPT = """You are a careful meeting analyst.
Tasks:
1) Produce a concise summary (<= {max_words} words) of the conversation.
2) List decisions, action items (owner, due date if stated), blockers, and open questions.
3) Capture disagreements and tone shifts. If sarcasm was detected, quote the line and note the real intent.
4) Do NOT invent facts. Only use the transcript. Prefer exact phrases with timestamps.
5) Keep speaker labels (S1..S5). Use UTC timestamps in mm:ss.
Output JSON with keys: summary, decisions[], action_items[], disagreements[], risks[].
"""
```

---

### 2. Service Layer

#### `src/insightsvc/services/vad.py` ⭐
**Priority: MEDIUM | Complexity: MEDIUM**

Voice activity detection (optional if using pyannote diarization which includes VAD).

```python
def run_vad(audio: np.ndarray, sr: int, threshold: float) -> list[tuple[float, float]]:
    """Detect speech segments.
    
    Use silero-vad or pyannote.audio VAD.
    Return list of (start, end) tuples.
    """
    pass
```

---

#### `src/insightsvc/services/nlp.py` ⭐⭐⭐
**Priority: HIGH | Complexity: HIGH**

**Most complex file** - implements mood fusion logic.

```python
class MoodAnalyzer:
    def __init__(self, emotion_model, sentiment_model, sarcasm_model, config):
        self.emotion_model = emotion_model
        self.sentiment_model = sentiment_model
        self.sarcasm_model = sarcasm_model
        self.fusion_weights = config.fusion_weights
        self.window_sec = config.emotion_window_sec
    
    def analyze_mood(
        self,
        utterances: list,
        audio: np.ndarray,
        sr: int
    ) -> Mood:
        """
        1. For each speaker:
           a. Extract utterance audio segments
           b. Run emotion model (valence/arousal)
           c. Run sentiment model on text
           d. Run sarcasm model on text with context
           e. Fuse scores using calibrated weights:
              valence_final = sigmoid(w1*valence_audio + w2*(sent_pos - sent_neg) + w3*(1 - sarcasm) + b)
           f. Build timeline of MoodTimepoint objects
           g. Compute final rating (mean valence, label, confidence)
        
        2. Return Mood object with per_speaker list
        """
        pass
    
    def _fuse_scores(self, valence_audio, sentiment_scores, sarcasm_prob):
        """Late fusion with calibrated weights."""
        pass
    
    def _detect_sarcasm_with_context(self, text, prev_utterances):
        """Run sarcasm detection with conversation context."""
        pass
```

**Key algorithm (from spec):**
```python
valence_final = sigmoid(
    w1 * valence_audio +
    w2 * (sent_pos - sent_neg) +
    w3 * (1 - sarcasm_prob) +
    bias
)

arousal_final = sigmoid(
    a1 * arousal_audio +
    a2 * prosody_energy +
    bias
)
```

---

#### `src/insightsvc/services/calibration.py` ⭐⭐
**Priority: LOW | Complexity: MEDIUM**

Temperature scaling for probability calibration.

```python
class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """Learn temperature on validation set."""
        # Optimize temperature to minimize NLL
        pass
    
    def calibrate(self, probs):
        """Apply temperature scaling."""
        return probs ** (1 / self.temperature)
```

For now, can use identity (temperature=1.0).

---

#### `src/insightsvc/services/storage.py` ⭐
**Priority: HIGH | Complexity: LOW**

```python
def write_artifacts(
    job_id: UUID,
    transcript: Transcript,
    summary: Summary,
    mood: Mood,
    models: dict,
    artifacts_dir: Path
) -> Artifacts:
    """Write JSON artifacts and markdown report.
    
    1. Create job directory
    2. Write transcript.json
    3. Write summary.json
    4. Write mood.json
    5. Generate report.md with:
       - Title block
       - Summary
       - Decisions & action items
       - Per-speaker mood with ASCII sparklines
       - Sarcasm snippets with timestamps
    6. Return Artifacts with paths
    """
    pass

def generate_report_md(...) -> str:
    """Generate markdown report."""
    pass
```

---

#### `src/insightsvc/services/orchestrator.py` ⭐⭐⭐
**Priority: HIGH | Complexity: LOW** (orchestration only)

Move the TODO pipeline logic from `pipelines/analyze_meeting.py` into a clean orchestrator:

```python
class PipelineOrchestrator:
    def __init__(self, models: dict, config: Settings):
        self.models = models
        self.config = config
    
    async def run_pipeline(self, request: AnalyzeRequest, job_id: UUID) -> AnalyzeResponse:
        """Execute full pipeline."""
        # 1. Load audio
        # 2. Run diarization
        # 3. Run ASR
        # 4. Align words
        # 5. Run NLP
        # 6. Summarize
        # 7. Write artifacts
        # 8. Return response
        pass
```

---

### 3. Complete Pipeline Integration

#### `src/insightsvc/pipelines/analyze_meeting.py`
**Priority: HIGH**

Replace TODOs with actual model calls:

```python
class MeetingAnalysisPipeline:
    def __init__(self):
        settings = get_settings()
        
        # Initialize all models
        self.asr_model = WhisperASR(settings.asr_model_name, settings.device)
        
        if settings.diarization_backend == "pyannote" and settings.hf_token:
            self.diarization_model = PyannoteDiarizer(...)
        else:
            self.diarization_model = HybridDiarizer(...)
        
        self.emotion_model = Wav2Vec2Emotion(...)
        self.sentiment_model = RoBERTaSentiment(...)
        self.sarcasm_model = RoBERTaSarcasm(...)
        self.summarizer = MistralSummarizer(...)
        
        self.mood_analyzer = MoodAnalyzer(...)
        self.orchestrator = PipelineOrchestrator(...)
    
    async def analyze(self, request, job_id):
        return await self.orchestrator.run_pipeline(request, job_id)
```

---

#### `src/insightsvc/api/routes.py`
**Priority: HIGH**

Replace TODO in `/analyze` endpoint:

```python
from insightsvc.pipelines.analyze_meeting import MeetingAnalysisPipeline

# Global pipeline instance (initialized on startup)
_pipeline: MeetingAnalysisPipeline | None = None

def get_pipeline() -> MeetingAnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MeetingAnalysisPipeline()
    return _pipeline

@router.post("/analyze", ...)
async def analyze_audio(...):
    # ... existing code ...
    
    pipeline = get_pipeline()
    result = await pipeline.analyze(req, job_id)
    
    return result
```

---

## Priority 2: Testing & Validation

### Unit Tests

Create tests for each component:

1. **`tests/unit/test_audio_io.py`** - Audio loading, resampling, normalization
2. **`tests/unit/test_timecode.py`** - Timestamp conversions, overlap calculations
3. **`tests/unit/test_nlp.py`** - Fusion logic, sarcasm detection
4. **`tests/unit/test_models.py`** - Mock model outputs

### Integration Tests

**`tests/integration/test_pipeline.py`**

```python
@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """Test full pipeline with sample audio."""
    # Create sample audio (2 speakers, 30 seconds)
    sample_audio = generate_test_audio()
    
    request = AnalyzeRequest(audio_uri=str(sample_audio))
    pipeline = MeetingAnalysisPipeline()
    
    result = await pipeline.analyze(request, uuid.uuid4())
    
    # Assertions
    assert result.transcript.detected_speakers >= 2
    assert len(result.transcript.words) > 0
    assert len(result.summary.summary) > 0
    assert result.metrics.der is not None
```

### Sample Data

**`tests/data/`**

Create or source:
- `sample_2speakers.wav` (clean, 10-30 seconds)
- `sample_3speakers_overlap.wav` (with crosstalk)
- `sample_sarcasm.wav` (with sarcastic utterances)
- Reference transcripts for WER calculation

**Generation script:**
```python
# tests/data/generate_samples.py
from TTS import TTS  # or use edge-tts, gTTS
def generate_test_audio():
    # Create synthetic conversations
    pass
```

---

## Priority 3: Benchmarking & Metrics

### `scripts/benchmark.py`

```python
"""Benchmark WER, DER, F1, and throughput."""

import time
from pathlib import Path
from jiwer import wer

def compute_wer(reference: str, hypothesis: str) -> float:
    return wer(reference, hypothesis)

def compute_der(ref_rttm: Path, hyp_rttm: Path) -> float:
    # Use pyannote.metrics
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate()
    # Load and evaluate
    pass

def benchmark_throughput(audio_dir: Path):
    """Measure RTF on sample folder."""
    for audio_file in audio_dir.glob("*.wav"):
        start = time.time()
        # Run pipeline
        duration = time.time() - start
        audio_duration = get_audio_duration(audio_file)
        rtf = duration / audio_duration
        print(f"{audio_file.name}: RTF={rtf:.2f}")

if __name__ == "__main__":
    benchmark_throughput(Path("tests/data"))
```

---

## Priority 4: Ops & Deployment

### Environment Setup

1. **Create `.env` from `.env.example`**
2. **Obtain HF token:** https://huggingface.co/settings/tokens
3. **Accept pyannote licenses**

### Docker Optimization

**Multi-stage build:**
```dockerfile
# Build stage
FROM python:3.11-slim as builder
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt > requirements.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
CMD ["uvicorn", "insightsvc.api.app:create_app", "--factory", "--host", "0.0.0.0"]
```

### Prometheus Configuration

**`prometheus.yml`:**
```yaml
scrape_configs:
  - job_name: 'insightsvc'
    static_configs:
      - targets: ['insightsvc:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

---

## Priority 5: Polish & Documentation

1. **Complete docstrings** - Ensure all public functions have Google-style docs
2. **Type hints** - Run `mypy` and fix all errors
3. **Linting** - Run `ruff` and `black`
4. **README examples** - Add curl examples, Jupyter notebook demo
5. **API documentation** - Verify OpenAPI docs at `/docs`
6. **Performance tuning:**
   - Batch processing for long audios
   - Model quantization (int8/int4)
   - ONNX export for faster inference
   - Multi-GPU support

---

## Quick Start for Implementation

### Week 1: Core Models
1. Day 1-2: Complete `hf_diarize.py` (Pyannote + Hybrid)
2. Day 3: Complete `hf_emotion.py`, `hf_sentiment.py`, `hf_sarcasm.py`
3. Day 4-5: Complete `hf_summarize.py` with prompt engineering

### Week 2: Integration
1. Day 1-2: Implement `nlp.py` (fusion logic)
2. Day 3: Implement `storage.py` and report generation
3. Day 4: Complete `orchestrator.py` and integrate into `analyze_meeting.py`
4. Day 5: Wire up API routes and test end-to-end

### Week 3: Testing & Validation
1. Day 1-2: Create test fixtures and sample data
2. Day 3: Write unit tests (target >80% coverage)
3. Day 4: Integration tests
4. Day 5: Benchmark and optimize

### Week 4: Deployment & Docs
1. Day 1: Docker optimization
2. Day 2: Prometheus dashboards
3. Day 3: Documentation polish
4. Day 4-5: Performance tuning and final validation

---

## Validation Checklist

Before considering complete, verify:

- [ ] Service starts without errors
- [ ] `/healthz` returns 200
- [ ] `/analyze` accepts audio file and returns valid JSON
- [ ] Detects 2-5 speakers within ±1 of ground truth
- [ ] Assigns ≥95% of words a speaker (clean audio)
- [ ] Summary contains only transcript facts (no hallucinations)
- [ ] Sarcasm examples include timestamped quotes
- [ ] Per-speaker mood rating present with confidence
- [ ] All outputs include model metadata
- [ ] `pytest` passes all tests
- [ ] `mypy src` has zero errors
- [ ] `ruff check src` passes
- [ ] Docker image builds successfully
- [ ] RTF < 1.5× on GPU for 60-min audio

---

## Resources

- **Pyannote:** https://github.com/pyannote/pyannote-audio
- **Whisper:** https://github.com/openai/whisper
- **HF Transformers:** https://huggingface.co/docs/transformers
- **Prompt Engineering:** https://www.promptingguide.ai/
- **WER/DER Metrics:** https://github.com/jitsi/jiwer, pyannote.metrics

---

## Notes

- All TODOs in code files are marked with clear context
- Use `logger.info()` at each pipeline stage for observability
- Wrap model errors in `AnalysisError` with helpful hints
- For S3 support, use `boto3` with presigned URLs
- Consider adding caching for repeat analyses

---

Good luck! 🚀
