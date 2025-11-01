# Testing the /analyze Endpoint

This guide shows how to test the `/analyze` endpoint with the sample audio files.

## Prerequisites

1. **Start the server:**
```bash
make run
```

2. **Verify server is running:**
```bash
curl http://localhost:8000/healthz
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models_loaded": true
}
```

---

## Method 1: Simple File Upload

**Minimal command - auto-detect everything:**

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav"
```

---

## Method 2: File Upload with Parameters

**Specify speaker count and language:**

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  -F "expected_speakers=2" \
  -F "language_hint=en" \
  -F "enable_overlaps=true" \
  -F "return_word_confidence=true"
```

---

## Method 3: Using Different Sample Files

### Clean Audio (2 speakers, no background noise):
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  -F "expected_speakers=2"
```

### Noisy Audio (2 speakers with background noise):
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_noisy.wav" \
  -F "expected_speakers=2"
```

### Sarcastic Speech (2 speakers with sarcasm):
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_sarcasm.wav" \
  -F "expected_speakers=2"
```

### Overlapping Speech (3 speakers with overlaps):
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_3speakers_overlap.wav" \
  -F "expected_speakers=3" \
  -F "enable_overlaps=true"
```

---

## Method 4: Pretty Print Response with jq

**Install jq (if not already installed):**
```bash
brew install jq  # macOS
```

**Analyze and format response:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  -F "expected_speakers=2" \
  | jq '.'
```

**Extract specific fields:**
```bash
# Get just the job_id
curl -s -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  | jq -r '.job_id'

# Get summary info
curl -s -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  | jq '{job_id, duration: .transcript.duration_sec, summary: .summary.summary}'
```

---

## Method 5: Save Response to File

**Save full response:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  -F "expected_speakers=2" \
  > response.json

# View it
cat response.json | jq '.'
```

**Extract job_id for artifact retrieval:**
```bash
JOB_ID=$(curl -s -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  | jq -r '.job_id')

echo "Job ID: $JOB_ID"
```

---

## Method 6: Retrieve Artifacts (After Analysis)

**Once the pipeline is implemented, you can retrieve artifacts:**

```bash
# Save job_id from analysis
JOB_ID="your-job-id-here"

# Get transcript
curl http://localhost:8000/artifacts/${JOB_ID}/transcript.json

# Get summary
curl http://localhost:8000/artifacts/${JOB_ID}/summary.json

# Get mood analysis
curl http://localhost:8000/artifacts/${JOB_ID}/mood.json

# Get markdown report
curl http://localhost:8000/artifacts/${JOB_ID}/report.md
```

---

## Method 7: Using HTTPie (Alternative to curl)

**Install HTTPie:**
```bash
brew install httpie  # macOS
```

**Simple upload:**
```bash
http --form POST http://localhost:8000/analyze \
  file@tests/data/sample_2speakers_clean.wav
```

**With parameters:**
```bash
http --form POST http://localhost:8000/analyze \
  file@tests/data/sample_2speakers_clean.wav \
  expected_speakers=2 \
  language_hint=en \
  enable_overlaps=true
```

---

## Method 8: Using Python requests

**Create a test script:**

```python
# test_analyze.py
import requests
from pathlib import Path

def analyze_audio(file_path: str, expected_speakers: int = None):
    """Analyze audio file using the API."""
    url = "http://localhost:8000/analyze"
    
    # Open file
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {}
        
        if expected_speakers:
            data['expected_speakers'] = expected_speakers
        
        # Make request
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        
        return response.json()

# Test it
if __name__ == '__main__':
    result = analyze_audio(
        'tests/data/sample_2speakers_clean.wav',
        expected_speakers=2
    )
    
    print(f"Job ID: {result['job_id']}")
    print(f"Duration: {result['transcript']['duration_sec']}s")
    print(f"Summary: {result['summary']['summary']}")
```

**Run it:**
```bash
python test_analyze.py
```

---

## Method 9: Interactive Testing with Swagger UI

1. **Open Swagger UI in browser:**
   ```
   http://localhost:8000/docs
   ```

2. **Find the `/analyze` endpoint** and click on it

3. **Click "Try it out"**

4. **Fill in the form:**
   - Click "Choose File" and select `tests/data/sample_2speakers_clean.wav`
   - Set `expected_speakers` to `2`
   - Set `language_hint` to `en`

5. **Click "Execute"**

6. **See the response** below with formatted JSON

---

## Expected Response Structure

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "models": {
    "asr": {
      "name": "openai/whisper-large-v3",
      "version": null,
      "config": {}
    },
    "diarization": {
      "name": "pyannote/speaker-diarization-3.1",
      "version": null,
      "config": {}
    }
  },
  "transcript": {
    "words": [],
    "utterances": [],
    "speakers": [],
    "duration_sec": 0.0,
    "detected_language": "en"
  },
  "summary": {
    "summary": "TODO: Implement pipeline",
    "decisions": [],
    "action_items": [],
    "disagreements": [],
    "risks": [],
    "open_questions": []
  },
  "mood": {
    "per_speaker": []
  },
  "artifacts": {
    "transcript_json": "",
    "summary_json": "",
    "mood_json": "",
    "report_md": ""
  },
  "metrics": {
    "wer": null,
    "der": null,
    "processing_time_sec": 0.045,
    "rtf": 0.0
  },
  "created_at": "2025-11-01T14:15:00.000000"
}
```

**Note:** Currently returns placeholder data as the ML pipeline is not yet implemented.

---

## Quick Test Commands

**Test all sample files:**
```bash
for sample in tests/data/sample_*.wav; do
  echo "Testing: $sample"
  curl -s -X POST http://localhost:8000/analyze \
    -F "file=@$sample" \
    | jq '{job_id, processing_time: .metrics.processing_time_sec}'
  echo ""
done
```

**Performance test:**
```bash
time curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  -o /dev/null -s -w "HTTP %{http_code} - %{time_total}s\n"
```

---

## Troubleshooting

**Server not responding:**
```bash
# Check if server is running
lsof -i :8000

# Start server if not running
make run
```

**Port 8000 in use:**
```bash
# Use a different port
poetry run uvicorn insightsvc.api.app:create_app \
  --factory --reload --host 0.0.0.0 --port 8001
```

**File not found:**
```bash
# Make sure you're in the project root
pwd  # Should show: /Users/anushkaparthiban/convobrief

# List available samples
ls -lh tests/data/*.wav
```
