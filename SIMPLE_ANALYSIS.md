# Simplified Analysis Endpoint

## Overview

The `/analyze/simple` endpoint provides a streamlined interface for quick conversation insights. Unlike the complex `/analyze` endpoint, this simplified version requires only a file upload and returns easy-to-read results.

## Endpoint Details

**URL**: `POST /analyze/simple`

**Content-Type**: `multipart/form-data`

**Authentication**: None required

## Request

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Audio file (.wav, .mp3, .m4a, .flac) |

### Example Request

#### Using curl

```bash
curl -X POST http://localhost:8000/analyze/simple \
  -F "file=@meeting.wav"
```

#### Using Python requests

```python
import requests

with open("meeting.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/simple",
        files={"file": f}
    )

result = response.json()
print(f"Detected {result['speakers_detected']} speakers")
```

#### Using Swagger UI

1. Navigate to `http://localhost:8000/docs`
2. Find the `/analyze/simple` endpoint under "Analysis"
3. Click "Try it out"
4. Click "Choose File" and select your audio file
5. Click "Execute"
6. View the response below

## Response

### Success Response (200 OK)

```json
{
  "speakers_detected": 3,
  "participants": [
    {
      "label": "Person 1",
      "talk_time_sec": 120.5,
      "tone": "calm",
      "mood": "positive",
      "sarcasm_detected": false,
      "context_summary": "Provided clear updates and maintained a professional, reassuring tone throughout the conversation."
    },
    {
      "label": "Person 2",
      "talk_time_sec": 85.3,
      "tone": "assertive",
      "mood": "neutral",
      "sarcasm_detected": true,
      "context_summary": "Offered direct feedback with occasional sarcasm. Focused on practical concerns and action items."
    },
    {
      "label": "Person 3",
      "talk_time_sec": 45.2,
      "tone": "enthusiastic",
      "mood": "positive",
      "sarcasm_detected": false,
      "context_summary": "Contributed ideas actively with positive energy. Engaged constructively with others' suggestions."
    }
  ],
  "overall_conversation_mood": "positive and collaborative",
  "feedback_summary": "Participants maintained professional engagement with constructive tone variations. The conversation was productive with clear communication and mutual respect.",
  "processing_time_sec": 12.45
}
```

### Response Fields

#### Top Level

| Field | Type | Description |
|-------|------|-------------|
| `speakers_detected` | integer | Number of unique speakers identified in the audio |
| `participants` | array | Detailed analysis for each participant |
| `overall_conversation_mood` | string | Summary of the overall conversational atmosphere |
| `feedback_summary` | string | High-level assessment of conversation dynamics |
| `processing_time_sec` | float | Time taken to process the audio (in seconds) |

#### Participant Object

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Speaker identifier (e.g., "Person 1", "Speaker A") |
| `talk_time_sec` | float | Total speaking time in seconds |
| `tone` | string | Overall speaking tone (e.g., "calm", "assertive", "anxious", "enthusiastic") |
| `mood` | string | Overall emotional state (e.g., "positive", "neutral", "negative", "mixed") |
| `sarcasm_detected` | boolean | Whether sarcasm was detected in this speaker's contributions |
| `context_summary` | string | Brief summary of the speaker's contributions and communication style |

### Error Responses

#### 400 Bad Request

Returned when the file is invalid or cannot be processed.

```json
{
  "detail": "Unsupported file format. Allowed: .wav, .mp3, .m4a, .flac"
}
```

#### 500 Internal Server Error

Returned when analysis fails unexpectedly.

```json
{
  "detail": "Analysis failed: [error details]"
}
```

## Use Cases

### 1. Quick Meeting Analysis

Upload a recorded meeting to quickly understand:
- How many people participated
- Who spoke most/least
- Overall meeting mood
- Presence of sarcasm or tension

### 2. Interview Assessment

Analyze job interviews to assess:
- Candidate's tone and mood
- Interviewer's communication style
- Overall interaction quality

### 3. Customer Service Quality

Evaluate customer service calls:
- Agent's tone and professionalism
- Customer's mood throughout the call
- Detection of frustration or satisfaction

### 4. Team Health Check

Monitor team meeting dynamics:
- Participation balance
- Overall team mood
- Signs of conflict or collaboration

## Comparison with Full `/analyze` Endpoint

| Feature | `/analyze` | `/analyze/simple` |
|---------|-----------|-------------------|
| Input complexity | Complex (multiple parameters) | Simple (file only) |
| Response detail | Very detailed (transcript, artifacts) | Concise (key insights) |
| Processing time | Longer (full pipeline) | Faster (essential analysis) |
| Artifacts generated | Yes (JSON, MD files) | No |
| Word-level timing | Yes | No |
| Job ID tracking | Yes | No |
| Use case | Detailed analysis, archival | Quick insights, real-time |

## Implementation Notes

### Current Status

⚠️ **Placeholder Implementation**: The endpoint currently returns simulated data to demonstrate the API structure. Once the ML pipeline is implemented, it will provide real analysis based on:

1. **Voice Activity Detection (VAD)**: Identify speech segments
2. **Speaker Diarization**: Separate speakers using pyannote.audio
3. **Speech Recognition**: Transcribe audio using Whisper
4. **Emotion Analysis**: Analyze valence/arousal using wav2vec2
5. **Sentiment Analysis**: Classify text sentiment using RoBERTa
6. **Sarcasm Detection**: Identify sarcastic speech patterns
7. **Context Summarization**: Generate summaries using Mistral LLM

### Backend Processing

When fully implemented, the endpoint will:

```python
# Pseudo-code of the actual pipeline
audio_data = load_audio(file_content)
speech_segments = detect_voice_activity(audio_data)
speakers = diarize_speakers(speech_segments)
transcripts = transcribe_with_whisper(audio_data, speakers)
emotions = analyze_emotions(audio_data, speakers)
sentiments = analyze_sentiment(transcripts)
sarcasm = detect_sarcasm(transcripts, audio_features)
summaries = generate_context_summaries(transcripts, emotions, sentiments)
```

### Performance

Expected performance (with actual ML pipeline):
- Short audio (< 1 min): ~5-10 seconds
- Medium audio (1-5 min): ~15-30 seconds
- Long audio (5-30 min): ~1-3 minutes

Factors affecting speed:
- Audio length
- Number of speakers
- Audio quality
- GPU availability (CUDA vs CPU)

## Testing

### Unit Test Example

```python
import pytest
from fastapi.testclient import TestClient
from insightsvc.api.app import create_app

client = TestClient(create_app())

def test_analyze_simple_success():
    with open("tests/data/sample_2speakers_clean.wav", "rb") as f:
        response = client.post(
            "/analyze/simple",
            files={"file": ("test.wav", f, "audio/wav")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "speakers_detected" in data
    assert data["speakers_detected"] >= 1
    assert len(data["participants"]) == data["speakers_detected"]
    assert "overall_conversation_mood" in data
    assert "processing_time_sec" in data

def test_analyze_simple_invalid_format():
    response = client.post(
        "/analyze/simple",
        files={"file": ("test.txt", b"not an audio file", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
```

### Integration Test

```bash
# Start server
make run

# Test with sample file
curl -X POST http://localhost:8000/analyze/simple \
  -F "file=@tests/data/sample_2speakers_clean.wav" \
  | jq '.speakers_detected'

# Expected output: 2
```

## Security Considerations

- **File Size Limit**: Configured in settings (default: 500MB)
- **File Type Validation**: Only audio formats accepted
- **Timeout Protection**: Long-running analysis should have timeout
- **Rate Limiting**: Consider implementing rate limits in production
- **Input Sanitization**: Filename validation to prevent path traversal

## Future Enhancements

### Planned Features

1. **Language Detection**: Auto-detect spoken language
2. **Emotion Timeline**: Show how mood changes over time
3. **Key Moments**: Highlight important discussion points
4. **Action Items**: Extract actionable tasks from conversation
5. **Confidence Scores**: Add confidence metrics for each analysis
6. **Real-time Streaming**: Support live audio analysis

### Configuration Options (Future)

```python
# Potential future parameters
@router.post("/analyze/simple")
async def analyze_audio_simple(
    file: UploadFile = File(...),
    language: str | None = None,  # Force specific language
    min_speakers: int = 2,        # Minimum expected speakers
    max_speakers: int = 5,        # Maximum expected speakers
    detect_sarcasm: bool = True,  # Enable/disable sarcasm detection
):
    ...
```

## Best Practices

### 1. File Preparation

- **Format**: Use WAV or FLAC for best quality
- **Sample Rate**: 16kHz recommended (will be resampled if different)
- **Channels**: Mono or stereo (will be converted to mono)
- **Quality**: Higher quality audio = better analysis results

### 2. API Usage

```python
# Good: Handle errors gracefully
try:
    response = requests.post(url, files={"file": audio_file})
    response.raise_for_status()
    result = response.json()
except requests.HTTPError as e:
    print(f"API error: {e.response.json()}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 3. Response Handling

```python
# Good: Validate response structure
result = response.json()
if result.get("speakers_detected", 0) > 0:
    for participant in result.get("participants", []):
        print(f"{participant['label']}: {participant['mood']}")
```

## Troubleshooting

### Issue: "File must have a valid filename"

**Cause**: Uploaded file has no filename
**Solution**: Ensure file has a proper name with extension

```python
# Wrong
files = {"file": file_content}

# Correct
files = {"file": ("meeting.wav", file_content, "audio/wav")}
```

### Issue: "Unsupported file format"

**Cause**: File extension not recognized
**Solution**: Use supported formats: .wav, .mp3, .m4a, .flac

### Issue: "Analysis failed"

**Cause**: Internal processing error
**Solution**: Check logs for details, ensure audio is valid

## Support

For issues or questions:
- Check logs: `tail -f logs/insightsvc.log`
- Review API docs: `http://localhost:8000/docs`
- See main documentation: `README.md`
- Report issues: GitHub Issues

## Related Documentation

- [Full Analysis Endpoint](./TESTING.md) - Detailed `/analyze` endpoint
- [Dashboard Guide](./DASHBOARD.md) - Streamlit visualization
- [API Reference](http://localhost:8000/docs) - Interactive OpenAPI docs
- [Implementation Guide](./IMPLEMENTATION_GUIDE.md) - ML pipeline details
