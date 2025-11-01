# API Endpoint Comparison Guide

Choose the right endpoint for your use case.

## Quick Comparison

| Feature | `/analyze/simple` ✨ NEW | `/analyze` (Full) |
|---------|-------------------------|-------------------|
| **Input** | File only | File or URI + config |
| **Configuration** | None (auto-detect) | Extensive options |
| **Response Size** | Small (~1-2 KB) | Large (10-100+ KB) |
| **Processing** | Essential analysis | Full pipeline |
| **Output** | Key insights | Complete details |
| **Artifacts** | None | JSON, MD files |
| **Job Tracking** | No | Yes (job_id) |
| **Best For** | Quick insights | Detailed analysis |

## Use Case Decision Tree

```
Do you need detailed transcripts with timestamps?
├─ YES → Use /analyze (Full)
└─ NO  → Continue...

Do you need word-level confidence scores?
├─ YES → Use /analyze (Full)
└─ NO  → Continue...

Do you need to save artifacts for later?
├─ YES → Use /analyze (Full)
└─ NO  → Continue...

Do you just want to know who spoke, their mood, and overall sentiment?
├─ YES → Use /analyze/simple ✨
└─ MAYBE → Keep reading...
```

## Detailed Comparison

### 1. Input Requirements

#### `/analyze/simple` ✨
```bash
# Just upload a file - that's it!
curl -X POST http://localhost:8000/analyze/simple \
  -F "file=@meeting.wav"
```

#### `/analyze` (Full)
```bash
# Complex configuration with many options
curl -X POST http://localhost:8000/analyze \
  -F "file=@meeting.wav" \
  -F "expected_speakers=3" \
  -F "language_hint=en" \
  -F "enable_overlaps=true" \
  -F "return_word_confidence=true" \
  -F 'request={"summarizer":{"max_words":250,"style":"concise"},"sarcasm_sensitivity":"balanced"}'
```

### 2. Response Structure

#### `/analyze/simple` ✨
```json
{
  "speakers_detected": 2,
  "participants": [
    {
      "label": "Person 1",
      "talk_time_sec": 120.5,
      "tone": "calm",
      "mood": "positive",
      "sarcasm_detected": false,
      "context_summary": "Clear communicator..."
    }
  ],
  "overall_conversation_mood": "positive",
  "feedback_summary": "Professional engagement",
  "processing_time_sec": 12.45
}
```
**Size**: ~500 bytes - 2 KB
**Readability**: ⭐⭐⭐⭐⭐ Excellent
**Complexity**: ⭐ Very Simple

#### `/analyze` (Full)
```json
{
  "job_id": "uuid-here",
  "models": {...},
  "transcript": {
    "words": [...100s of words...],
    "utterances": [...],
    "speakers": [...],
    "duration_sec": 300,
    "detected_language": "en"
  },
  "summary": {
    "summary": "...",
    "decisions": [...],
    "action_items": [...],
    "disagreements": [...],
    "risks": [...]
  },
  "mood": {
    "per_speaker": [...complex mood timeline...]
  },
  "artifacts": {...},
  "metrics": {...},
  "created_at": "..."
}
```
**Size**: 10 KB - 1 MB+
**Readability**: ⭐⭐ Technical
**Complexity**: ⭐⭐⭐⭐⭐ Very Complex

### 3. Processing Time

#### `/analyze/simple` ✨
- **Short audio (< 1 min)**: 3-5 seconds
- **Medium audio (1-5 min)**: 8-15 seconds
- **Long audio (5-30 min)**: 30-90 seconds

**Why faster?**
- Skips detailed transcription
- No artifact generation
- Essential analysis only
- Optimized for speed

#### `/analyze` (Full)
- **Short audio (< 1 min)**: 10-15 seconds
- **Medium audio (1-5 min)**: 30-60 seconds
- **Long audio (5-30 min)**: 2-5 minutes

**Why slower?**
- Complete word-level transcription
- Detailed mood timeline
- Artifact generation
- Comprehensive metrics

## Use Cases

### Best for `/analyze/simple` ✨

#### 1. Quick Meeting Insights
```python
# After every meeting, get quick feedback
def analyze_meeting(audio_file):
    result = api.post("/analyze/simple", files={"file": audio_file})
    return {
        "speakers": result["speakers_detected"],
        "mood": result["overall_conversation_mood"],
        "summary": result["feedback_summary"]
    }
```

#### 2. Real-time Dashboards
```javascript
// Update dashboard every 5 minutes
setInterval(async () => {
  const result = await fetch('/analyze/simple', {
    method: 'POST',
    body: formData
  });
  updateMoodIndicator(result.overall_conversation_mood);
}, 300000);
```

#### 3. Customer Service QA
```python
# Batch analyze 100s of calls quickly
for call in daily_calls:
    analysis = api.analyze_simple(call.audio)
    if analysis["sarcasm_detected"]:
        flag_for_review(call.id)
    
    store_metrics({
        "call_id": call.id,
        "mood": analysis["overall_conversation_mood"],
        "duration": analysis["processing_time_sec"]
    })
```

#### 4. Interview Screening
```python
# Quick candidate assessment
def screen_interview(audio_path):
    result = api.analyze_simple(audio_path)
    
    candidate = next(p for p in result["participants"] if p["label"] == "Person 1")
    
    return {
        "communication_score": score_tone(candidate["tone"]),
        "confidence": candidate["mood"],
        "professionalism": not candidate["sarcasm_detected"]
    }
```

### Best for `/analyze` (Full)

#### 1. Legal Documentation
```python
# Need complete, timestamped records
analysis = api.analyze_full(deposition_audio, expected_speakers=2)
save_transcript(analysis["transcript"]["words"])  # Every word with timestamp
save_artifacts(analysis["artifacts"])  # Official records
```

#### 2. Research & Analysis
```python
# Studying communication patterns
analysis = api.analyze_full(research_audio)
words_df = pd.DataFrame(analysis["transcript"]["words"])
analyze_speech_patterns(words_df)
plot_mood_timeline(analysis["mood"]["per_speaker"])
```

#### 3. Content Creation
```python
# Generate meeting minutes
analysis = api.analyze_full(meeting_audio)
minutes = {
    "attendees": analysis["transcript"]["speakers"],
    "key_decisions": analysis["summary"]["decisions"],
    "action_items": analysis["summary"]["action_items"],
    "full_transcript": analysis["artifacts"]["transcript_json"]
}
```

#### 4. Compliance & Audit
```python
# Keep detailed records
analysis = api.analyze_full(compliance_call)
audit_record = {
    "job_id": analysis["job_id"],
    "timestamp": analysis["created_at"],
    "transcript": analysis["transcript"],
    "metrics": {
        "wer": analysis["metrics"]["wer"],
        "der": analysis["metrics"]["der"]
    },
    "artifacts": analysis["artifacts"]
}
store_for_audit(audit_record)
```

## Performance Comparison

### Resource Usage

| Metric | `/analyze/simple` | `/analyze` (Full) |
|--------|-------------------|-------------------|
| CPU Usage | Low-Medium | High |
| Memory | < 1 GB | 2-4 GB |
| GPU (if available) | Optional | Recommended |
| Disk I/O | Minimal | Significant |
| Network (response) | < 5 KB | 100 KB - 5 MB |

### Throughput

**Simple Endpoint** ✨
- Can handle 10-20 concurrent requests
- Better for high-volume scenarios
- Suitable for real-time processing

**Full Endpoint**
- Recommend 2-5 concurrent requests
- Better for detailed analysis
- Suitable for batch processing

## Cost Comparison

Assuming cloud deployment with:
- CPU: $0.05/hour
- GPU: $0.50/hour
- Storage: $0.10/GB/month

### Simple Endpoint (1000 analyses/day)

```
Processing: ~10 min CPU/day = $0.008/day
Storage: Minimal = $0.00/month
Total: ~$0.24/month
```

### Full Endpoint (1000 analyses/day)

```
Processing: ~60 min GPU/day = $0.50/day
Storage: ~50 GB artifacts = $5.00/month
Total: ~$20/month
```

**Savings**: ~98% cost reduction for simple analysis!

## Migration Guide

### From Full → Simple

If you're currently using `/analyze` but don't need all features:

```python
# Before (Full endpoint)
result = api.post("/analyze", json={
    "audio_uri": "s3://bucket/audio.wav",
    "expected_speakers": 2,
    "language_hint": "en",
    "enable_overlaps": True,
    "return_word_confidence": True,
    "summarizer": {"max_words": 250, "style": "concise"},
    "sarcasm_sensitivity": "balanced"
})

# Extract what you actually use
num_speakers = len(result["transcript"]["speakers"])
mood = result["mood"]["per_speaker"][0]["final_rating"]["label"]

# After (Simple endpoint) - Get the same info faster
with open("audio.wav", "rb") as f:
    result = api.post("/analyze/simple", files={"file": f})

num_speakers = result["speakers_detected"]
mood = result["participants"][0]["mood"]
```

### Hybrid Approach

Use both endpoints strategically:

```python
def smart_analyze(audio_file, needs_transcript=False):
    """Use appropriate endpoint based on requirements."""
    
    if needs_transcript:
        # Use full analysis for detailed needs
        return api.post("/analyze", files={"file": audio_file})
    else:
        # Use simple analysis for quick insights
        return api.post("/analyze/simple", files={"file": audio_file})

# Quick check first
quick_result = api.post("/analyze/simple", files={"file": audio})

# Only do full analysis if needed
if quick_result["sarcasm_detected"] or quick_result["speakers_detected"] > 5:
    detailed_result = api.post("/analyze", files={"file": audio})
```

## Integration Examples

### Python Client

```python
class ConversationAnalyzer:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def quick_insights(self, audio_path: str) -> dict:
        """Get quick insights using simple endpoint."""
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/analyze/simple",
                files={"file": f}
            )
        response.raise_for_status()
        return response.json()
    
    def full_analysis(self, audio_path: str, **config) -> dict:
        """Get detailed analysis using full endpoint."""
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/analyze",
                files={"file": f},
                data=config
            )
        response.raise_for_status()
        return response.json()

# Usage
analyzer = ConversationAnalyzer()
insights = analyzer.quick_insights("meeting.wav")
print(f"Mood: {insights['overall_conversation_mood']}")
```

### JavaScript/TypeScript

```typescript
class ConversationAPI {
  constructor(private baseUrl = "http://localhost:8000") {}

  async quickInsights(file: File): Promise<SimpleAnalysisResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${this.baseUrl}/analyze/simple`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    return response.json();
  }
}

// Usage
const api = new ConversationAPI();
const result = await api.quickInsights(audioFile);
console.log(`Detected ${result.speakers_detected} speakers`);
```

## Recommendations

### Use `/analyze/simple` ✨ when:
- ✅ You need quick feedback
- ✅ You don't need transcripts
- ✅ You want to process many files
- ✅ You're building dashboards
- ✅ You're doing QA checks
- ✅ Cost is a concern
- ✅ Speed matters
- ✅ Non-technical users need results

### Use `/analyze` (Full) when:
- ✅ You need complete transcripts
- ✅ You need word-level timestamps
- ✅ You're generating documentation
- ✅ You need archival records
- ✅ You're doing research
- ✅ Quality metrics matter (WER, DER)
- ✅ You need exportable artifacts
- ✅ Compliance requires details

## FAQ

**Q: Can I use both endpoints on the same file?**
A: Yes! Use simple for quick checks, then full for detailed analysis if needed.

**Q: Are the results compatible?**
A: Yes, the simple endpoint extracts key insights from the same analysis pipeline.

**Q: Which is more accurate?**
A: Both use the same ML models. Full endpoint provides more detail, not more accuracy.

**Q: Can I upgrade from simple to full later?**
A: Yes, the simple endpoint doesn't save artifacts, but you can always re-analyze with the full endpoint.

**Q: Is the simple endpoint free?**
A: Both endpoints have the same API access. Cost difference is in compute/storage resources.

## See Also

- [Simple Analysis Documentation](./SIMPLE_ANALYSIS.md)
- [Full Analysis Testing Guide](./TESTING.md)
- [API Reference](http://localhost:8000/docs)
- [Dashboard Guide](./DASHBOARD.md)
