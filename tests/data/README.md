# Test Audio Samples

This directory contains synthetic audio samples for testing the AI Conversation Insights Service.

## Quick Start

### Generate Samples

```bash
# From project root
python tests/data/generate_samples.py

# Or specify output directory
python tests/data/generate_samples.py --output-dir tests/data
```

This will create 4 test audio files with ground truth metadata.

## Generated Samples

### 1. `sample_2speakers_clean.wav` (23s)
- **Speakers**: 2 (1 male, 1 female)
- **Scenario**: Clean conversation, no overlaps
- **Use for**: Basic diarization and ASR testing

**Expected Output**:
- Speaker detection: 2 speakers
- Word assignment rate: >95%
- No overlap detection

### 2. `sample_3speakers_overlap.wav` (26s)
- **Speakers**: 3 (2 male, 1 female)
- **Scenario**: Conversation with crosstalk
- **Overlaps**: 2 regions (12.5-15.5s, 19.0-22.5s)
- **Use for**: Overlap detection and handling

**Expected Output**:
- Speaker detection: 3 speakers
- Overlap regions detected
- Words in overlap marked with `overlap=true`

### 3. `sample_2speakers_sarcasm.wav` (24s)
- **Speakers**: 2
- **Scenario**: Conversation with sarcastic remarks
- **Sarcastic utterances**: 3 marked in metadata
- **Use for**: Sarcasm detection and mood analysis

**Expected Output**:
- Sarcastic utterances identified
- Mood analysis shows mismatch between text sentiment and prosody
- Evidence quotes with timestamps

### 4. `sample_2speakers_noisy.wav` (11s)
- **Speakers**: 2
- **Scenario**: High background noise
- **Noise level**: 5x normal
- **Use for**: Robustness testing

**Expected Output**:
- Speaker detection: 2 speakers
- Word assignment rate: >85% (noisy conditions)
- Lower confidence scores

## Metadata Format

Each audio file has a corresponding `.json` file with ground truth:

```json
{
  "filename": "sample_2speakers_clean.wav",
  "duration_sec": 23.0,
  "sample_rate": 16000,
  "num_speakers": 2,
  "speakers": [
    {"id": "S1", "gender": "male", "base_freq": 120.0},
    {"id": "S2", "gender": "female", "base_freq": 220.0}
  ],
  "utterances": [
    {
      "speaker": "S1",
      "start": 0.0,
      "end": 2.5,
      "text": "Hello everyone, let's start the meeting."
    },
    ...
  ],
  "has_overlap": false,
  "has_sarcasm": false
}
```

## Using Samples in Tests

### Unit Tests

```python
import json
from pathlib import Path

def test_with_sample():
    # Load audio
    audio_path = Path("tests/data/sample_2speakers_clean.wav")
    
    # Load ground truth
    metadata_path = audio_path.with_suffix(".json")
    with open(metadata_path) as f:
        ground_truth = json.load(f)
    
    # Run pipeline
    result = pipeline.analyze(audio_path)
    
    # Verify against ground truth
    assert len(result.speakers) == ground_truth["num_speakers"]
```

### Integration Tests

```python
import pytest
from pathlib import Path

@pytest.mark.parametrize("sample", [
    "sample_2speakers_clean.wav",
    "sample_3speakers_overlap.wav",
    "sample_2speakers_sarcasm.wav",
])
def test_pipeline_with_samples(sample):
    audio_path = Path("tests/data") / sample
    result = pipeline.analyze(audio_path)
    
    # Load ground truth and validate
    ...
```

## Audio Characteristics

All samples:
- **Format**: WAV (PCM)
- **Sample Rate**: 16 kHz
- **Channels**: Mono
- **Bit Depth**: 16-bit (when saved via soundfile)
- **Duration**: 10-30 seconds

## Limitations

⚠️ **Important**: These are **synthetic** audio samples, not real speech!

The samples use:
- **Sine wave harmonics** to simulate speech formants
- **Simple pitch differences** to distinguish speakers
- **Basic amplitude modulation** for prosody

They are useful for:
- ✅ Testing pipeline infrastructure
- ✅ Validating speaker separation logic
- ✅ Testing timing and alignment
- ✅ Integration testing

They are NOT suitable for:
- ❌ Realistic ASR evaluation (no actual words)
- ❌ Emotion detection validation (no real prosody)
- ❌ Production benchmarking

## Real Audio Samples

For production validation, use real meeting recordings:

### Option 1: Public Datasets
- **AMI Corpus**: Meeting recordings with transcripts
  - URL: https://groups.inf.ed.ac.uk/ami/corpus/
  - License: CC BY 4.0

- **ICSI Meeting Corpus**: Meetings with diarization
  - URL: https://groups.inf.ed.ac.uk/ami/icsi/

### Option 2: Generate with TTS

```python
# Install TTS library
# pip install TTS

from TTS.api import TTS

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech
tts.tts_to_file(
    text="Hello, this is speaker one.",
    file_path="speaker1.wav"
)
```

### Option 3: Record Your Own

Use any audio recording tool to create test meetings with known:
- Speaker count
- Utterance timestamps
- Ground truth transcripts

## Manifest File

`manifest.json` contains metadata for all samples:

```json
{
  "generated_at": "2025-10-30",
  "total_samples": 4,
  "samples": [...]
}
```

Use this for batch testing:

```python
import json

with open("tests/data/manifest.json") as f:
    manifest = json.load(f)

for sample in manifest["samples"]:
    audio_path = f"tests/data/{sample['filename']}"
    # Test each sample
```

## Dependencies

Minimal (numpy only):
```bash
pip install numpy
```

Recommended (for better audio quality):
```bash
pip install numpy soundfile
```

## Troubleshooting

### "soundfile not installed"
The script will fall back to basic WAV writing. For better quality:
```bash
pip install soundfile
```

### "Import numpy could not be resolved"
Install numpy:
```bash
pip install numpy
```

### Audio files too quiet
The samples are normalized to 0.8 to prevent clipping. Adjust the `amplitude` parameter in `generate_speech_like_signal()` if needed.

## Contributing

To add new test scenarios:

1. Create a new `generate_*_scenario()` function
2. Add it to `generate_all_samples()`
3. Document the scenario in this README

Example scenarios to add:
- Long meetings (60+ minutes)
- More speakers (4-5)
- Different languages (when multilingual models added)
- Specific edge cases (very short utterances, long silences)
