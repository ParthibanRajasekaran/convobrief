"""Generate synthetic audio samples for testing.

Creates test audio files with multiple speakers, overlaps, and various scenarios
to test the pipeline without requiring real meeting recordings.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def generate_silence(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence.

    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Silent audio array.
    """
    num_samples = int(duration_sec * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def generate_tone(
    frequency: float,
    duration_sec: float,
    amplitude: float = 0.3,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate a simple sine wave tone.

    Args:
        frequency: Frequency in Hz.
        duration_sec: Duration in seconds.
        amplitude: Amplitude (0-1).
        sample_rate: Sample rate in Hz.

    Returns:
        Tone audio array.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    return tone.astype(np.float32)


def apply_envelope(
    audio: np.ndarray,
    attack_sec: float = 0.01,
    release_sec: float = 0.01,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Apply attack/release envelope to prevent clicks.

    Args:
        audio: Input audio.
        attack_sec: Attack time in seconds.
        release_sec: Release time in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Audio with envelope applied.
    """
    attack_samples = int(attack_sec * sample_rate)
    release_samples = int(release_sec * sample_rate)

    envelope = np.ones_like(audio)

    # Attack
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples, dtype=np.float32)

    # Release
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples, dtype=np.float32)

    return audio * envelope


def generate_speech_like_signal(
    duration_sec: float,
    base_freq: float = 150.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate speech-like signal with formants.

    This creates a more realistic speech-like sound by combining multiple
    harmonics at typical formant frequencies.

    Args:
        duration_sec: Duration in seconds.
        base_freq: Base frequency (fundamental) in Hz.
        sample_rate: Sample rate in Hz.

    Returns:
        Speech-like audio array.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)

    # Generate fundamental
    signal = 0.4 * np.sin(2 * np.pi * base_freq * t)

    # Add harmonics (formants)
    formants = [
        (base_freq * 2, 0.3),
        (base_freq * 3, 0.2),
        (800, 0.15),  # Typical first formant
        (1200, 0.1),  # Typical second formant
    ]

    for freq, amp in formants:
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Add slight frequency modulation for naturalness
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
    signal = signal * vibrato

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.5

    # Apply envelope
    signal = apply_envelope(signal, attack_sec=0.02, release_sec=0.05, sample_rate=sample_rate)

    return signal.astype(np.float32)


def add_background_noise(
    audio: np.ndarray,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Add background noise to audio.

    Args:
        audio: Input audio.
        noise_level: Noise amplitude (0-1).

    Returns:
        Audio with noise added.
    """
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_level
    return audio + noise


def mix_audio_segments(
    segments: list[tuple[np.ndarray, float, float]],
    total_duration: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Mix multiple audio segments at specified times.

    Args:
        segments: List of (audio, start_time, end_time) tuples.
        total_duration: Total duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Mixed audio array.
    """
    total_samples = int(total_duration * sample_rate)
    mixed = np.zeros(total_samples, dtype=np.float32)

    for audio, start_time, _ in segments:
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(audio)

        # Clip if necessary
        if end_sample > total_samples:
            audio = audio[: total_samples - start_sample]
            end_sample = total_samples

        # Mix
        mixed[start_sample:end_sample] += audio

    return mixed


def save_audio(
    audio: np.ndarray,
    path: Path,
    sample_rate: int = 16000,
) -> None:
    """Save audio to WAV file.

    Args:
        audio: Audio array.
        path: Output path.
        sample_rate: Sample rate in Hz.
    """
    try:
        import soundfile as sf

        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sample_rate)
        print(f"✓ Saved: {path}")
    except ImportError:
        print("Warning: soundfile not installed. Install with: pip install soundfile")
        # Fallback to basic WAV writing
        import wave

        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        print(f"✓ Saved: {path}")


def generate_2speaker_clean(output_dir: Path) -> dict[str, Any]:
    """Generate clean 2-speaker conversation (no overlap).

    Returns:
        Metadata dictionary with ground truth.
    """
    print("\n📝 Generating 2-speaker clean conversation...")

    sample_rate = 16000
    segments = []
    utterances = []

    # Speaker 1: Lower pitch (male-like)
    s1_freq = 120.0
    # Speaker 2: Higher pitch (female-like)
    s2_freq = 220.0

    # Conversation structure
    timeline = [
        ("S1", 0.0, 2.5, "Hello everyone, let's start the meeting."),
        ("S2", 3.0, 5.0, "Good morning, I agree."),
        ("S1", 5.5, 8.5, "Today we need to discuss the project timeline."),
        ("S2", 9.0, 12.0, "Yes, and we should finalize the budget."),
        ("S1", 12.5, 15.0, "I'll take the action item for the timeline."),
        ("S2", 15.5, 18.0, "Great, I'll handle the budget analysis."),
        ("S1", 18.5, 20.0, "Any questions?"),
        ("S2", 20.5, 22.0, "No, that sounds good."),
    ]

    for speaker, start, end, text in timeline:
        duration = end - start
        freq = s1_freq if speaker == "S1" else s2_freq
        audio = generate_speech_like_signal(duration, freq, sample_rate)

        segments.append((audio, start, end))
        utterances.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    # Mix and add slight background noise
    total_duration = 23.0
    mixed = mix_audio_segments(segments, total_duration, sample_rate)
    mixed = add_background_noise(mixed, noise_level=0.005)

    # Normalize
    mixed = mixed / np.max(np.abs(mixed)) * 0.8

    # Save
    audio_path = output_dir / "sample_2speakers_clean.wav"
    save_audio(mixed, audio_path, sample_rate)

    # Save ground truth
    metadata = {
        "filename": audio_path.name,
        "duration_sec": total_duration,
        "sample_rate": sample_rate,
        "num_speakers": 2,
        "speakers": [
            {"id": "S1", "gender": "male", "base_freq": s1_freq},
            {"id": "S2", "gender": "female", "base_freq": s2_freq},
        ],
        "utterances": utterances,
        "has_overlap": False,
        "has_sarcasm": False,
    }

    metadata_path = output_dir / "sample_2speakers_clean.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    return metadata


def generate_3speaker_overlap(output_dir: Path) -> dict[str, Any]:
    """Generate 3-speaker conversation with overlaps (crosstalk).

    Returns:
        Metadata dictionary with ground truth.
    """
    print("\n📝 Generating 3-speaker conversation with overlaps...")

    sample_rate = 16000
    segments = []
    utterances = []

    # Three speakers with different pitches
    s1_freq = 110.0  # Low male
    s2_freq = 150.0  # Mid male
    s3_freq = 230.0  # Female

    # Conversation with some overlaps
    timeline = [
        ("S1", 0.0, 3.0, "Let's review the quarterly results."),
        ("S2", 3.5, 6.0, "The numbers look promising."),
        ("S3", 6.5, 9.0, "I agree, revenue is up fifteen percent."),
        ("S1", 9.5, 12.0, "But we need to address the cost increase."),
        # Overlap: S2 and S3 speaking simultaneously
        ("S2", 12.5, 15.0, "The marketing spend was necessary."),
        ("S3", 13.0, 15.5, "I think we should increase it even more."),  # Overlaps with S2
        ("S1", 16.0, 18.5, "Let's schedule a follow-up meeting."),
        # Another overlap
        ("S2", 19.0, 21.0, "I can prepare a detailed report."),
        ("S3", 20.0, 22.5, "And I'll analyze the market trends."),  # Overlaps with S2
        ("S1", 23.0, 25.0, "Perfect, thank you both."),
    ]

    for speaker, start, end, text in timeline:
        duration = end - start
        freq = {"S1": s1_freq, "S2": s2_freq, "S3": s3_freq}[speaker]
        audio = generate_speech_like_signal(duration, freq, sample_rate)

        segments.append((audio, start, end))
        utterances.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    # Mix with overlaps
    total_duration = 26.0
    mixed = mix_audio_segments(segments, total_duration, sample_rate)
    mixed = add_background_noise(mixed, noise_level=0.008)

    # Normalize
    mixed = mixed / np.max(np.abs(mixed)) * 0.8

    # Save
    audio_path = output_dir / "sample_3speakers_overlap.wav"
    save_audio(mixed, audio_path, sample_rate)

    # Save ground truth
    metadata = {
        "filename": audio_path.name,
        "duration_sec": total_duration,
        "sample_rate": sample_rate,
        "num_speakers": 3,
        "speakers": [
            {"id": "S1", "gender": "male", "base_freq": s1_freq},
            {"id": "S2", "gender": "male", "base_freq": s2_freq},
            {"id": "S3", "gender": "female", "base_freq": s3_freq},
        ],
        "utterances": utterances,
        "has_overlap": True,
        "overlap_regions": [(12.5, 15.5), (19.0, 22.5)],
        "has_sarcasm": False,
    }

    metadata_path = output_dir / "sample_3speakers_overlap.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    return metadata


def generate_sarcasm_scenario(output_dir: Path) -> dict[str, Any]:
    """Generate conversation with sarcastic utterances.

    Returns:
        Metadata dictionary with ground truth.
    """
    print("\n📝 Generating conversation with sarcasm...")

    sample_rate = 16000
    segments = []
    utterances = []

    s1_freq = 130.0
    s2_freq = 210.0

    # Conversation with sarcastic remarks
    timeline = [
        ("S1", 0.0, 2.5, "The project deadline is tomorrow.", False),
        ("S2", 3.0, 5.5, "Oh wonderful, plenty of time.", True),  # Sarcastic
        ("S1", 6.0, 8.5, "We're way behind schedule.", False),
        ("S2", 9.0, 11.5, "Yeah, this is going great.", True),  # Sarcastic
        ("S1", 12.0, 14.5, "Can you finish your part today?", False),
        ("S2", 15.0, 17.5, "Sure, I'll just work all night.", True),  # Sarcastic
        ("S1", 18.0, 20.0, "I know it's difficult.", False),
        ("S2", 20.5, 23.0, "Okay, I'll do my best.", False),
    ]

    for speaker, start, end, text, is_sarcastic in timeline:
        duration = end - start
        freq = s1_freq if speaker == "S1" else s2_freq

        # Modulate tone for sarcasm (slightly different prosody)
        if is_sarcastic:
            # Sarcasm often has different pitch contour
            audio = generate_speech_like_signal(duration, freq * 1.1, sample_rate)
        else:
            audio = generate_speech_like_signal(duration, freq, sample_rate)

        segments.append((audio, start, end))
        utterances.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
                "is_sarcastic": is_sarcastic,
            }
        )

    # Mix
    total_duration = 24.0
    mixed = mix_audio_segments(segments, total_duration, sample_rate)
    mixed = add_background_noise(mixed, noise_level=0.006)

    # Normalize
    mixed = mixed / np.max(np.abs(mixed)) * 0.8

    # Save
    audio_path = output_dir / "sample_2speakers_sarcasm.wav"
    save_audio(mixed, audio_path, sample_rate)

    # Save ground truth
    sarcastic_utterances = [u for u in utterances if u.get("is_sarcastic")]

    metadata = {
        "filename": audio_path.name,
        "duration_sec": total_duration,
        "sample_rate": sample_rate,
        "num_speakers": 2,
        "speakers": [
            {"id": "S1", "gender": "male", "base_freq": s1_freq},
            {"id": "S2", "gender": "female", "base_freq": s2_freq},
        ],
        "utterances": utterances,
        "has_overlap": False,
        "has_sarcasm": True,
        "sarcastic_utterances": sarcastic_utterances,
    }

    metadata_path = output_dir / "sample_2speakers_sarcasm.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    return metadata


def generate_noisy_audio(output_dir: Path) -> dict[str, Any]:
    """Generate conversation with significant background noise.

    Returns:
        Metadata dictionary with ground truth.
    """
    print("\n📝 Generating noisy conversation...")

    sample_rate = 16000
    segments = []
    utterances = []

    s1_freq = 140.0
    s2_freq = 200.0

    timeline = [
        ("S1", 0.0, 2.0, "Can you hear me?"),
        ("S2", 2.5, 4.5, "Yes, barely with all this noise."),
        ("S1", 5.0, 7.5, "Let's continue anyway."),
        ("S2", 8.0, 10.0, "Okay, sounds good."),
    ]

    for speaker, start, end, text in timeline:
        duration = end - start
        freq = s1_freq if speaker == "S1" else s2_freq
        audio = generate_speech_like_signal(duration, freq, sample_rate)

        segments.append((audio, start, end))
        utterances.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    # Mix with HIGH background noise
    total_duration = 11.0
    mixed = mix_audio_segments(segments, total_duration, sample_rate)
    mixed = add_background_noise(mixed, noise_level=0.05)  # 5x normal noise

    # Normalize
    mixed = mixed / np.max(np.abs(mixed)) * 0.8

    # Save
    audio_path = output_dir / "sample_2speakers_noisy.wav"
    save_audio(mixed, audio_path, sample_rate)

    # Save ground truth
    metadata = {
        "filename": audio_path.name,
        "duration_sec": total_duration,
        "sample_rate": sample_rate,
        "num_speakers": 2,
        "speakers": [
            {"id": "S1", "gender": "male", "base_freq": s1_freq},
            {"id": "S2", "gender": "female", "base_freq": s2_freq},
        ],
        "utterances": utterances,
        "has_overlap": False,
        "has_sarcasm": False,
        "noise_level": "high",
    }

    metadata_path = output_dir / "sample_2speakers_noisy.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    return metadata


def generate_all_samples(output_dir: Path) -> None:
    """Generate all test samples.

    Args:
        output_dir: Output directory for samples.
    """
    print("=" * 60)
    print("Generating Test Audio Samples")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []

    # Generate each scenario
    metadata_list.append(generate_2speaker_clean(output_dir))
    metadata_list.append(generate_3speaker_overlap(output_dir))
    metadata_list.append(generate_sarcasm_scenario(output_dir))
    metadata_list.append(generate_noisy_audio(output_dir))

    # Save combined manifest
    manifest = {
        "generated_at": "2025-10-30",
        "total_samples": len(metadata_list),
        "samples": metadata_list,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Sample Generation Complete!")
    print("=" * 60)
    print(f"Generated {len(metadata_list)} audio samples")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nSample files:")
    for meta in metadata_list:
        print(
            f"  - {meta['filename']} ({meta['duration_sec']:.1f}s, {meta['num_speakers']} speakers)"
        )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic audio samples for testing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data"),
        help="Output directory for samples",
    )

    args = parser.parse_args()

    generate_all_samples(args.output_dir)


if __name__ == "__main__":
    main()
