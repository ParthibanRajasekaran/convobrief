"""Test script to validate pipeline with generated samples.

Run this after implementing the pipeline to verify it works correctly
with the synthetic test data.
"""

import asyncio
import json
from pathlib import Path

# TODO: Uncomment when pipeline is implemented
# from insightsvc.pipelines.analyze_meeting import MeetingAnalysisPipeline
# from insightsvc.schemas import AnalyzeRequest


async def test_sample(audio_path: Path, pipeline=None) -> dict:
    """Test pipeline with a single sample.

    Args:
        audio_path: Path to audio file.
        pipeline: Pipeline instance (TODO: uncomment when ready).

    Returns:
        Test results dictionary.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {audio_path.name}")
    print(f"{'='*60}")

    # Load ground truth
    metadata_path = audio_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            ground_truth = json.load(f)
        print(
            f"Ground truth: {ground_truth['num_speakers']} speakers, "
            f"{ground_truth['duration_sec']:.1f}s"
        )
    else:
        ground_truth = None
        print("No ground truth metadata found")

    # TODO: Uncomment when pipeline is implemented
    # # Create request
    # request = AnalyzeRequest(
    #     audio_uri=str(audio_path),
    #     expected_speakers=None,  # Let it auto-detect
    #     language_hint="en",
    #     enable_overlaps=True,
    #     return_word_confidence=True,
    # )
    #
    # # Run pipeline
    # print("Running pipeline...")
    # result = await pipeline.analyze(request, uuid4())
    #
    # # Validate results
    # print("\nResults:")
    # print(f"  Detected speakers: {len(result.transcript.speakers)}")
    # print(f"  Total words: {len(result.transcript.words)}")
    # print(f"  Total utterances: {len(result.transcript.utterances)}")
    # print(f"  Processing time: {result.metrics.processing_time_sec:.2f}s")
    # print(f"  RTF: {result.metrics.rtf:.2f}x")
    #
    # # Compare with ground truth
    # test_results = {
    #     "audio_file": audio_path.name,
    #     "detected_speakers": len(result.transcript.speakers),
    #     "expected_speakers": ground_truth["num_speakers"] if ground_truth else None,
    #     "speaker_detection_correct": False,
    #     "total_words": len(result.transcript.words),
    #     "processing_time_sec": result.metrics.processing_time_sec,
    #     "rtf": result.metrics.rtf,
    # }
    #
    # if ground_truth:
    #     # Check speaker detection accuracy (within ±1)
    #     detected = len(result.transcript.speakers)
    #     expected = ground_truth["num_speakers"]
    #     test_results["speaker_detection_correct"] = abs(detected - expected) <= 1
    #
    #     if test_results["speaker_detection_correct"]:
    #         print(f"  ✅ Speaker detection correct: {detected} vs {expected}")
    #     else:
    #         print(f"  ❌ Speaker detection incorrect: {detected} vs {expected}")
    #
    #     # Check for overlaps if expected
    #     if ground_truth.get("has_overlap"):
    #         overlap_words = sum(1 for w in result.transcript.words if w.overlap)
    #         print(f"  Overlap words detected: {overlap_words}")
    #
    #     # Check for sarcasm if expected
    #     if ground_truth.get("has_sarcasm"):
    #         sarcasm_count = sum(
    #             1 for speaker_mood in result.mood.per_speaker
    #             for timepoint in speaker_mood.timeline
    #             if timepoint.sarcasm.is_sarcastic
    #         )
    #         print(f"  Sarcasm instances detected: {sarcasm_count}")
    #
    # return test_results

    # Placeholder until pipeline is implemented
    print("\n⚠️  Pipeline not yet implemented - skipping test")
    return {
        "audio_file": audio_path.name,
        "status": "skipped",
        "reason": "Pipeline not implemented",
    }


async def test_all_samples(samples_dir: Path) -> None:
    """Test pipeline with all generated samples.

    Args:
        samples_dir: Directory containing test samples.
    """
    print("=" * 60)
    print("Testing Pipeline with Generated Samples")
    print("=" * 60)

    # Find all WAV files
    audio_files = sorted(samples_dir.glob("*.wav"))

    if not audio_files:
        print(f"\n❌ No audio files found in {samples_dir}")
        print("Run: python tests/data/generate_samples.py")
        return

    print(f"\nFound {len(audio_files)} audio files")

    # TODO: Initialize pipeline when implemented
    # pipeline = MeetingAnalysisPipeline()

    # Test each sample
    results = []
    for audio_path in audio_files:
        try:
            result = await test_sample(audio_path, pipeline=None)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error testing {audio_path.name}: {e}")
            results.append(
                {
                    "audio_file": audio_path.name,
                    "status": "error",
                    "error": str(e),
                }
            )

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for result in results:
        status = result.get("status", "completed")
        if status == "completed":
            correct = "✅" if result.get("speaker_detection_correct") else "❌"
            print(
                f"{correct} {result['audio_file']}: "
                f"{result['detected_speakers']} speakers, "
                f"RTF={result['rtf']:.2f}x"
            )
        elif status == "skipped":
            print(f"⚠️  {result['audio_file']}: {result['reason']}")
        elif status == "error":
            print(f"❌ {result['audio_file']}: {result['error']}")

    # Save results
    results_path = samples_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": "2025-10-30",
                "total_samples": len(results),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_path}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test pipeline with generated audio samples")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("tests/data"),
        help="Directory containing test samples",
    )

    args = parser.parse_args()

    # Run async tests
    asyncio.run(test_all_samples(args.samples_dir))


if __name__ == "__main__":
    main()
