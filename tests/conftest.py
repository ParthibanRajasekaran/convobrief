"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_audio_path():
    """Path to sample audio file for testing."""
    # TODO: Add sample audio files to tests/data/
    return "tests/data/sample_2speakers.wav"


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from insightsvc.config import Settings

    return Settings(
        device="cpu",
        log_level="DEBUG",
        artifacts_dir="/tmp/test_artifacts",
        hf_token="test_token",
    )
