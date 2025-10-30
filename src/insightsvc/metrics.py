"""Prometheus metrics for observability.

Tracks processing times, model performance, error rates, and resource usage.
Metrics are exposed via /metrics endpoint in Prometheus format.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary

# Request metrics
requests_total = Counter(
    "insightsvc_requests_total",
    "Total number of analysis requests",
    ["endpoint", "status"],
)

requests_in_progress = Gauge(
    "insightsvc_requests_in_progress",
    "Number of requests currently being processed",
)

# Processing time metrics
processing_time = Histogram(
    "insightsvc_processing_seconds",
    "Time spent processing requests",
    ["stage"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

audio_duration = Histogram(
    "insightsvc_audio_duration_seconds",
    "Duration of audio files processed",
    buckets=(10, 30, 60, 300, 600, 1800, 3600, 7200, 10800),
)

rtf_metric = Histogram(
    "insightsvc_rtf",
    "Real-time factor (processing_time / audio_duration)",
    buckets=(0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)

# Model metrics
model_inference_time = Summary(
    "insightsvc_model_inference_seconds",
    "Model inference time",
    ["model_type"],
)

# Diarization metrics
speakers_detected = Histogram(
    "insightsvc_speakers_detected",
    "Number of speakers detected",
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
)

der_metric = Histogram(
    "insightsvc_diarization_error_rate",
    "Diarization Error Rate",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
)

# ASR metrics
wer_metric = Histogram(
    "insightsvc_word_error_rate",
    "Word Error Rate",
    buckets=(0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
)

words_transcribed = Counter(
    "insightsvc_words_transcribed_total",
    "Total number of words transcribed",
)

# NLP metrics
sarcasm_detected = Counter(
    "insightsvc_sarcasm_detected_total",
    "Number of sarcastic utterances detected",
)

mood_valence = Histogram(
    "insightsvc_mood_valence",
    "Mood valence distribution",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Error metrics
errors_total = Counter(
    "insightsvc_errors_total",
    "Total number of errors",
    ["error_type", "recoverable"],
)

# Resource metrics
gpu_memory_used = Gauge(
    "insightsvc_gpu_memory_bytes",
    "GPU memory used",
    ["device"],
)

cpu_percent = Gauge(
    "insightsvc_cpu_percent",
    "CPU usage percentage",
)

# Model loading
models_loaded = Gauge(
    "insightsvc_models_loaded",
    "Number of models loaded",
    ["model_type"],
)
