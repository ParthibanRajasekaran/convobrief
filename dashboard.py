"""Interactive Dashboard for AI Conversation Insights Service.

A Streamlit-based frontend for visualizing meeting analysis results including
transcripts, speaker diarization, mood analysis, and quality metrics.

Usage:
    streamlit run dashboard.py

Requirements:
    - FastAPI backend running on http://localhost:8000
    - streamlit, requests, plotly, pandas
"""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Configuration
API_BASE_URL = "http://localhost:8000"
SUPPORTED_FORMATS = ["wav", "mp3", "m4a", "flac"]
MAX_FILE_SIZE_MB = 500

# Page configuration
st.set_page_config(
    page_title="AI Conversation Insights",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def check_backend_health() -> dict[str, Any] | None:
    """Check if the FastAPI backend is running and healthy.

    Returns:
        Health status dict or None if backend is unreachable.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def upload_and_analyze(
    file_bytes: bytes,
    filename: str,
    expected_speakers: int | None = None,
    language_hint: str | None = None,
    request_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload audio file and trigger analysis via FastAPI backend.

    Args:
        file_bytes: Audio file content as bytes.
        filename: Original filename.
        expected_speakers: Expected number of speakers.
        language_hint: ISO 639-1 language code.
        request_config: Additional request configuration (summarizer, sarcasm_sensitivity).

    Returns:
        Analysis result JSON from backend.

    Raises:
        requests.HTTPError: If API call fails.
    """
    files = {"file": (filename, file_bytes)}
    data = {}

    if expected_speakers is not None:
        data["expected_speakers"] = expected_speakers
    if language_hint:
        data["language_hint"] = language_hint

    # Add JSON request config if provided
    if request_config:
        data["request"] = json.dumps(request_config)

    response = requests.post(
        f"{API_BASE_URL}/analyze",
        files=files,
        data=data,
        timeout=300,  # 5 minutes timeout for large files
    )
    response.raise_for_status()
    return response.json()


def get_artifact(job_id: str, artifact_name: str) -> str:
    """Retrieve artifact file from backend.

    Args:
        job_id: Job ID from analysis response.
        artifact_name: Artifact filename (e.g., 'report.md').

    Returns:
        Artifact content as string.

    Raises:
        requests.HTTPError: If artifact not found.
    """
    response = requests.get(
        f"{API_BASE_URL}/artifacts/{job_id}/{artifact_name}",
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def render_summary_tab(result: dict[str, Any]) -> None:
    """Render the Summary tab with meeting overview and key points.

    Args:
        result: Analysis result from backend.
    """
    st.header("📋 Meeting Summary")

    summary_data = result.get("summary", {})

    # Main summary text
    summary_text = summary_data.get("summary", "No summary available")
    st.markdown("### Overview")
    st.info(summary_text)

    # Create columns for key information
    col1, col2 = st.columns(2)

    with col1:
        # Decisions
        decisions = summary_data.get("decisions", [])
        if decisions:
            st.markdown("### ✅ Decisions Made")
            for i, decision in enumerate(decisions, 1):
                with st.expander(f"Decision {i} (t={decision.get('timestamp', 0):.1f}s)"):
                    st.write(decision.get("text", ""))
                    st.caption(f"Speakers: {', '.join(decision.get('speakers', []))}")
                    st.caption(f"Confidence: {decision.get('confidence', 0):.2%}")
        else:
            st.markdown("### ✅ Decisions Made")
            st.caption("No decisions detected")

        # Disagreements
        disagreements = summary_data.get("disagreements", [])
        if disagreements:
            st.markdown("### ⚠️ Disagreements")
            for i, disagreement in enumerate(disagreements, 1):
                with st.expander(f"Disagreement {i} (t={disagreement.get('timestamp', 0):.1f}s)"):
                    st.write(disagreement.get("text", ""))
                    st.caption(f"Speakers: {', '.join(disagreement.get('speakers', []))}")
                    intensity = disagreement.get("intensity", 0)
                    st.progress(intensity, text=f"Intensity: {intensity:.2%}")
        else:
            st.markdown("### ⚠️ Disagreements")
            st.caption("No disagreements detected")

    with col2:
        # Action items
        action_items = summary_data.get("action_items", [])
        if action_items:
            st.markdown("### 📌 Action Items")
            for i, item in enumerate(action_items, 1):
                with st.expander(f"Action {i} (t={item.get('timestamp', 0):.1f}s)"):
                    st.write(item.get("text", ""))
                    if item.get("owner"):
                        st.caption(f"Owner: {item['owner']}")
                    if item.get("due_date"):
                        st.caption(f"Due: {item['due_date']}")
        else:
            st.markdown("### 📌 Action Items")
            st.caption("No action items detected")

        # Risks
        risks = summary_data.get("risks", [])
        if risks:
            st.markdown("### 🚨 Risks & Blockers")
            for i, risk in enumerate(risks, 1):
                with st.expander(f"Risk {i} (t={risk.get('timestamp', 0):.1f}s)"):
                    st.write(risk.get("text", ""))
                    st.caption(f"Category: {risk.get('category', 'Unknown')}")
        else:
            st.markdown("### 🚨 Risks & Blockers")
            st.caption("No risks identified")

    # Open questions
    open_questions = summary_data.get("open_questions", [])
    if open_questions:
        st.markdown("### ❓ Open Questions")
        for i, question in enumerate(open_questions, 1):
            st.write(f"{i}. {question}")
    else:
        st.markdown("### ❓ Open Questions")
        st.caption("No open questions")


def render_transcript_tab(result: dict[str, Any]) -> None:
    """Render the Transcript tab with speaker-attributed dialogue.

    Args:
        result: Analysis result from backend.
    """
    st.header("💬 Transcript")

    transcript_data = result.get("transcript", {})
    utterances = transcript_data.get("utterances", [])
    speakers_stats = transcript_data.get("speakers", [])
    duration = transcript_data.get("duration_sec", 0)
    detected_language = transcript_data.get("detected_language", "Unknown")

    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Duration", f"{duration:.1f}s")
    with col2:
        st.metric("Utterances", len(utterances))
    with col3:
        st.metric("Speakers", len(speakers_stats))
    with col4:
        st.metric("Language", detected_language or "Auto-detected")

    # Speaker statistics
    if speakers_stats:
        st.markdown("### 👥 Speaker Statistics")
        speaker_df = pd.DataFrame(speakers_stats)
        speaker_df = speaker_df.rename(
            columns={
                "id": "Speaker",
                "talk_time_sec": "Talk Time (s)",
                "utterance_count": "Utterances",
                "avg_confidence": "Avg Confidence",
            }
        )
        speaker_df["Talk Time (s)"] = speaker_df["Talk Time (s)"].round(1)
        speaker_df["Avg Confidence"] = speaker_df["Avg Confidence"].round(3)

        # Visualize speaker distribution
        fig = px.bar(
            speaker_df,
            x="Speaker",
            y="Talk Time (s)",
            color="Avg Confidence",
            title="Speaker Talk Time Distribution",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show table
        st.dataframe(speaker_df, hide_index=True, use_container_width=True)

    # Timeline visualization
    if utterances:
        st.markdown("### 📊 Timeline Visualization")

        # Create timeline data
        timeline_data = []
        for utt in utterances:
            timeline_data.append(
                {
                    "Speaker": utt.get("speaker", "Unknown"),
                    "Start": utt.get("start", 0),
                    "End": utt.get("end", 0),
                    "Duration": utt.get("end", 0) - utt.get("start", 0),
                    "Text": utt.get("text", "")[:50] + "..." if len(utt.get("text", "")) > 50 else utt.get("text", ""),
                    "Overlap": "Yes" if utt.get("overlap", False) else "No",
                }
            )

        timeline_df = pd.DataFrame(timeline_data)

        # Create Gantt-style chart
        fig = px.timeline(
            timeline_df,
            x_start="Start",
            x_end="End",
            y="Speaker",
            color="Speaker",
            hover_data=["Text", "Overlap"],
            title="Speaker Timeline",
        )
        fig.update_yaxes(categoryorder="category ascending")
        st.plotly_chart(fig, use_container_width=True)

    # Full transcript
    st.markdown("### 📝 Full Transcript")

    if not utterances:
        st.warning("No utterances available in transcript")
        return

    # Group by speaker filter
    speaker_filter = st.multiselect(
        "Filter by speaker",
        options=sorted(set(u.get("speaker", "Unknown") for u in utterances)),
        default=sorted(set(u.get("speaker", "Unknown") for u in utterances)),
    )

    # Display utterances
    for i, utterance in enumerate(utterances):
        speaker = utterance.get("speaker", "Unknown")

        if speaker not in speaker_filter:
            continue

        start = utterance.get("start", 0)
        end = utterance.get("end", 0)
        text = utterance.get("text", "")
        overlap = utterance.get("overlap", False)
        confidence = utterance.get("confidence", 0)

        # Color-code by speaker
        timestamp_str = f"[{start:.1f}s - {end:.1f}s]"
        overlap_badge = "🔀 " if overlap else ""

        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"**{speaker}**")
                st.caption(timestamp_str)
            with col2:
                st.markdown(f"{overlap_badge}{text}")
                st.caption(f"Confidence: {confidence:.2%}")
            st.divider()


def render_mood_tab(result: dict[str, Any]) -> None:
    """Render the Mood Analysis tab with emotion and sentiment visualization.

    Args:
        result: Analysis result from backend.
    """
    st.header("🎭 Mood Analysis")

    mood_data = result.get("mood", {})
    per_speaker = mood_data.get("per_speaker", [])

    if not per_speaker:
        st.warning("No mood analysis data available")
        return

    # Speaker selector
    speaker_names = [s.get("speaker", "Unknown") for s in per_speaker]
    selected_speaker = st.selectbox("Select Speaker", speaker_names)

    # Find selected speaker data
    speaker_mood = None
    for s in per_speaker:
        if s.get("speaker") == selected_speaker:
            speaker_mood = s
            break

    if not speaker_mood:
        st.error(f"No mood data found for {selected_speaker}")
        return

    # Final mood rating
    final_rating = speaker_mood.get("final_rating", {})
    col1, col2, col3 = st.columns(3)

    with col1:
        valence = final_rating.get("valence", 0)
        st.metric("Overall Valence", f"{valence:.2f}", help="0=Negative, 1=Positive")
    with col2:
        label = final_rating.get("label", "Unknown")
        st.metric("Mood Label", label)
    with col3:
        confidence = final_rating.get("confidence", 0)
        st.metric("Confidence", f"{confidence:.2%}")

    # Valence gauge
    fig_valence = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=valence,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Valence (Positive ↔ Negative)"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.3], "color": "lightcoral"},
                    {"range": [0.3, 0.7], "color": "lightyellow"},
                    {"range": [0.7, 1], "color": "lightgreen"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 0.5,
                },
            },
        )
    )
    st.plotly_chart(fig_valence, use_container_width=True)

    # Timeline data
    timeline = speaker_mood.get("timeline", [])

    if timeline:
        st.markdown("### 📈 Mood Over Time")

        # Prepare timeline dataframe
        timeline_df = pd.DataFrame(timeline)

        # Create subplot with valence and arousal
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Valence Over Time", "Arousal Over Time"),
            vertical_spacing=0.15,
        )

        # Valence line
        fig.add_trace(
            go.Scatter(
                x=timeline_df.get("t", []),
                y=timeline_df.get("valence", []),
                mode="lines+markers",
                name="Valence",
                line=dict(color="blue", width=2),
                fill="tozeroy",
            ),
            row=1,
            col=1,
        )

        # Arousal line
        fig.add_trace(
            go.Scatter(
                x=timeline_df.get("t", []),
                y=timeline_df.get("arousal", []),
                mode="lines+markers",
                name="Arousal",
                line=dict(color="orange", width=2),
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Valence", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Arousal", range=[0, 1], row=2, col=1)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment distribution
        st.markdown("### 🏷️ Sentiment Distribution")
        sentiment_counts = timeline_df["sentiment"].value_counts()
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Text Sentiment Labels",
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

        # Sarcasm detection
        st.markdown("### 😏 Sarcasm Detection")
        sarcasm_data = []
        for t in timeline:
            sarcasm_info = t.get("sarcasm", {})
            sarcasm_data.append(
                {
                    "Time": t.get("t", 0),
                    "Is Sarcastic": sarcasm_info.get("is_sarcastic", False),
                    "Probability": sarcasm_info.get("prob", 0),
                    "Rationale": sarcasm_info.get("rationale", "N/A"),
                }
            )

        sarcasm_df = pd.DataFrame(sarcasm_data)
        sarcastic_count = sarcasm_df["Is Sarcastic"].sum()

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Sarcastic Moments", sarcastic_count)
        with col2:
            if sarcastic_count > 0:
                sarcastic_df = sarcasm_df[sarcasm_df["Is Sarcastic"]]
                st.dataframe(sarcastic_df, hide_index=True, use_container_width=True)
            else:
                st.info("No sarcastic moments detected")


def render_metrics_tab(result: dict[str, Any]) -> None:
    """Render the Metrics tab with analysis quality and performance stats.

    Args:
        result: Analysis result from backend.
    """
    st.header("📊 Analysis Metrics")

    metrics_data = result.get("metrics", {})
    models_data = result.get("models", {})

    # Performance metrics
    st.markdown("### ⚡ Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        processing_time = metrics_data.get("processing_time_sec", 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")

    with col2:
        rtf = metrics_data.get("rtf", 0)
        st.metric("Real-Time Factor", f"{rtf:.3f}x", help="Processing time / audio duration")

    with col3:
        transcript_duration = result.get("transcript", {}).get("duration_sec", 0)
        st.metric("Audio Duration", f"{transcript_duration:.1f}s")

    # Quality metrics
    st.markdown("### 🎯 Quality Metrics")
    col1, col2 = st.columns(2)

    with col1:
        wer = metrics_data.get("wer")
        if wer is not None:
            st.metric("Word Error Rate (WER)", f"{wer:.2%}", help="Lower is better")
        else:
            st.metric("Word Error Rate (WER)", "N/A", help="Reference transcript needed")

    with col2:
        der = metrics_data.get("der")
        if der is not None:
            st.metric("Diarization Error Rate (DER)", f"{der:.2%}", help="Lower is better")
        else:
            st.metric("Diarization Error Rate (DER)", "N/A", help="Reference diarization needed")

    # Model information
    st.markdown("### 🤖 Models Used")

    if models_data:
        model_rows = []
        for model_type, model_info in models_data.items():
            model_rows.append(
                {
                    "Type": model_type.upper(),
                    "Model": model_info.get("name", "Unknown"),
                    "Version": model_info.get("version", "N/A") or "N/A",
                }
            )

        model_df = pd.DataFrame(model_rows)
        st.dataframe(model_df, hide_index=True, use_container_width=True)
    else:
        st.info("No model information available")

    # Job metadata
    st.markdown("### 📋 Job Metadata")
    col1, col2 = st.columns(2)

    with col1:
        job_id = result.get("job_id", "Unknown")
        st.code(f"Job ID: {job_id}", language=None)

    with col2:
        created_at = result.get("created_at", "Unknown")
        st.code(f"Created: {created_at}", language=None)


def render_report_tab(job_id: str) -> None:
    """Render the Full Report tab with formatted Markdown report.

    Args:
        job_id: Job ID to retrieve report artifact.
    """
    st.header("📄 Full Report")

    try:
        with st.spinner("Loading report..."):
            report_md = get_artifact(job_id, "report_md")
            st.markdown(report_md)
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            st.warning("Report artifact not yet generated. This feature is coming soon!")
            st.info("The backend will generate a comprehensive Markdown report including all analysis results.")
        else:
            st.error(f"Failed to load report: {e}")
    except Exception as e:
        st.error(f"Error loading report: {e}")


def main() -> None:
    """Main dashboard application."""
    # Header
    st.title("🎙️ AI Conversation Insights Dashboard")
    st.markdown("Analyze meeting audio for speaker diarization, transcription, mood, and insights.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Backend health check
        health = check_backend_health()
        if health:
            st.success("✅ Backend Connected")
            st.json(health)
        else:
            st.error("❌ Backend Unreachable")
            st.warning(f"Please ensure FastAPI is running on {API_BASE_URL}")
            st.stop()

        st.divider()

        # Upload section
        st.header("📁 Upload Audio")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
        )

        # Analysis options
        with st.expander("Advanced Options"):
            expected_speakers = st.number_input(
                "Expected Speakers",
                min_value=2,
                max_value=10,
                value=None,
                help="Leave empty for auto-detection",
            )

            language_hint = st.selectbox(
                "Language Hint",
                options=["", "en", "es", "fr", "de", "zh", "ja", "ko", "hi", "ar"],
                format_func=lambda x: "Auto-detect" if x == "" else x.upper(),
                help="ISO 639-1 language code",
            )

            max_words = st.slider("Summary Max Words", 50, 1000, 250, 50)
            style = st.selectbox("Summary Style", ["concise", "detailed", "bullet"])
            sarcasm_sensitivity = st.selectbox(
                "Sarcasm Sensitivity",
                ["low", "balanced", "high"],
                index=1,
            )

        # Analyze button
        analyze_button = st.button("🚀 Analyze Audio", type="primary", use_container_width=True)

    # Main content area
    if uploaded_file is None:
        st.info("👈 Please upload an audio file to begin analysis")
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Upload** a `.wav` or `.mp3` file using the sidebar
        2. **Configure** analysis options (optional)
        3. **Click** "Analyze Audio" to process
        4. **Explore** results across multiple tabs:
           - **Summary**: Key decisions, action items, and insights
           - **Transcript**: Speaker-attributed dialogue with timeline
           - **Mood**: Emotion and sentiment analysis
           - **Metrics**: Quality and performance statistics
           - **Report**: Full Markdown report (if available)
        """)
        st.markdown("---")
        st.markdown("### 🔧 Requirements")
        st.markdown(f"""
        - FastAPI backend running on `{API_BASE_URL}`
        - Audio file under {MAX_FILE_SIZE_MB}MB
        - Supported formats: {', '.join(SUPPORTED_FORMATS)}
        """)
        return

    # File size check
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
        return

    st.info(f"📎 Selected: **{uploaded_file.name}** ({file_size_mb:.1f}MB)")

    # Analyze
    if analyze_button:
        try:
            with st.spinner("🔄 Analyzing audio... This may take a few minutes."):
                # Prepare request config
                request_config = {
                    "summarizer": {
                        "max_words": max_words,
                        "style": style,
                    },
                    "sarcasm_sensitivity": sarcasm_sensitivity,
                }

                # Call API
                start_time = time.time()
                result = upload_and_analyze(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    expected_speakers=expected_speakers if expected_speakers else None,
                    language_hint=language_hint if language_hint else None,
                    request_config=request_config,
                )
                elapsed_time = time.time() - start_time

                # Store in session state
                st.session_state.analysis_result = result
                st.session_state.analysis_time = elapsed_time

                st.success(f"✅ Analysis complete in {elapsed_time:.1f}s!")
                st.balloons()

        except requests.HTTPError as e:
            st.error(f"API Error: {e.response.status_code}")
            try:
                error_detail = e.response.json()
                st.json(error_detail)
            except Exception:
                st.code(e.response.text)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

    # Display results if available
    if "analysis_result" in st.session_state:
        result = st.session_state.analysis_result
        job_id = result.get("job_id", "unknown")

        st.markdown("---")

        # Tabs
        tabs = st.tabs(["📋 Summary", "💬 Transcript", "🎭 Mood", "📊 Metrics", "📄 Report"])

        with tabs[0]:
            render_summary_tab(result)

        with tabs[1]:
            render_transcript_tab(result)

        with tabs[2]:
            render_mood_tab(result)

        with tabs[3]:
            render_metrics_tab(result)

        with tabs[4]:
            render_report_tab(job_id)

        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"analysis_{job_id}.json",
                mime="application/json",
            )

        with col2:
            # Try to get report if available
            try:
                report_md = get_artifact(job_id, "report_md")
                st.download_button(
                    label="Download Report (MD)",
                    data=report_md,
                    file_name=f"report_{job_id}.md",
                    mime="text/markdown",
                )
            except Exception:
                st.button("Download Report (MD)", disabled=True, help="Report not available")


if __name__ == "__main__":
    main()
