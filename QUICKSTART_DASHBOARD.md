# Quick Start Guide: Testing the Dashboard

This guide will help you quickly test the new Streamlit dashboard.

## Prerequisites

✅ Dashboard dependencies installed (streamlit, plotly, pandas, requests)

## Step 1: Start the Backend

Open a terminal and start the FastAPI backend:

```bash
cd /Users/anushkaparthiban/convobrief
make run
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Keep this terminal running.

## Step 2: Start the Dashboard

Open a **second terminal** and start the Streamlit dashboard:

```bash
cd /Users/anushkaparthiban/convobrief
make dashboard
```

Or directly:
```bash
poetry run streamlit run dashboard.py
```

The dashboard will automatically open in your browser at: `http://localhost:8501`

## Step 3: Upload and Analyze

1. **Upload a file**: Click "Browse files" or drag-and-drop a `.wav` file
   - Try: `tests/data/sample_2speakers_clean.wav`
   
2. **Configure options** (optional):
   - Expected Speakers: `2`
   - Language Hint: `en`
   - Summary Max Words: `250`
   - Summary Style: `concise`
   - Sarcasm Sensitivity: `balanced`

3. **Click "🚀 Analyze Audio"**

4. **Wait** for analysis to complete (this is showing placeholder data for now)

5. **Explore** the results in different tabs:
   - **📋 Summary**: Decisions, action items, disagreements, risks
   - **💬 Transcript**: Speaker timeline and full dialogue
   - **🎭 Mood**: Valence/arousal charts and sentiment analysis
   - **📊 Metrics**: Processing time, WER, DER, model info
   - **📄 Report**: Full Markdown report (when available)

## What You Should See

### Summary Tab
- Overview text box
- Decisions with timestamps
- Action items with owners
- Disagreements with intensity
- Risks and open questions

### Transcript Tab
- Speaker statistics table
- Timeline (Gantt chart) of speakers
- Full transcript with filtering
- Speaker attribution and confidence

### Mood Tab
- Valence gauge (0-1 scale)
- Valence and arousal over time (line charts)
- Sentiment distribution (pie chart)
- Sarcasm detection results

### Metrics Tab
- Processing time and RTF
- Quality metrics (WER, DER)
- Model information table
- Job metadata

## Current Limitations

⚠️ **Note**: The backend currently returns **placeholder data** because the full ML pipeline is not yet implemented. You'll see:

- Summary: "TODO: Implement pipeline"
- Empty transcript (no words/utterances)
- Empty mood data
- Processing time will be very fast (~0.05s)

This is expected! The dashboard is fully functional and ready to display real data once the ML pipeline is implemented.

## Troubleshooting

### "Backend Unreachable"
- Make sure the FastAPI backend is running on port 8000
- Check: `curl http://localhost:8000/healthz`
- If it returns JSON with `"status": "healthy"`, backend is working

### Import Errors
If you see "ModuleNotFoundError: No module named 'streamlit'":
```bash
poetry run pip install streamlit plotly pandas requests
```

### Port Already in Use
If port 8501 is busy:
```bash
streamlit run dashboard.py --server.port 8502
```

## Next Steps

Once the ML pipeline is implemented (VAD, diarization, ASR, NLP), you'll be able to:

1. Upload real meeting audio
2. See actual transcripts with speaker attribution
3. View real mood analysis with valence/arousal
4. Get meaningful summaries with decisions and action items
5. Export full reports

## Testing with Sample Data

For now, you can test the dashboard's UI and workflow:

```bash
# Upload any .wav file from tests/data/
# Example:
tests/data/sample_2speakers_clean.wav
tests/data/sample_2speakers_noisy.wav
tests/data/sample_3speakers_overlap.wav
```

The dashboard will successfully process the upload and display the placeholder response structure.

## Screenshots

### Main Upload Screen
- Sidebar with file uploader
- Advanced options expander
- Backend health check indicator

### Results Tabs
- 5 tabs with comprehensive visualizations
- Interactive Plotly charts
- Filterable transcript view
- Download buttons for JSON/MD export

Enjoy exploring the dashboard! 🎉
