# 🎙️ AI Conversation Insights Dashboard

Interactive Streamlit-based visualization layer for the AI Conversation Insights Service.

## Features

- 📁 **File Upload**: Drag and drop `.wav`, `.mp3`, `.m4a`, or `.flac` files
- 📋 **Summary View**: Key decisions, action items, disagreements, and risks
- 💬 **Transcript**: Speaker-attributed dialogue with timeline visualization
- 🎭 **Mood Analysis**: Valence/arousal tracking with emotion and sentiment
- 📊 **Metrics**: Performance stats (WER, DER, RTF) and model information
- 📄 **Full Report**: Formatted Markdown report export
- 💾 **Export**: Download results as JSON or Markdown

## Installation

### Option 1: Using Poetry (Recommended)

```bash
# Install dashboard dependencies
poetry install --with dashboard

# Run the dashboard
poetry run streamlit run dashboard.py
```

### Option 2: Using pip

```bash
# Install from requirements
pip install streamlit plotly pandas requests

# Run the dashboard
streamlit run dashboard.py
```

## Usage

### 1. Start the FastAPI Backend

First, ensure the backend service is running:

```bash
# Terminal 1: Start backend
make run

# Or manually
poetry run uvicorn insightsvc.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

The backend should be accessible at `http://localhost:8000`

### 2. Launch the Dashboard

```bash
# Terminal 2: Start dashboard
poetry run streamlit run dashboard.py

# Or if using pip
streamlit run dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

### 3. Analyze Audio

1. **Upload** an audio file using the sidebar
2. **Configure** advanced options (optional):
   - Expected number of speakers
   - Language hint
   - Summary style and length
   - Sarcasm detection sensitivity
3. **Click** "🚀 Analyze Audio"
4. **Explore** results across tabs:
   - **Summary**: Meeting insights and key points
   - **Transcript**: Full dialogue with speaker attribution
   - **Mood**: Emotion analysis with interactive charts
   - **Metrics**: Quality and performance statistics
   - **Report**: Comprehensive Markdown report

## Configuration

Edit the following constants in `dashboard.py` to customize:

```python
# API endpoint (change if backend runs on different host/port)
API_BASE_URL = "http://localhost:8000"

# Supported audio formats
SUPPORTED_FORMATS = ["wav", "mp3", "m4a", "flac"]

# Maximum file size in MB
MAX_FILE_SIZE_MB = 500
```

## Visualizations

### Summary Tab
- Decisions with confidence scores
- Action items with ownership
- Disagreements with intensity
- Risks and blockers
- Open questions

### Transcript Tab
- Speaker statistics (talk time, utterances, confidence)
- Timeline visualization (Gantt-style)
- Filterable full transcript
- Overlap indicators

### Mood Tab
- Overall valence gauge
- Valence/arousal timeline
- Sentiment distribution pie chart
- Sarcasm detection results

### Metrics Tab
- Processing time and real-time factor
- Word Error Rate (WER)
- Diarization Error Rate (DER)
- Model information table

## Requirements

- **Backend**: FastAPI service running on `http://localhost:8000`
- **Python**: 3.11+
- **Dependencies**:
  - `streamlit >= 1.28.0`
  - `plotly >= 5.18.0`
  - `pandas >= 2.1.0`
  - `requests >= 2.31.0`

## Troubleshooting

### Backend Unreachable

If you see "❌ Backend Unreachable":

1. Check that the backend is running: `curl http://localhost:8000/healthz`
2. Verify the port in `API_BASE_URL` matches your backend
3. Check firewall settings if using remote backend

### File Upload Errors

- **422 Validation Error**: Check that the audio file format is supported
- **File Too Large**: Reduce file size or increase `MAX_FILE_SIZE_MB`
- **Timeout**: For large files, the analysis may take several minutes

### Missing Visualizations

If charts don't appear:
- Check that `plotly` is installed: `pip show plotly`
- Clear browser cache and refresh
- Check browser console for JavaScript errors

## Development

### Adding New Visualizations

1. Create a new function in `dashboard.py`:
   ```python
   def render_my_custom_tab(result: dict[str, Any]) -> None:
       st.header("My Custom Tab")
       # Your visualization code
   ```

2. Add to tabs in `main()`:
   ```python
   tabs = st.tabs(["Summary", "Transcript", "Mood", "Metrics", "Report", "Custom"])
   with tabs[5]:
       render_my_custom_tab(result)
   ```

### Styling

Streamlit uses a theming system. Customize via `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## Examples

### Quick Test with Sample Data

```bash
# Start backend
make run

# In another terminal, start dashboard
streamlit run dashboard.py

# Upload tests/data/sample_2speakers_clean.wav
# Expected speakers: 2
# Click "Analyze Audio"
```

### API Integration

The dashboard uses these endpoints:

- `GET /healthz` - Check backend status
- `POST /analyze` - Upload and analyze audio
- `GET /artifacts/{job_id}/{artifact_name}` - Retrieve results

## License

Same as main project (see LICENSE).

## Contributing

See CONTRIBUTING.md for development guidelines.
