# Copilot Instructions for PlaylistBuilder

## Project Overview

Playlist Sculptor is a Python 3 Streamlit app that creates cohesive playlists from YouTube URLs using machine learning. It downloads audio, extracts features, and uses a JAX-based 11D autoencoder with discriminator to learn song representations for intelligent playlist ordering.

## Technology Stack

- **Python**: Version 3.9+
- **Web Framework**: Streamlit for the UI
- **Audio Processing**: yt-dlp for YouTube downloads, librosa for audio feature extraction
- **Machine Learning**: JAX/JAXlib (pure SGD, no optax)
- **Audio I/O**: soundfile
- **Visualization**: matplotlib
- **Linting**: ruff

## Project Structure

```
PlaylistBuilder/
├── playlist_sculptor.py          # Streamlit entry point
├── src/
│   └── playlist_sculptor/
│       ├── __init__.py
│       ├── app.py               # Streamlit UI and main application logic
│       └── playlist_sculptor.py  # Core module with audio processing utilities
├── data/                          # Audio files and models (gitignored)
├── scripts/
│   └── run_app.sh               # Run script
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
└── README.md
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use ruff for linting with a line length of 100 characters
- Target Python 3.9 compatibility

### Running the Application

```bash
# Using the run script
./scripts/run_app.sh

# Or directly with streamlit
streamlit run playlist_sculptor.py
```

### Installing Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install as editable package with dev dependencies
pip install -e ".[dev]"
```

### Linting

```bash
# Run ruff linter
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Testing

```bash
# Run tests with pytest
pytest

# Run tests with coverage
pytest --cov
```

## Key Components

### Audio Feature Extraction

The project extracts these audio features using librosa:
- MFCCs (Mel-frequency cepstral coefficients)
- Chroma features
- Spectral centroid and rolloff
- Zero crossing rate
- Tempo

### Machine Learning Model

The 11D autoencoder with adversarial training uses:
- Pure JAX implementation without optax
- Simple SGD optimizer
- Discriminator for adversarial training

## Best Practices

- Keep audio and model files in the `data/` directory (gitignored)
- Do not commit `.wav`, `.mp3`, or `.npz` files
- Ensure all new code passes ruff linting before committing
- Write docstrings for new functions and classes
- Handle exceptions gracefully, especially for network operations (YouTube downloads)
