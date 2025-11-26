# Playlist Sculptor

🎵 A Python 3 Streamlit app to sculpt playlists from YouTube URLs.

## Overview

Playlist Sculptor helps you create cohesive playlists from YouTube videos using machine learning. It downloads audio, extracts features, and uses a JAX-based 11D autoencoder with discriminator to learn song representations for intelligent playlist ordering and personalized song recommendations.

## Features

### Song Management
- **Add songs individually**: Enter YouTube URLs one at a time
- **Batch import from file**: Load multiple YouTube URLs from a text file (one URL per line)
- **Song state tracking**: Classify songs as Accepted, Rejected, or Neutral
- **Persistent metadata**: Song states and metadata saved to JSON for session persistence
- **Audio preview**: Listen to downloaded songs directly in the app

### Audio Feature Extraction
Uses librosa to extract rich audio features:
- **Rhythm**: Tempo, beat regularity, onset strength (mean/variance)
- **Loudness/Energy**: RMS mean/variance, loudness in dB
- **Spectral shape**: Centroid, bandwidth, rolloff, zero crossing rate
- **Tonal content**: 12-dimensional chroma features (chroma_cqt)
- **Timbre**: 13 MFCCs (mean and variance)
- **Harmonic/Percussive**: Harmonic vs percussive energy ratio

### Machine Learning Pipeline
1. **Feature Autoencoder (AE)**: 
   - Extracts self-supervised N=11 dimensional embeddings per song
   - Learns from clustering of all extracted audio features (~51 dimensions)
   - Linear encoder/decoder with feature normalization (z-score)
   - Trained with MSE reconstruction loss using pure JAX SGD
   
2. **Playlist Embedding** (77 dimensions for N=11):
   - Computes a playlist representation from accepted songs
   - **Mean**: N=11 dimensional mean of track embeddings
   - **Covariance**: Flattened upper triangular (including diagonal) of NxN covariance matrix
   - For N=11: 11 (mean) + 66 (upper triangular) = **77 features**
   
3. **Discriminator for Recommendations**:
   - Binary classifier that predicts if a song fits the playlist
   - Input: playlist embedding (77) + track embedding (11) = **88 dimensions**
   - Trained on accepted (positive) and rejected (negative) examples
   - Outputs probability score for neutral songs

### Playlist Sculpting
- **Song recommendations**: Probability-ranked suggestions for neutral songs
- **Accept/Reject workflow**: Iteratively refine your playlist with feedback
- **Reconsider rejected**: Option to restore rejected songs to neutral
- **Similarity matrix**: Visualize pairwise song similarities as heatmap
- **Latent space visualization**: 2D scatter plot of songs in embedding space
- **Greedy playlist ordering**: Optimal song sequence for smooth transitions

## Installation

```bash
# Clone the repository
git clone https://github.com/Joshuaalbert/PlaylistBuilder.git
cd PlaylistBuilder

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Usage

### Run the Streamlit App

```bash
# Using the run script
./scripts/run_app.sh

# Or directly with streamlit
streamlit run playlist_sculptor.py
```

### Workflow

1. **Load Songs**: 
   - Enter YouTube URLs individually, OR
   - Create a text file with URLs (one per line) and load in batch

2. **Download & Extract Features**:
   - Click to download audio and compute features for all songs

3. **Train Feature Autoencoder**:
   - Set epochs and learning rate
   - Trains the 11D embedding model

4. **Sculpt Your Playlist**:
   - Review suggested songs ranked by predicted fit
   - Accept songs you like, reject ones you don't
   - Train the discriminator on your feedback
   - Get refined recommendations based on your preferences

5. **Reconsider**: 
   - Optionally restore rejected songs to neutral for re-evaluation

### Using as a Library

```python
from playlist_sculptor.playlist_sculptor import (
    download_audio,
    extract_features,
    train_autoencoder,
    encode_features,
    compute_similarity,
)

# Download audio from YouTube
audio_path = download_audio("https://www.youtube.com/watch?v=...")

# Extract features
features = extract_features(audio_path)

# Train autoencoder (with multiple songs)
import numpy as np
features_array = np.array([f.combined for f in all_features])
enc_params, dec_params, disc_params = train_autoencoder(features_array)

# Encode songs to latent space
latent_codes = encode_features(enc_params, features_array)

# Compute similarity
similarity = compute_similarity(latent_codes[0], latent_codes[1])
```

## Project Structure

```
playlist_sculptor/
├── playlist_sculptor.py          # Main Streamlit app (all features consolidated)
├── src/
│   └── playlist_sculptor/
│       ├── __init__.py
│       └── playlist_sculptor.py  # Core module (alternative entry point)
├── data/                          # Audio files and models
│   └── .gitkeep
├── scripts/
│   └── run_app.sh               # Run script
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
├── LICENSE                      # License file
└── README.md                    # This file
```

## Requirements

- Python 3.9+
- streamlit>=1.28.0
- yt-dlp>=2023.10.0
- librosa>=0.10.0
- jax>=0.4.20
- jaxlib>=0.4.20
- numpy>=1.24.0
- soundfile>=0.12.0
- matplotlib>=3.7.0

## How It Works

1. **Add Songs**: Enter YouTube URLs or load from a text file
2. **Extract Features**: Download audio and compute audio features
3. **Train Feature AE**: Learn 11D song embeddings
4. **Accept/Reject Songs**: Provide feedback on suggested songs
5. **Train Discriminator**: Model learns your preferences
6. **Get Recommendations**: Probability-ranked song suggestions
7. **Sculpt Playlist**: View similarity matrix and optimal ordering

## Author

Joshua Albert

## License

MIT License
