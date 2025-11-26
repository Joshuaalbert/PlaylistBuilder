# Playlist Sculptor

🎵 A Python 3 Streamlit app to sculpt playlists from YouTube URLs.

## Overview

Playlist Sculptor helps you create cohesive playlists from YouTube videos using machine learning. It downloads audio, extracts features, and uses a JAX-based 11D autoencoder with discriminator to learn song representations for intelligent playlist ordering.

## Features

- **YouTube Audio Download**: Uses yt-dlp to download audio from YouTube URLs
- **Audio Feature Extraction**: Uses librosa to extract:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Chroma features
  - Spectral centroid and rolloff
  - Zero crossing rate
  - Tempo
- **Machine Learning**: JAX-based 11D autoencoder with adversarial training (no optax, pure SGD)
- **Playlist Sculpting**: Orders songs by similarity for smooth listening transitions

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
├── playlist_sculptor.py          # Streamlit entry point
├── src/
│   └── playlist_sculptor/
│       ├── __init__.py
│       └── playlist_sculptor.py  # Core module with main()
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

## How It Works

1. **Add Songs**: Enter YouTube URLs to download and analyze songs
2. **Train Model**: Train the 11D autoencoder to learn song representations
3. **Sculpt Playlist**: View similarity matrix and get optimized playlist ordering

## Author

Joshua Albert

## License

MIT License
