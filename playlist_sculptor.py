"""
Playlist Sculptor - Streamlit App Entry Point

A Streamlit application to sculpt playlists from YouTube URLs.
Uses yt-dlp to download audio, librosa for features, and JAX for an 11D autoencoder + discriminator.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Load the core module from src/ directory using importlib to avoid naming conflicts
_src_path = Path(__file__).parent / "src" / "playlist_sculptor" / "playlist_sculptor.py"
_spec = importlib.util.spec_from_file_location("core_module", _src_path)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)

# Import functions from the core module
download_audio = _core.download_audio
extract_features = _core.extract_features
train_autoencoder = _core.train_autoencoder
encode_features = _core.encode_features
compute_similarity = _core.compute_similarity
save_model = _core.save_model
load_model = _core.load_model
get_data_dir = _core.get_data_dir
LATENT_DIM = _core.LATENT_DIM


def init_session_state():
    """Initialize Streamlit session state."""
    if "songs" not in st.session_state:
        st.session_state.songs = []
    if "features" not in st.session_state:
        st.session_state.features = []
    if "latent_codes" not in st.session_state:
        st.session_state.latent_codes = []
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "enc_params" not in st.session_state:
        st.session_state.enc_params = None
    if "dec_params" not in st.session_state:
        st.session_state.dec_params = None
    if "disc_params" not in st.session_state:
        st.session_state.disc_params = None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Playlist Sculptor",
        page_icon="🎵",
        layout="wide",
    )

    init_session_state()

    st.title("🎵 Playlist Sculptor")
    st.markdown(
        """
        Sculpt your perfect playlist from YouTube URLs using machine learning!
        
        This app uses:
        - **yt-dlp** to download audio from YouTube
        - **librosa** to extract audio features
        - **JAX** to train an 11D autoencoder with discriminator for learning song representations
        """
    )

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Add Songs", "Train Model", "Sculpt Playlist", "About"],
    )

    if page == "Add Songs":
        render_add_songs_page()
    elif page == "Train Model":
        render_train_model_page()
    elif page == "Sculpt Playlist":
        render_sculpt_playlist_page()
    elif page == "About":
        render_about_page()


def render_add_songs_page():
    """Render the Add Songs page."""
    st.header("📥 Add Songs")
    st.markdown("Enter YouTube URLs to add songs to your collection.")

    # URL input
    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter a YouTube video URL to download and analyze",
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Add Song", type="primary"):
            if url:
                add_song(url)
            else:
                st.warning("Please enter a YouTube URL")

    # Display current songs
    st.subheader(f"📚 Song Collection ({len(st.session_state.songs)} songs)")

    if st.session_state.songs:
        for i, song in enumerate(st.session_state.songs):
            with st.expander(f"Song {i + 1}: {song['url'][:50]}..."):
                st.write(f"**URL:** {song['url']}")
                st.write(f"**Audio File:** {song['audio_path']}")
                if song.get("features") is not None:
                    st.write(f"**Features extracted:** {len(song['features'].combined)} dimensions")
                    st.write(f"**Tempo:** {song['features'].tempo:.1f} BPM")

                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.songs.pop(i)
                    if i < len(st.session_state.features):
                        st.session_state.features.pop(i)
                    if i < len(st.session_state.latent_codes):
                        st.session_state.latent_codes.pop(i)
                    st.rerun()
    else:
        st.info("No songs added yet. Enter a YouTube URL above to get started!")


def add_song(url: str):
    """Add a song from a YouTube URL."""
    with st.spinner("Downloading audio..."):
        try:
            audio_path = download_audio(url)
            st.success(f"Downloaded: {audio_path.name}")
        except Exception as e:
            st.error(f"Failed to download: {e}")
            return

    with st.spinner("Extracting features..."):
        try:
            features = extract_features(audio_path)
            st.success(f"Extracted {len(features.combined)} audio features")
        except Exception as e:
            st.error(f"Failed to extract features: {e}")
            return

    # Store in session state
    song_data = {
        "url": url,
        "audio_path": str(audio_path),
        "features": features,
    }
    st.session_state.songs.append(song_data)
    st.session_state.features.append(features.combined)

    # Reset model if songs change
    st.session_state.model_trained = False
    st.session_state.latent_codes = []

    st.success("Song added successfully!")
    st.rerun()


def render_train_model_page():
    """Render the Train Model page."""
    st.header("🧠 Train Model")
    st.markdown(
        """
        Train the 11D autoencoder with discriminator on your song collection.
        This will learn a compressed representation of your songs that captures their musical characteristics.
        """
    )

    num_songs = len(st.session_state.songs)
    st.info(f"Current collection: {num_songs} songs")

    if num_songs < 2:
        st.warning("Add at least 2 songs before training the model.")
        return

    # Training parameters
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider("Epochs", min_value=10, max_value=500, value=100, step=10)

    with col2:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.01,
        )

    if st.button("Train Model", type="primary"):
        train_model(epochs, learning_rate)

    # Model status
    st.subheader("Model Status")
    if st.session_state.model_trained:
        st.success("✅ Model trained and ready!")
        st.write(f"Latent dimension: {LATENT_DIM}")
        st.write(f"Songs encoded: {len(st.session_state.latent_codes)}")

        # Save model button
        if st.button("Save Model"):
            try:
                path = save_model(
                    st.session_state.enc_params,
                    st.session_state.dec_params,
                    st.session_state.disc_params,
                )
                st.success(f"Model saved to: {path}")
            except Exception as e:
                st.error(f"Failed to save model: {e}")

        # Load model button
        if st.button("Load Model"):
            try:
                enc, dec, disc = load_model()
                st.session_state.enc_params = enc
                st.session_state.dec_params = dec
                st.session_state.disc_params = disc
                st.session_state.model_trained = True

                # Re-encode features
                features_array = np.array(st.session_state.features)
                st.session_state.latent_codes = encode_features(enc, features_array).tolist()
                st.success("Model loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load model: {e}")
    else:
        st.info("Model not trained yet. Click 'Train Model' above.")


def train_model(epochs: int, learning_rate: float):
    """Train the autoencoder model."""
    features_array = np.array(st.session_state.features)

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Initializing model...")
    progress_bar.progress(10)

    try:
        status_text.text("Training autoencoder...")
        enc_params, dec_params, disc_params = train_autoencoder(
            features_array,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        progress_bar.progress(80)

        status_text.text("Encoding songs...")
        latent_codes = encode_features(enc_params, features_array)
        progress_bar.progress(90)

        # Store in session state
        st.session_state.enc_params = enc_params
        st.session_state.dec_params = dec_params
        st.session_state.disc_params = disc_params
        st.session_state.latent_codes = latent_codes.tolist()
        st.session_state.model_trained = True

        progress_bar.progress(100)
        status_text.text("Training complete!")

        st.success("Model trained successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Training failed: {e}")
        raise


def render_sculpt_playlist_page():
    """Render the Sculpt Playlist page."""
    st.header("🎨 Sculpt Playlist")
    st.markdown("Organize and sculpt your playlist based on song similarities.")

    if not st.session_state.model_trained:
        st.warning("Please train the model first on the 'Train Model' page.")
        return

    if len(st.session_state.songs) < 2:
        st.warning("Add more songs to sculpt a playlist.")
        return

    # Similarity matrix
    st.subheader("Song Similarity Matrix")
    latent_codes = np.array(st.session_state.latent_codes)
    n_songs = len(latent_codes)

    similarity_matrix = np.zeros((n_songs, n_songs))
    for i in range(n_songs):
        for j in range(n_songs):
            similarity_matrix[i, j] = compute_similarity(latent_codes[i], latent_codes[j])

    # Display as heatmap
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(similarity_matrix, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(n_songs))
    ax.set_yticks(range(n_songs))
    ax.set_xticklabels([f"Song {i + 1}" for i in range(n_songs)])
    ax.set_yticklabels([f"Song {i + 1}" for i in range(n_songs)])
    plt.colorbar(im, ax=ax, label="Similarity")
    ax.set_title("Song Similarity (Cosine Distance in Latent Space)")
    st.pyplot(fig)

    # Playlist ordering
    st.subheader("🎶 Suggested Playlist Order")
    st.markdown("Songs ordered by similarity for smooth transitions:")

    # Simple greedy ordering based on similarity
    playlist_order = generate_playlist_order(similarity_matrix)

    for i, song_idx in enumerate(playlist_order):
        song = st.session_state.songs[song_idx]
        st.write(f"{i + 1}. **Song {song_idx + 1}** - {song['url'][:60]}...")
        if song.get("features"):
            st.caption(f"   Tempo: {song['features'].tempo:.1f} BPM")

    # Latent space visualization
    st.subheader("📊 Latent Space Visualization")

    if latent_codes.shape[1] >= 2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(latent_codes[:, 0], latent_codes[:, 1], s=100, c=range(n_songs), cmap="tab10")
        for i in range(n_songs):
            ax2.annotate(f"Song {i + 1}", (latent_codes[i, 0], latent_codes[i, 1]))
        ax2.set_xlabel("Latent Dimension 1")
        ax2.set_ylabel("Latent Dimension 2")
        ax2.set_title("Songs in Latent Space (First 2 Dimensions)")
        st.pyplot(fig2)


def generate_playlist_order(similarity_matrix: np.ndarray) -> list:
    """Generate playlist order using greedy nearest neighbor."""
    n_songs = len(similarity_matrix)
    if n_songs == 0:
        return []

    # Start with first song
    order = [0]
    remaining = set(range(1, n_songs))

    while remaining:
        current = order[-1]
        # Find most similar remaining song
        best_next = None
        best_sim = -float("inf")

        for candidate in remaining:
            sim = similarity_matrix[current, candidate]
            if sim > best_sim:
                best_sim = sim
                best_next = candidate

        if best_next is not None:
            order.append(best_next)
            remaining.remove(best_next)

    return order


def render_about_page():
    """Render the About page."""
    st.header("ℹ️ About Playlist Sculptor")

    st.markdown(
        """
        ## Overview
        
        Playlist Sculptor is a Python application that helps you create cohesive playlists
        from YouTube videos using machine learning.
        
        ## How It Works
        
        1. **Download Audio**: Uses yt-dlp to download audio from YouTube URLs
        2. **Extract Features**: Uses librosa to extract audio features including:
           - MFCCs (Mel-frequency cepstral coefficients)
           - Chroma features
           - Spectral centroid and rolloff
           - Zero crossing rate
           - Tempo
        3. **Learn Representations**: Uses a JAX-based 11D autoencoder with adversarial training
           (discriminator) to learn compressed song representations
        4. **Sculpt Playlist**: Orders songs by similarity for smooth listening transitions
        
        ## Technology Stack
        
        - **Streamlit**: Web interface
        - **yt-dlp**: YouTube audio download
        - **librosa**: Audio feature extraction
        - **JAX**: Neural network training (without optax, using pure SGD)
        
        ## Project Structure
        
        ```
        playlist_sculptor/
        ├── playlist_sculptor.py          # Streamlit entry point
        ├── src/
        │   └── playlist_sculptor/
        │       └── playlist_sculptor.py  # Core module with main()
        ├── data/                          # Audio files and models
        ├── scripts/
        │   └── run_app.sh               # Run script
        ├── pyproject.toml               # Project configuration
        ├── requirements.txt             # Dependencies
        └── README.md                    # Documentation
        ```
        
        ## Author
        
        Joshua Albert
        """
    )


if __name__ == "__main__":
    main()
