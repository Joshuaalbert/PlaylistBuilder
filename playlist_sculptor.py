"""
Playlist Sculptor - Streamlit App Entry Point

A Streamlit application to sculpt playlists from YouTube URLs.
Uses yt-dlp to download audio, librosa for features, and JAX for an 11D autoencoder + discriminator.

This unified app combines features from both the original playlist_sculptor.py and app.py:
- Individual URL input and batch loading from text files
- Accept/Reject/Neutral song classification
- Feature autoencoder for song embeddings
- Playlist embedding (mean + covariance)
- Discriminator for personalized song recommendations
- Similarity matrix and latent space visualization
- Audio preview playback
- SQLite database for persistent storage with playlist support
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import librosa
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
import yt_dlp
from jax import random

from src.playlist_sculptor import db

# =========================
# Paths & Constants
# =========================

DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"

LATENT_DIM = 11
HIDDEN_DIM = 64
BATCH_SIZE = 32
SAMPLE_RATE = 22050
MIN_STD_THRESHOLD = 1e-6  # Minimum threshold for standard deviation normalization


@dataclass
class FeatureAEParams:
    """Parameters for the feature autoencoder."""
    W_enc: jax.Array
    b_enc: jax.Array
    W_dec: jax.Array
    b_dec: jax.Array
    mean: jax.Array
    std: jax.Array


@dataclass
class DiscriminatorParams:
    """Parameters for the discriminator."""
    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array


# =========================
# Utility Functions
# =========================

def ensure_dirs():
    """Create necessary directories."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_song_list_from_txt(txt_path: str) -> List[int]:
    """Load YouTube URLs from a text file and add to database.

    Returns list of song IDs.
    """
    with open(txt_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    song_ids = []
    for url in urls:
        song_id = db.add_song(url)
        song_ids.append(song_id)

    return song_ids


# =========================
# Audio Download
# =========================

def download_audio_for_song(song: db.Song) -> db.Song:
    """Download audio for a song using yt-dlp."""
    if song.audio_path and os.path.exists(song.audio_path):
        return song

    ensure_dirs()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(song.youtube_url, download=True)
        if info is None:
            return song
        vid_id = info.get("id")
        ext = info.get("ext", "m4a")
        audio_path = str(AUDIO_DIR / f"{vid_id}.{ext}")
        db.update_song_audio_path(song.id, audio_path)
        song.audio_path = audio_path
        return song


# =========================
# Feature Extraction
# =========================

def extract_features_from_audio(audio_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract comprehensive audio features using librosa."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    if len(y) == 0:
        raise RuntimeError(f"Empty audio: {audio_path}")

    duration = len(y) / sr

    # Rhythm / tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # librosa >= 0.10.0 returns tempo as numpy array, convert to scalar
    if isinstance(tempo, np.ndarray) and len(tempo) > 0:
        tempo = float(tempo[0])
    else:
        tempo = float(tempo) if not isinstance(tempo, np.ndarray) else 0.0
    if len(beats) > 1:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        ibi = np.diff(beat_times)
        beat_reg = np.std(ibi)
    else:
        beat_reg = 0.5

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = float(onset_env.mean())
    onset_var = float(onset_env.var())

    # Loudness / energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(rms.mean())
    rms_var = float(rms.var())
    loudness_db = 20.0 * np.log10(max(rms_mean, 1e-6))

    # Spectral shape
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()

    # Harmonic vs percussive
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.mean(np.abs(y_harm)))
    perc_energy = float(np.mean(np.abs(y_perc)))
    perc_ratio = perc_energy / (harm_energy + perc_energy + 1e-6)

    # Chroma (12)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Timbre: MFCC mean & var
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.var(axis=1)

    feats = np.concatenate([
        np.array([
            duration,
            tempo,
            beat_reg,
            onset_mean,
            onset_var,
            rms_mean,
            rms_var,
            loudness_db,
            centroid,
            bandwidth,
            rolloff,
            zcr,
            perc_ratio,
        ], dtype=np.float32),
        chroma_mean.astype(np.float32),
        mfcc_mean.astype(np.float32),
        mfcc_var.astype(np.float32),
    ])
    return feats


def extract_and_store_features_for_song(song: db.Song) -> Optional[np.ndarray]:
    """Extract and store features for a song in the database."""
    if song.features is not None:
        return song.features

    if not song.audio_path or not os.path.exists(song.audio_path):
        return None

    try:
        features = extract_features_from_audio(song.audio_path)
        db.update_song_features(song.id, features)
        return features
    except Exception:
        return None


def build_feature_matrix_for_songs(songs: List[db.Song]) -> np.ndarray:
    """Build feature matrix for a list of songs, extracting if needed."""
    all_feats = []
    for song in songs:
        if song.features is not None:
            all_feats.append(song.features)
        elif song.audio_path and os.path.exists(song.audio_path):
            features = extract_and_store_features_for_song(song)
            if features is not None:
                all_feats.append(features)
            else:
                all_feats.append(np.zeros(51, dtype=np.float32))
        else:
            all_feats.append(np.zeros(51, dtype=np.float32))

    if not all_feats:
        return np.array([])

    return np.stack(all_feats, axis=0)


# =========================
# Feature Autoencoder (JAX)
# =========================

def compute_feature_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of features."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < MIN_STD_THRESHOLD] = 1.0
    return mean, std


def init_feature_ae(rng_key: jax.Array, input_dim: int, latent_dim: int,
                    mean: np.ndarray, std: np.ndarray) -> FeatureAEParams:
    """Initialize feature autoencoder parameters."""
    k1, k2 = random.split(rng_key)
    W_enc = random.normal(k1, (latent_dim, input_dim)) * 0.01
    b_enc = jnp.zeros((latent_dim,))
    W_dec = random.normal(k2, (input_dim, latent_dim)) * 0.01
    b_dec = jnp.zeros((input_dim,))
    return FeatureAEParams(
        W_enc=W_enc,
        b_enc=b_enc,
        W_dec=W_dec,
        b_dec=b_dec,
        mean=jnp.array(mean),
        std=jnp.array(std),
    )


def ae_encode(params: FeatureAEParams, x: jax.Array) -> jax.Array:
    """Encode features to latent space."""
    x_norm = (x - params.mean) / params.std
    h = jnp.tanh(params.W_enc @ x_norm + params.b_enc)
    return h


def ae_decode(params: FeatureAEParams, z: jax.Array) -> jax.Array:
    """Decode latent space to features."""
    x_norm_hat = params.W_dec @ z + params.b_dec
    x_hat = x_norm_hat * params.std + params.mean
    return x_hat


def ae_batch_loss(params: FeatureAEParams, batch_x: jax.Array) -> jax.Array:
    """Compute autoencoder reconstruction loss."""
    def single_loss(x):
        z = ae_encode(params, x)
        x_hat = ae_decode(params, z)
        return jnp.mean((x_hat - x) ** 2)
    losses = jax.vmap(single_loss)(batch_x)
    return jnp.mean(losses)


ae_grad = jax.jit(jax.grad(ae_batch_loss))


def train_feature_ae(params: FeatureAEParams, features: np.ndarray,
                     num_epochs: int, lr: float, batch_size: int = BATCH_SIZE) -> FeatureAEParams:
    """Train the feature autoencoder."""
    X = jnp.array(features)
    n = X.shape[0]

    for epoch in range(num_epochs):
        perm = np.random.permutation(n)
        X_shuf = X[perm]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = X_shuf[start:end]
            grads = ae_grad(params, batch)
            params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

    return params


def save_feature_ae_to_db(params: FeatureAEParams):
    """Save autoencoder parameters to database."""
    params_dict = {
        "W_enc": np.array(params.W_enc),
        "b_enc": np.array(params.b_enc),
        "W_dec": np.array(params.W_dec),
        "b_dec": np.array(params.b_dec),
        "mean": np.array(params.mean),
        "std": np.array(params.std),
    }
    db.save_embedding_model(params_dict)


def load_feature_ae_from_db() -> Optional[FeatureAEParams]:
    """Load autoencoder parameters from database."""
    params_dict = db.load_embedding_model()
    if params_dict is None:
        return None
    return FeatureAEParams(
        W_enc=jnp.array(params_dict["W_enc"]),
        b_enc=jnp.array(params_dict["b_enc"]),
        W_dec=jnp.array(params_dict["W_dec"]),
        b_dec=jnp.array(params_dict["b_dec"]),
        mean=jnp.array(params_dict["mean"]),
        std=jnp.array(params_dict["std"]),
    )


def compute_track_embeddings(params: FeatureAEParams, features: np.ndarray) -> np.ndarray:
    """Compute track embeddings from features."""
    X = jnp.array(features)
    Z = jax.vmap(lambda x: ae_encode(params, x))(X)
    return np.array(Z)


# =========================
# Playlist Embedding
# =========================

def compute_playlist_embedding(track_embs: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Compute playlist embedding from accepted tracks."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None

    E = track_embs[idx]
    mu = E.mean(axis=0)
    centered = E - mu
    if E.shape[0] > 1:
        cov = centered.T @ centered / (E.shape[0] - 1)
    else:
        cov = np.eye(E.shape[1], dtype=np.float32) * 1e-6

    tri_idx = np.triu_indices(E.shape[1])
    cov_flat = cov[tri_idx]
    playlist_vec = np.concatenate([mu, cov_flat], axis=0)
    return playlist_vec.astype(np.float32)


# =========================
# Discriminator (JAX)
# =========================

def init_discriminator(rng_key: jax.Array, input_dim: int, hidden_dim: int) -> DiscriminatorParams:
    """Initialize discriminator parameters."""
    k1, k2 = random.split(rng_key)
    W1 = random.normal(k1, (hidden_dim, input_dim)) * 0.05
    b1 = jnp.zeros((hidden_dim,))
    W2 = random.normal(k2, (1, hidden_dim)) * 0.05
    b2 = jnp.zeros((1,))
    return DiscriminatorParams(W1=W1, b1=b1, W2=W2, b2=b2)


def disc_forward(params: DiscriminatorParams, x: jax.Array) -> jax.Array:
    """Forward pass through discriminator."""
    h = jnp.tanh(params.W1 @ x + params.b1)
    logit = params.W2 @ h + params.b2
    return logit[0]


def disc_batch_loss(params: DiscriminatorParams, X: jax.Array, y: jax.Array) -> jax.Array:
    """Compute discriminator loss."""
    def single_loss(x, y_i):
        logit = disc_forward(params, x)
        return jnp.mean(jnp.maximum(logit, 0) - logit * y_i + jnp.log1p(jnp.exp(-jnp.abs(logit))))
    losses = jax.vmap(single_loss)(X, y)
    return jnp.mean(losses)


disc_grad = jax.jit(jax.grad(disc_batch_loss))


def train_discriminator(params: DiscriminatorParams,
                        playlist_vec: np.ndarray,
                        track_embs: np.ndarray,
                        labels: np.ndarray,
                        num_epochs: int,
                        lr: float,
                        batch_size: int = BATCH_SIZE) -> DiscriminatorParams:
    """Train the discriminator."""
    P = jnp.array(playlist_vec)

    def build_input_vec(t_emb):
        return jnp.concatenate([P, t_emb], axis=0)

    inputs = jax.vmap(build_input_vec)(jnp.array(track_embs))
    y = jnp.array(labels)

    n = inputs.shape[0]
    for epoch in range(num_epochs):
        perm = np.random.permutation(n)
        X_shuf = inputs[perm]
        y_shuf = y[perm]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_x = X_shuf[start:end]
            batch_y = y_shuf[start:end]
            grads = disc_grad(params, batch_x, batch_y)
            params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

    return params


def save_discriminator_to_db(playlist_id: int, params: DiscriminatorParams):
    """Save discriminator parameters to database for a playlist."""
    params_dict = {
        "W1": np.array(params.W1),
        "b1": np.array(params.b1),
        "W2": np.array(params.W2),
        "b2": np.array(params.b2),
    }
    db.save_playlist_discriminator(playlist_id, params_dict)


def load_discriminator_from_db(playlist_id: int) -> Optional[DiscriminatorParams]:
    """Load discriminator parameters from database for a playlist."""
    params_dict = db.load_playlist_discriminator(playlist_id)
    if params_dict is None:
        return None
    return DiscriminatorParams(
        W1=jnp.array(params_dict["W1"]),
        b1=jnp.array(params_dict["b1"]),
        W2=jnp.array(params_dict["W2"]),
        b2=jnp.array(params_dict["b2"]),
    )


def predict_accept_probs(params: DiscriminatorParams,
                         playlist_vec: np.ndarray,
                         track_embs: np.ndarray) -> np.ndarray:
    """Predict acceptance probabilities for tracks."""
    P = jnp.array(playlist_vec)

    def forward_single(t_emb):
        x = jnp.concatenate([P, t_emb], axis=0)
        logit = disc_forward(params, x)
        prob = jax.nn.sigmoid(logit)
        return prob

    probs = jax.vmap(forward_single)(jnp.array(track_embs))
    return np.array(probs)


def compute_similarity(z1: np.ndarray, z2: np.ndarray) -> float:
    """Compute cosine similarity between two latent vectors."""
    z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)
    return float(np.dot(z1_norm, z2_norm))


def generate_playlist_order(similarity_matrix: np.ndarray) -> list:
    """Generate optimal playlist order using greedy nearest neighbor."""
    n_songs = len(similarity_matrix)
    if n_songs == 0:
        return []

    order = [0]
    remaining = set(range(1, n_songs))

    while remaining:
        current = order[-1]
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


# =========================
# Streamlit App
# =========================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Playlist Sculptor",
        page_icon="🎵",
        layout="wide",
    )

    ensure_dirs()
    db.init_database()

    st.title("🎵 Playlist Sculptor")
    st.markdown(
        """
        Sculpt your perfect playlist from YouTube URLs using machine learning!

        This app uses:
        - **yt-dlp** to download audio from YouTube
        - **librosa** to extract audio features
        - **JAX** to train an 11D autoencoder + discriminator for song recommendations
        - **SQLite** database for persistent storage with multi-playlist support
        """
    )

    # Initialize playlist selection in session state
    if "current_playlist_id" not in st.session_state:
        st.session_state["current_playlist_id"] = None

    # Sidebar navigation
    st.sidebar.title("Navigation")

    # Playlist selector in sidebar
    render_playlist_selector()

    # Refresh cache button
    if st.sidebar.button("🔄 Refresh Cache"):
        db.clear_all_caches()
        st.rerun()

    page = st.sidebar.radio(
        "Select a page",
        ["Manage Playlists", "Song Library", "Extract Features", "Train Models",
         "Sculpt Playlist", "Visualizations", "About"],
    )

    if page == "Manage Playlists":
        render_manage_playlists_page()
    elif page == "Song Library":
        render_song_library_page()
    elif page == "Extract Features":
        render_extract_features_page()
    elif page == "Train Models":
        render_train_models_page()
    elif page == "Sculpt Playlist":
        render_sculpt_playlist_page()
    elif page == "Visualizations":
        render_visualizations_page()
    elif page == "About":
        render_about_page()


def render_playlist_selector():
    """Render playlist selector in sidebar."""
    st.sidebar.subheader("🎵 Current Playlist")

    playlists = db.get_all_playlists()

    if not playlists:
        st.sidebar.info("No playlists yet. Create one first.")
        return

    playlist_options = {p.name: p.id for p in playlists}
    playlist_names = list(playlist_options.keys())

    # Find current selection
    current_idx = 0
    if st.session_state["current_playlist_id"]:
        for i, p in enumerate(playlists):
            if p.id == st.session_state["current_playlist_id"]:
                current_idx = i
                break

    selected_name = st.sidebar.selectbox(
        "Select playlist",
        playlist_names,
        index=current_idx,
        key="playlist_selector",
    )

    if selected_name:
        st.session_state["current_playlist_id"] = playlist_options[selected_name]


def render_manage_playlists_page():
    """Render the Manage Playlists page."""
    st.header("📋 Manage Playlists")

    # Create new playlist
    st.subheader("Create New Playlist")
    col1, col2 = st.columns([2, 1])
    with col1:
        new_name = st.text_input("Playlist name", placeholder="My Awesome Playlist")
    with col2:
        new_desc = st.text_input("Description (optional)", placeholder="A mix of...")

    if st.button("Create Playlist", type="primary"):
        if new_name:
            playlist_id = db.create_playlist(new_name, new_desc if new_desc else None)
            db.clear_playlists_cache()
            st.success(f"Created playlist '{new_name}' (ID: {playlist_id})")
            st.session_state["current_playlist_id"] = playlist_id
            st.rerun()
        else:
            st.warning("Please enter a playlist name")

    st.divider()

    # List existing playlists
    st.subheader("Existing Playlists")
    playlists = db.get_all_playlists()

    if not playlists:
        st.info("No playlists yet. Create one above!")
        return

    for playlist in playlists:
        songs_in_playlist = db.get_songs_in_playlist(playlist.id)
        num_songs = len(songs_in_playlist)
        num_accepted = sum(1 for _, ps in songs_in_playlist if ps.accepted)
        num_rejected = sum(1 for _, ps in songs_in_playlist if ps.rejected)

        with st.expander(f"📁 {playlist.name} ({num_songs} songs)"):
            st.write(f"**Description:** {playlist.description or 'No description'}")
            st.write(f"**Songs:** {num_songs} total, {num_accepted} accepted, {num_rejected} rejected")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select", key=f"select_{playlist.id}"):
                    st.session_state["current_playlist_id"] = playlist.id
                    st.rerun()
            with col2:
                if st.button("Delete", key=f"delete_{playlist.id}", type="secondary"):
                    db.delete_playlist(playlist.id)
                    db.clear_playlists_cache()
                    db.clear_playlist_songs_cache()
                    if st.session_state["current_playlist_id"] == playlist.id:
                        st.session_state["current_playlist_id"] = None
                    st.rerun()


def render_song_library_page():
    """Render the Song Library page (global songs)."""
    st.header("📚 Song Library")

    # Add individual URL
    st.subheader("Add Individual Song")
    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter a YouTube video URL to add to your library",
    )

    if st.button("Add Song", type="primary"):
        if url:
            existing = db.get_song_by_url(url)
            if existing:
                st.warning("This URL is already in your library!")
            else:
                song_id = db.add_song(url)
                db.clear_songs_cache()
                st.success(f"Added song (ID: {song_id})!")
                st.rerun()
        else:
            st.warning("Please enter a YouTube URL")

    st.divider()

    # Load from text file
    st.subheader("Load from Text File")
    txt_path = st.text_input(
        "Path to text file with YouTube URLs",
        value="all_songs.txt",
        help="Text file with one YouTube URL per line",
    )

    if st.button("Load from File"):
        try:
            song_ids = load_song_list_from_txt(txt_path)
            db.clear_songs_cache()
            st.success(f"Loaded {len(song_ids)} songs from {txt_path}")
            st.rerun()
        except FileNotFoundError:
            st.error(f"File not found: {txt_path}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.divider()

    # Display all songs
    songs = db.get_all_songs()
    st.subheader(f"📚 All Songs ({len(songs)} total)")

    if not songs:
        st.info("No songs in library. Add URLs above or load from a text file.")
        return

    # Option to add songs to current playlist
    current_playlist_id = st.session_state.get("current_playlist_id")

    for song in songs:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            status = "✅ Has audio" if song.audio_path and os.path.exists(song.audio_path) else "⏳ No audio"
            features_status = "📊 Has features" if song.features is not None else ""
            st.write(f"**{song.id}**: {song.youtube_url[:50]}... {status} {features_status}")
        with col2:
            if current_playlist_id:
                ps = db.get_playlist_song_status(current_playlist_id, song.id)
                if ps is None:
                    if st.button("Add to playlist", key=f"add_to_pl_{song.id}"):
                        db.add_song_to_playlist(current_playlist_id, song.id)
                        db.clear_playlist_songs_cache()
                        st.rerun()
                else:
                    st.write("✓ In playlist")
        with col3:
            if st.button("🗑️", key=f"delete_song_{song.id}"):
                db.delete_song(song.id)
                db.clear_songs_cache()
                db.clear_playlist_songs_cache()
                st.rerun()


def render_extract_features_page():
    """Render the Extract Features page."""
    st.header("🎧 Extract Features")

    songs = db.get_all_songs()

    if not songs:
        st.warning("No songs in library. Go to 'Song Library' page first.")
        return

    st.write(f"Total songs: {len(songs)}")

    # Show download/extract status
    downloaded = sum(1 for s in songs if s.audio_path and os.path.exists(s.audio_path))
    with_features = sum(1 for s in songs if s.features is not None)
    st.write(f"Downloaded: {downloaded}/{len(songs)}")
    st.write(f"With features: {with_features}/{len(songs)}")

    if st.button("Download Audio & Extract Features", type="primary"):
        with st.spinner("Processing songs..."):
            progress_bar = st.progress(0)
            for i, song in enumerate(songs):
                progress_bar.progress((i + 1) / len(songs))

                # Download if needed
                if not song.audio_path or not os.path.exists(song.audio_path):
                    song = download_audio_for_song(song)

                # Extract features if needed
                if song.features is None and song.audio_path and os.path.exists(song.audio_path):
                    extract_and_store_features_for_song(song)

            db.clear_songs_cache()
            st.success("Processing complete!")
            st.rerun()


def render_train_models_page():
    """Render the Train Models page."""
    st.header("🧠 Train Models")

    # Get all songs with features for training embedding model
    features, song_ids = db.get_all_songs_feature_matrix()

    if len(features) == 0:
        st.warning("No songs with features. Go to 'Extract Features' page first.")
        return

    num_songs, feat_dim = features.shape
    st.write(f"Songs with features: {num_songs}, Feature dimension: {feat_dim}")

    # Feature Autoencoder Section (trained on ALL songs)
    st.subheader("Feature Autoencoder (Shared Embedding Model)")
    st.info("The embedding model is trained on ALL songs in the library, not just songs in the current playlist.")

    ae_params = load_feature_ae_from_db()
    if ae_params is not None:
        st.success("✅ Feature AE model loaded from database")
    else:
        st.info("No feature AE model found. Train one below.")

    col1, col2 = st.columns(2)
    with col1:
        ae_epochs = st.number_input("AE Epochs", min_value=1, max_value=500, value=50)
    with col2:
        ae_lr = st.number_input("AE Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")

    if st.button("Train Feature AE on All Songs"):
        with st.spinner("Training Feature AE on all songs..."):
            mean, std = compute_feature_stats(features)
            rng = random.PRNGKey(0)
            if ae_params is None or ae_params.W_enc.shape[1] != feat_dim:
                ae_params = init_feature_ae(rng, feat_dim, LATENT_DIM, mean, std)
            else:
                ae_params = FeatureAEParams(
                    W_enc=ae_params.W_enc,
                    b_enc=ae_params.b_enc,
                    W_dec=ae_params.W_dec,
                    b_dec=ae_params.b_dec,
                    mean=jnp.array(mean),
                    std=jnp.array(std),
                )
            ae_params = train_feature_ae(ae_params, features, num_epochs=ae_epochs, lr=ae_lr)
            save_feature_ae_to_db(ae_params)
            st.success("Feature AE trained and saved to database!")
            st.rerun()

    st.divider()

    # Discriminator Section (per playlist)
    st.subheader("Discriminator (Per-Playlist Recommendations)")

    current_playlist_id = st.session_state.get("current_playlist_id")
    if not current_playlist_id:
        st.warning("Select a playlist first to train its discriminator.")
        return

    playlist = db.get_playlist(current_playlist_id)
    if not playlist:
        st.error("Selected playlist not found.")
        return

    st.write(f"Training discriminator for: **{playlist.name}**")

    ae_params = load_feature_ae_from_db()
    if ae_params is None:
        st.warning("Train the Feature AE first before training the discriminator.")
        return

    # Get songs in the current playlist with their features
    playlist_features, playlist_song_ids, accepted_flags, rejected_flags = db.get_playlist_feature_matrix(current_playlist_id)

    if len(playlist_features) == 0:
        st.warning("No songs with features in this playlist. Add songs and extract features first.")
        return

    track_embs = compute_track_embeddings(ae_params, playlist_features)
    accepted_mask = np.array(accepted_flags, dtype=bool)
    rejected_mask = np.array(rejected_flags, dtype=bool)

    st.write(f"Songs in playlist: {len(playlist_song_ids)}")
    st.write(f"Accepted: {sum(accepted_mask)}, Rejected: {sum(rejected_mask)}")

    playlist_vec = compute_playlist_embedding(track_embs, accepted_mask)
    input_dim_disc = (LATENT_DIM + (LATENT_DIM * (LATENT_DIM + 1)) // 2) + LATENT_DIM

    disc_params = load_discriminator_from_db(current_playlist_id)
    if disc_params is not None:
        st.success("✅ Discriminator model loaded for this playlist")
    else:
        st.info("No discriminator model found for this playlist. Train one below.")

    col1, col2 = st.columns(2)
    with col1:
        disc_epochs = st.number_input("Disc Epochs", min_value=1, max_value=1000, value=100)
    with col2:
        disc_lr = st.number_input("Disc Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")

    if st.button("Train Discriminator"):
        pos_idx = np.where(accepted_mask)[0]
        neg_idx = np.where(rejected_mask)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0 or playlist_vec is None:
            st.error("Need at least one accepted AND one rejected song to train discriminator.")
        else:
            with st.spinner("Training Discriminator..."):
                X_embs = np.concatenate([track_embs[pos_idx], track_embs[neg_idx]], axis=0)
                y_labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))], axis=0)
                rng = random.PRNGKey(42)
                if disc_params is None:
                    disc_params = init_discriminator(rng, input_dim_disc, HIDDEN_DIM)
                disc_params = train_discriminator(disc_params, playlist_vec, X_embs, y_labels,
                                                  num_epochs=disc_epochs, lr=disc_lr)
                save_discriminator_to_db(current_playlist_id, disc_params)
                st.success("Discriminator trained and saved for this playlist!")
                st.rerun()


def render_sculpt_playlist_page():
    """Render the Sculpt Playlist page with accept/reject workflow."""
    st.header("🎨 Sculpt Playlist")

    current_playlist_id = st.session_state.get("current_playlist_id")
    if not current_playlist_id:
        st.warning("Select a playlist first from the sidebar.")
        return

    playlist = db.get_playlist(current_playlist_id)
    if not playlist:
        st.error("Selected playlist not found.")
        return

    st.write(f"Working on: **{playlist.name}**")

    # Get songs in the current playlist
    playlist_songs = db.get_songs_in_playlist(current_playlist_id)

    if not playlist_songs:
        st.warning("No songs in this playlist. Add songs from the 'Song Library' page.")
        return

    # Get feature matrix
    playlist_features, playlist_song_ids, accepted_flags, rejected_flags = db.get_playlist_feature_matrix(current_playlist_id)

    if len(playlist_features) == 0:
        st.warning("No songs with features in this playlist. Extract features first.")
        return

    ae_params = load_feature_ae_from_db()
    if ae_params is None:
        st.warning("Train the Feature AE first on 'Train Models' page.")
        return

    track_embs = compute_track_embeddings(ae_params, playlist_features)
    accepted_mask = np.array(accepted_flags, dtype=bool)
    playlist_vec = compute_playlist_embedding(track_embs, accepted_mask)

    disc_params = load_discriminator_from_db(current_playlist_id)

    # Song Suggestions
    st.subheader("🎯 Song Suggestions")

    # Build song lookup
    song_id_to_idx = {sid: idx for idx, sid in enumerate(playlist_song_ids)}

    if playlist_vec is not None and disc_params is not None:
        probs = predict_accept_probs(disc_params, playlist_vec, track_embs)
        neutral_items = []
        for song, ps in playlist_songs:
            if not ps.accepted and not ps.rejected and song.id in song_id_to_idx:
                idx = song_id_to_idx[song.id]
                neutral_items.append((song, ps, float(probs[idx])))
        neutral_items.sort(key=lambda x: x[2], reverse=True)
        st.info("Showing probability-ranked suggestions from discriminator")
    else:
        neutral_items = [(song, ps, 0.5) for song, ps in playlist_songs
                        if not ps.accepted and not ps.rejected]
        st.info("Discriminator not trained yet. Showing all neutral songs.")

    max_to_show = st.number_input("Max suggestions to show", min_value=1, max_value=50, value=10)

    for song, ps, prob in neutral_items[:max_to_show]:
        st.markdown(f"**Song {song.id}** — {song.youtube_url} (p≈{prob:.2f})")

        # Audio preview
        if song.audio_path and os.path.exists(song.audio_path):
            try:
                with open(song.audio_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/m4a")
            except Exception:
                pass

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ Accept", key=f"accept_{song.id}"):
                db.update_song_status_in_playlist(current_playlist_id, song.id, accepted=True, rejected=False)
                db.clear_playlist_songs_cache()
                st.rerun()
        with c2:
            if st.button("❌ Reject", key=f"reject_{song.id}"):
                db.update_song_status_in_playlist(current_playlist_id, song.id, accepted=False, rejected=True)
                db.clear_playlist_songs_cache()
                st.rerun()
        with c3:
            if st.button("🗑️ Remove", key=f"remove_{song.id}"):
                db.remove_song_from_playlist(current_playlist_id, song.id)
                db.clear_playlist_songs_cache()
                st.rerun()

    st.divider()

    # Reconsider rejected
    st.subheader("🔄 Reconsider Rejected Songs")

    rejected_songs = [(song, ps) for song, ps in playlist_songs if ps.rejected]
    if rejected_songs:
        for song, ps in rejected_songs:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"{song.id}: {song.youtube_url[:50]}...")
            with c2:
                if st.button("Make Neutral", key=f"neutral_{song.id}"):
                    db.update_song_status_in_playlist(current_playlist_id, song.id, accepted=False, rejected=False)
                    db.clear_playlist_songs_cache()
                    st.rerun()
    else:
        st.info("No rejected songs to reconsider.")

    st.divider()

    # Show accepted songs
    st.subheader("✅ Accepted Songs")
    accepted_songs = [(song, ps) for song, ps in playlist_songs if ps.accepted]
    if accepted_songs:
        for song, ps in accepted_songs:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"{song.id}: {song.youtube_url[:50]}...")
            with c2:
                if st.button("Remove from accepted", key=f"unaccept_{song.id}"):
                    db.update_song_status_in_playlist(current_playlist_id, song.id, accepted=False, rejected=False)
                    db.clear_playlist_songs_cache()
                    st.rerun()
    else:
        st.info("No accepted songs yet.")


def render_visualizations_page():
    """Render visualizations page with similarity matrix and latent space."""
    st.header("📊 Visualizations")

    current_playlist_id = st.session_state.get("current_playlist_id")
    if not current_playlist_id:
        st.warning("Select a playlist first from the sidebar.")
        return

    playlist = db.get_playlist(current_playlist_id)
    if not playlist:
        st.error("Selected playlist not found.")
        return

    st.write(f"Visualizing: **{playlist.name}**")

    # Get playlist data
    playlist_features, playlist_song_ids, accepted_flags, rejected_flags = db.get_playlist_feature_matrix(current_playlist_id)

    if len(playlist_features) == 0:
        st.warning("No songs with features in this playlist.")
        return

    ae_params = load_feature_ae_from_db()
    if ae_params is None:
        st.warning("Train the Feature AE first.")
        return

    track_embs = compute_track_embeddings(ae_params, playlist_features)
    n_songs = len(track_embs)

    # Get playlist songs for status info
    playlist_songs = db.get_songs_in_playlist(current_playlist_id)
    song_id_to_status = {song.id: ps for song, ps in playlist_songs}

    # Similarity Matrix
    st.subheader("Song Similarity Matrix")

    similarity_matrix = np.zeros((n_songs, n_songs))
    for i in range(n_songs):
        for j in range(n_songs):
            similarity_matrix[i, j] = compute_similarity(track_embs[i], track_embs[j])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(n_songs))
    ax.set_yticks(range(n_songs))
    ax.set_xticklabels([f"{sid}" for sid in playlist_song_ids], fontsize=8)
    ax.set_yticklabels([f"{sid}" for sid in playlist_song_ids], fontsize=8)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Song Similarity (Cosine Distance in Latent Space)")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Suggested Playlist Order
    st.subheader("🎶 Suggested Playlist Order")
    st.markdown("Songs ordered by similarity for smooth transitions:")

    accepted_mask = np.array(accepted_flags, dtype=bool)
    accepted_indices = np.where(accepted_mask)[0]

    if len(accepted_indices) > 1:
        accepted_sim_matrix = similarity_matrix[np.ix_(accepted_indices, accepted_indices)]
        order_in_accepted = generate_playlist_order(accepted_sim_matrix)
        playlist_order = [accepted_indices[i] for i in order_in_accepted]

        for rank, idx in enumerate(playlist_order):
            song_id = playlist_song_ids[idx]
            song = db.get_song(song_id)
            if song:
                st.write(f"{rank + 1}. **Song {song_id}** - {song.youtube_url[:60]}...")
    elif len(accepted_indices) == 1:
        song_id = playlist_song_ids[accepted_indices[0]]
        song = db.get_song(song_id)
        if song:
            st.write(f"1. **Song {song_id}** - {song.youtube_url[:60]}...")
    else:
        st.info("Accept some songs to see the optimized playlist order.")

    st.divider()

    # Interactive Embedding Visualization
    st.subheader("📈 Interactive Embedding Visualization")
    st.markdown(
        "Explore the 11D song embeddings by selecting which dimensions to visualize. "
        "Hover over points to see song details."
    )

    if track_embs.shape[1] >= 2:
        latent_dim = track_embs.shape[1]
        dim_options = [f"Dimension {i + 1}" for i in range(latent_dim)]

        col1, col2 = st.columns(2)
        with col1:
            x_dim = st.selectbox("X-axis dimension", dim_options, index=0, key="emb_x_dim")
        with col2:
            y_dim = st.selectbox("Y-axis dimension", dim_options, index=1, key="emb_y_dim")

        x_idx = dim_options.index(x_dim)
        y_idx = dim_options.index(y_dim)

        # Prepare data for plotly
        statuses = []
        urls = []
        song_ids_list = []
        for sid in playlist_song_ids:
            ps = song_id_to_status.get(sid)
            song = db.get_song(sid)
            if ps and ps.accepted:
                statuses.append("Accepted")
            elif ps and ps.rejected:
                statuses.append("Rejected")
            else:
                statuses.append("Neutral")
            urls.append(song.youtube_url if song else "URL not available")
            song_ids_list.append(f"Song {sid}")

        # Create plotly scatter plot with hover info
        fig_interactive = px.scatter(
            x=track_embs[:, x_idx],
            y=track_embs[:, y_idx],
            color=statuses,
            color_discrete_map={
                "Accepted": "green",
                "Rejected": "red",
                "Neutral": "gray",
            },
            hover_name=song_ids_list,
            hover_data={
                "URL": urls,
                "Status": statuses,
            },
            labels={
                "x": x_dim,
                "y": y_dim,
                "color": "Status",
            },
            title=f"Song Embeddings: {x_dim} vs {y_dim}",
        )

        fig_interactive.update_traces(marker=dict(size=12, opacity=0.7))
        fig_interactive.update_layout(
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
        )

        st.plotly_chart(fig_interactive, use_container_width=True)


def render_about_page():
    """Render the About page."""
    st.header("ℹ️ About Playlist Sculptor")

    st.markdown(
        """
        ## Overview

        Playlist Sculptor is a Python application that helps you create cohesive playlists
        from YouTube videos using machine learning.

        ## How It Works

        1. **Manage Playlists**: Create and manage multiple playlists
        2. **Song Library**: Add YouTube URLs individually or load from a text file
        3. **Extract Features**: Download audio and compute rich audio features using librosa:
           - Rhythm: tempo, beat regularity, onset strength
           - Loudness: RMS, loudness in dB
           - Spectral: centroid, bandwidth, rolloff, zero crossing rate
           - Tonal: 12-dimensional chroma features
           - Timbre: 13 MFCCs (mean and variance)
           - Harmonic/percussive ratio
        4. **Train Models**:
           - Feature AE (shared): Learn 11D song embeddings from ALL songs
           - Discriminator (per playlist): Personalized recommendations
        5. **Sculpt Playlist**: Accept/reject songs to refine recommendations
        6. **Visualize**: Similarity matrix and latent space plots

        ## Key Features

        - **Multi-playlist support**: Same song can appear in multiple playlists
        - **SQLite database**: Persistent storage with automatic updates
        - **Shared embedding model**: One autoencoder trained on all songs
        - **Per-playlist discriminators**: Each playlist has its own trained recommender
        - **Streamlit caching**: Efficient database access with manual refresh

        ## Technology Stack

        - **Streamlit**: Web interface
        - **yt-dlp**: YouTube audio download
        - **librosa**: Audio feature extraction
        - **JAX**: Neural network training (pure SGD, no optax)
        - **SQLite**: Persistent database storage

        ## Author

        Joshua Albert
        """
    )


if __name__ == "__main__":
    main()
