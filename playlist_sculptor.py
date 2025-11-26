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
- Persistent song metadata storage
"""

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yt_dlp
from jax import random

# =========================
# Paths & Constants
# =========================

DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
FEATURES_PATH = DATA_DIR / "features.npy"
SONGS_META_PATH = DATA_DIR / "songs_meta.json"
AE_MODEL_PATH = DATA_DIR / "feature_ae.npz"
DISC_MODEL_PATH = DATA_DIR / "discriminator.npz"

LATENT_DIM = 11
HIDDEN_DIM = 64
BATCH_SIZE = 32
SAMPLE_RATE = 22050
MIN_STD_THRESHOLD = 1e-6  # Minimum threshold for standard deviation normalization


# =========================
# Dataclasses
# =========================

@dataclass
class SongMeta:
    """Metadata for a song."""
    id: int
    youtube_url: str
    audio_path: Optional[str] = None
    accepted: bool = False
    rejected: bool = False


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


def save_songs_meta(songs: List[SongMeta]):
    """Save song metadata to JSON file."""
    ensure_dirs()
    with open(SONGS_META_PATH, "w") as f:
        json.dump([dataclasses.asdict(s) for s in songs], f, indent=2)


def load_songs_meta() -> Optional[List[SongMeta]]:
    """Load song metadata from JSON file."""
    if not SONGS_META_PATH.exists():
        return None
    with open(SONGS_META_PATH, "r") as f:
        raw = json.load(f)
    return [SongMeta(**item) for item in raw]


def load_song_list_from_txt(txt_path: str) -> List[SongMeta]:
    """Load YouTube URLs from a text file."""
    with open(txt_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    return [SongMeta(id=i, youtube_url=url) for i, url in enumerate(urls)]


# =========================
# Audio Download
# =========================

def download_audio_for_song(song: SongMeta) -> SongMeta:
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


def build_or_load_feature_matrix(songs: List[SongMeta]) -> np.ndarray:
    """Build or load the feature matrix for all songs."""
    ensure_dirs()
    if FEATURES_PATH.exists():
        feats = np.load(FEATURES_PATH)
        if feats.shape[0] == len(songs):
            return feats

    all_feats = []
    for i, song in enumerate(songs):
        if not song.audio_path or not os.path.exists(song.audio_path):
            songs[i] = download_audio_for_song(song)
            song = songs[i]
        if not song.audio_path or not os.path.exists(song.audio_path):
            all_feats.append(np.zeros(51, dtype=np.float32))
            continue
        f = extract_features_from_audio(song.audio_path)
        all_feats.append(f)

    feats = np.stack(all_feats, axis=0)
    np.save(FEATURES_PATH, feats)
    save_songs_meta(songs)
    return feats


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


def save_feature_ae(params: FeatureAEParams, path: Path):
    """Save autoencoder parameters."""
    ensure_dirs()
    np.savez(
        path,
        W_enc=np.array(params.W_enc),
        b_enc=np.array(params.b_enc),
        W_dec=np.array(params.W_dec),
        b_dec=np.array(params.b_dec),
        mean=np.array(params.mean),
        std=np.array(params.std),
    )


def load_feature_ae(path: Path) -> Optional[FeatureAEParams]:
    """Load autoencoder parameters."""
    if not path.exists():
        return None
    data = np.load(path)
    return FeatureAEParams(
        W_enc=jnp.array(data["W_enc"]),
        b_enc=jnp.array(data["b_enc"]),
        W_dec=jnp.array(data["W_dec"]),
        b_dec=jnp.array(data["b_dec"]),
        mean=jnp.array(data["mean"]),
        std=jnp.array(data["std"]),
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


def save_discriminator(params: DiscriminatorParams, path: Path):
    """Save discriminator parameters."""
    ensure_dirs()
    np.savez(
        path,
        W1=np.array(params.W1),
        b1=np.array(params.b1),
        W2=np.array(params.W2),
        b2=np.array(params.b2),
    )


def load_discriminator(path: Path, input_dim: int, hidden_dim: int) -> Optional[DiscriminatorParams]:
    """Load discriminator parameters."""
    if not path.exists():
        return None
    data = np.load(path)
    return DiscriminatorParams(
        W1=jnp.array(data["W1"]),
        b1=jnp.array(data["b1"]),
        W2=jnp.array(data["W2"]),
        b2=jnp.array(data["b2"]),
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

    st.title("🎵 Playlist Sculptor")
    st.markdown(
        """
        Sculpt your perfect playlist from YouTube URLs using machine learning!

        This app uses:
        - **yt-dlp** to download audio from YouTube
        - **librosa** to extract audio features
        - **JAX** to train an 11D autoencoder + discriminator for song recommendations
        """
    )

    # Initialize session state from persistent storage
    if "songs" not in st.session_state:
        songs_meta = load_songs_meta()
        if songs_meta is not None:
            st.session_state["songs"] = songs_meta
        else:
            st.session_state["songs"] = []

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Load Songs", "Extract Features", "Train Models", "Sculpt Playlist", "Visualizations", "About"],
    )

    if page == "Load Songs":
        render_load_songs_page()
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


def render_load_songs_page():
    """Render the Load Songs page."""
    st.header("📥 Load Songs")

    songs: List[SongMeta] = st.session_state["songs"]

    # Option 1: Add individual URL
    st.subheader("Add Individual Song")
    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter a YouTube video URL to add to your collection",
    )

    if st.button("Add Song", type="primary"):
        if url:
            # Check for duplicates
            existing_urls = [s.youtube_url for s in songs]
            if url in existing_urls:
                st.warning("This URL is already in your collection!")
            else:
                new_id = len(songs)
                new_song = SongMeta(id=new_id, youtube_url=url)
                songs.append(new_song)
                save_songs_meta(songs)
                st.success(f"Added song {new_id}!")
                st.rerun()
        else:
            st.warning("Please enter a YouTube URL")

    st.divider()

    # Option 2: Load from text file
    st.subheader("Load from Text File")
    txt_path = st.text_input(
        "Path to text file with YouTube URLs",
        value="all_songs.txt",
        help="Text file with one YouTube URL per line",
    )

    if st.button("Load from File"):
        try:
            new_songs = load_song_list_from_txt(txt_path)
            st.session_state["songs"] = new_songs
            save_songs_meta(new_songs)
            st.success(f"Loaded {len(new_songs)} songs from {txt_path}")
            st.rerun()
        except FileNotFoundError:
            st.error(f"File not found: {txt_path}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.divider()

    # Display current songs
    st.subheader(f"📚 Song Collection ({len(songs)} songs)")

    if songs:
        # Three column view
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**✅ Accepted**")
            for s in songs:
                if s.accepted:
                    st.write(f"{s.id}: {s.youtube_url[:40]}...")

        with col2:
            st.markdown("**❌ Rejected**")
            for s in songs:
                if s.rejected:
                    st.write(f"{s.id}: {s.youtube_url[:40]}...")

        with col3:
            st.markdown("**⚪ Neutral**")
            for s in songs:
                if not s.accepted and not s.rejected:
                    st.write(f"{s.id}: {s.youtube_url[:40]}...")
    else:
        st.info("No songs loaded. Add URLs above or load from a text file.")


def render_extract_features_page():
    """Render the Extract Features page."""
    st.header("🎧 Extract Features")

    songs: List[SongMeta] = st.session_state.get("songs", [])

    if not songs:
        st.warning("No songs loaded. Go to 'Load Songs' page first.")
        return

    st.write(f"Total songs: {len(songs)}")

    # Show download/extract status
    downloaded = sum(1 for s in songs if s.audio_path and os.path.exists(s.audio_path))
    st.write(f"Downloaded: {downloaded}/{len(songs)}")

    if FEATURES_PATH.exists():
        features = np.load(FEATURES_PATH)
        st.success(f"Feature matrix exists: {features.shape}")
    else:
        st.info("No feature matrix yet.")

    if st.button("Download Audio & Extract Features", type="primary"):
        with st.spinner("Processing songs..."):
            progress_bar = st.progress(0)
            for i, song in enumerate(songs):
                progress_bar.progress((i + 1) / len(songs))
                if not song.audio_path or not os.path.exists(song.audio_path):
                    songs[i] = download_audio_for_song(song)

            feats = build_or_load_feature_matrix(songs)
            st.session_state["songs"] = songs
            st.success(f"Feature matrix shape: {feats.shape}")
            st.rerun()


def render_train_models_page():
    """Render the Train Models page."""
    st.header("🧠 Train Models")

    songs: List[SongMeta] = st.session_state.get("songs", [])

    if not songs:
        st.warning("No songs loaded. Go to 'Load Songs' page first.")
        return

    if not FEATURES_PATH.exists():
        st.warning("No features extracted. Go to 'Extract Features' page first.")
        return

    features = np.load(FEATURES_PATH)
    num_songs, feat_dim = features.shape
    st.write(f"Songs: {num_songs}, Feature dimension: {feat_dim}")

    # Feature Autoencoder Section
    st.subheader("Feature Autoencoder")

    ae_params = load_feature_ae(AE_MODEL_PATH)
    if ae_params is not None:
        st.success("✅ Feature AE model loaded")
    else:
        st.info("No feature AE model found. Train one below.")

    col1, col2 = st.columns(2)
    with col1:
        ae_epochs = st.number_input("AE Epochs", min_value=1, max_value=500, value=50)
    with col2:
        ae_lr = st.number_input("AE Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")

    if st.button("Train Feature AE"):
        with st.spinner("Training Feature AE..."):
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
            save_feature_ae(ae_params, AE_MODEL_PATH)
            st.success("Feature AE trained and saved!")
            st.rerun()

    st.divider()

    # Discriminator Section
    st.subheader("Discriminator (for recommendations)")

    ae_params = load_feature_ae(AE_MODEL_PATH)
    if ae_params is None:
        st.warning("Train the Feature AE first before training the discriminator.")
        return

    track_embs = compute_track_embeddings(ae_params, features)
    accepted_mask = np.array([s.accepted for s in songs], dtype=bool)
    rejected_mask = np.array([s.rejected for s in songs], dtype=bool)

    st.write(f"Accepted songs: {sum(accepted_mask)}")
    st.write(f"Rejected songs: {sum(rejected_mask)}")

    playlist_vec = compute_playlist_embedding(track_embs, accepted_mask)
    input_dim_disc = (LATENT_DIM + (LATENT_DIM * (LATENT_DIM + 1)) // 2) + LATENT_DIM

    disc_params = load_discriminator(DISC_MODEL_PATH, input_dim_disc, HIDDEN_DIM)
    if disc_params is not None:
        st.success("✅ Discriminator model loaded")
    else:
        st.info("No discriminator model found. Train one below.")

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
                save_discriminator(disc_params, DISC_MODEL_PATH)
                st.success("Discriminator trained and saved!")
                st.rerun()


def render_sculpt_playlist_page():
    """Render the Sculpt Playlist page with accept/reject workflow."""
    st.header("🎨 Sculpt Playlist")

    songs: List[SongMeta] = st.session_state.get("songs", [])

    if not songs:
        st.warning("No songs loaded. Go to 'Load Songs' page first.")
        return

    if not FEATURES_PATH.exists():
        st.warning("No features extracted. Go to 'Extract Features' page first.")
        return

    ae_params = load_feature_ae(AE_MODEL_PATH)
    if ae_params is None:
        st.warning("Train the Feature AE first on 'Train Models' page.")
        return

    features = np.load(FEATURES_PATH)
    track_embs = compute_track_embeddings(ae_params, features)

    accepted_mask = np.array([s.accepted for s in songs], dtype=bool)
    playlist_vec = compute_playlist_embedding(track_embs, accepted_mask)

    input_dim_disc = (LATENT_DIM + (LATENT_DIM * (LATENT_DIM + 1)) // 2) + LATENT_DIM
    disc_params = load_discriminator(DISC_MODEL_PATH, input_dim_disc, HIDDEN_DIM)

    # Song Suggestions
    st.subheader("🎯 Song Suggestions")

    if playlist_vec is not None and disc_params is not None:
        probs = predict_accept_probs(disc_params, playlist_vec, track_embs)
        neutral_idx = [i for i, s in enumerate(songs) if not s.accepted and not s.rejected]
        if neutral_idx:
            neutral_probs = [(i, probs[i]) for i in neutral_idx]
            neutral_probs.sort(key=lambda x: float(x[1]), reverse=True)
        else:
            neutral_probs = []
        st.info("Showing probability-ranked suggestions from discriminator")
    else:
        neutral_probs = [(i, 0.5) for i, s in enumerate(songs) if not s.accepted and not s.rejected]
        st.info("Discriminator not trained yet. Showing all neutral songs.")

    max_to_show = st.number_input("Max suggestions to show", min_value=1, max_value=50, value=10)

    for i, p in neutral_probs[:max_to_show]:
        s = songs[i]
        st.markdown(f"**Song {s.id}** — {s.youtube_url} (p≈{float(p):.2f})")

        # Audio preview
        if s.audio_path and os.path.exists(s.audio_path):
            try:
                with open(s.audio_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/m4a")
            except Exception:
                pass

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ Accept", key=f"accept_{i}"):
                s.accepted = True
                s.rejected = False
                save_songs_meta(songs)
                st.rerun()
        with c2:
            if st.button("❌ Reject", key=f"reject_{i}"):
                s.accepted = False
                s.rejected = True
                save_songs_meta(songs)
                st.rerun()
        with c3:
            st.write("")

    st.divider()

    # Reconsider rejected
    st.subheader("🔄 Reconsider Rejected Songs")

    rejected_songs = [s for s in songs if s.rejected]
    if rejected_songs:
        for s in rejected_songs:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"{s.id}: {s.youtube_url[:50]}...")
            with c2:
                if st.button("Make Neutral", key=f"neutral_{s.id}"):
                    s.rejected = False
                    save_songs_meta(songs)
                    st.rerun()
    else:
        st.info("No rejected songs to reconsider.")


def render_visualizations_page():
    """Render visualizations page with similarity matrix and latent space."""
    st.header("📊 Visualizations")

    songs: List[SongMeta] = st.session_state.get("songs", [])

    if not songs:
        st.warning("No songs loaded.")
        return

    if not FEATURES_PATH.exists():
        st.warning("No features extracted.")
        return

    ae_params = load_feature_ae(AE_MODEL_PATH)
    if ae_params is None:
        st.warning("Train the Feature AE first.")
        return

    features = np.load(FEATURES_PATH)
    track_embs = compute_track_embeddings(ae_params, features)
    n_songs = len(track_embs)

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
    ax.set_xticklabels([f"{i}" for i in range(n_songs)], fontsize=8)
    ax.set_yticklabels([f"{i}" for i in range(n_songs)], fontsize=8)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Song Similarity (Cosine Distance in Latent Space)")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Suggested Playlist Order
    st.subheader("🎶 Suggested Playlist Order")
    st.markdown("Songs ordered by similarity for smooth transitions:")

    accepted_mask = np.array([s.accepted for s in songs], dtype=bool)
    accepted_indices = np.where(accepted_mask)[0]

    if len(accepted_indices) > 1:
        accepted_sim_matrix = similarity_matrix[np.ix_(accepted_indices, accepted_indices)]
        order_in_accepted = generate_playlist_order(accepted_sim_matrix)
        playlist_order = [accepted_indices[i] for i in order_in_accepted]

        for rank, song_idx in enumerate(playlist_order):
            s = songs[song_idx]
            st.write(f"{rank + 1}. **Song {song_idx}** - {s.youtube_url[:60]}...")
    elif len(accepted_indices) == 1:
        st.write(f"1. **Song {accepted_indices[0]}** - {songs[accepted_indices[0]].youtube_url[:60]}...")
    else:
        st.info("Accept some songs to see the optimized playlist order.")

    st.divider()

    # Latent Space Visualization
    st.subheader("📈 Latent Space Visualization")

    if track_embs.shape[1] >= 2:
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        # Color by status
        colors = []
        for s in songs:
            if s.accepted:
                colors.append("green")
            elif s.rejected:
                colors.append("red")
            else:
                colors.append("gray")

        ax2.scatter(track_embs[:, 0], track_embs[:, 1], s=100, c=colors, alpha=0.7)
        for i in range(n_songs):
            ax2.annotate(f"{i}", (track_embs[i, 0], track_embs[i, 1]))

        ax2.set_xlabel("Latent Dimension 1")
        ax2.set_ylabel("Latent Dimension 2")
        ax2.set_title("Songs in Latent Space (First 2 Dimensions)")
        ax2.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Accepted'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Rejected'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Neutral'),
        ])
        st.pyplot(fig2)
        plt.close()


def render_about_page():
    """Render the About page."""
    st.header("ℹ️ About Playlist Sculptor")

    st.markdown(
        """
        ## Overview

        Playlist Sculptor is a Python application that helps you create cohesive playlists
        from YouTube videos using machine learning.

        ## How It Works

        1. **Load Songs**: Enter YouTube URLs individually or load from a text file
        2. **Extract Features**: Download audio and compute rich audio features using librosa:
           - Rhythm: tempo, beat regularity, onset strength
           - Loudness: RMS, loudness in dB
           - Spectral: centroid, bandwidth, rolloff, zero crossing rate
           - Tonal: 12-dimensional chroma features
           - Timbre: 13 MFCCs (mean and variance)
           - Harmonic/percussive ratio
        3. **Train Feature AE**: Learn 11D song embeddings
        4. **Sculpt Playlist**: Accept/reject songs to train the discriminator
        5. **Get Recommendations**: Probability-ranked suggestions based on your preferences
        6. **Visualize**: Similarity matrix and latent space plots

        ## Technology Stack

        - **Streamlit**: Web interface
        - **yt-dlp**: YouTube audio download
        - **librosa**: Audio feature extraction
        - **JAX**: Neural network training (pure SGD, no optax)

        ## Author

        Joshua Albert
        """
    )


if __name__ == "__main__":
    main()
