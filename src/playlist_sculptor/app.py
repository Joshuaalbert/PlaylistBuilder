import os
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st
import yt_dlp
import librosa

import jax
import jax.numpy as jnp
from jax import random


# =========================
# Paths & simple constants
# =========================

DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
FEATURES_PATH = os.path.join(DATA_DIR, "features.npy")       # [num_songs, D]
SONGS_META_PATH = os.path.join(DATA_DIR, "songs_meta.json")  # song list + flags
AE_MODEL_PATH = os.path.join(DATA_DIR, "feature_ae.npz")
DISC_MODEL_PATH = os.path.join(DATA_DIR, "discriminator.npz")

LATENT_DIM = 11           # track embedding dim
HIDDEN_DIM = 64           # discriminator hidden layer
BATCH_SIZE = 32


# =========================
# Dataclasses / types
# =========================

@dataclass
class SongMeta:
    id: int
    youtube_url: str
    audio_path: Optional[str] = None
    accepted: bool = False
    rejected: bool = False


@dataclass
class FeatureAEParams:
    W_enc: jax.Array  # [latent_dim, input_dim]
    b_enc: jax.Array  # [latent_dim]
    W_dec: jax.Array  # [input_dim, latent_dim]
    b_dec: jax.Array  # [input_dim]
    mean: jax.Array   # [input_dim]
    std: jax.Array    # [input_dim]


@dataclass
class DiscriminatorParams:
    W1: jax.Array  # [hidden_dim, input_dim]
    b1: jax.Array  # [hidden_dim]
    W2: jax.Array  # [1, hidden_dim]
    b2: jax.Array  # [1]


# =========================
# Utilities: I/O & state
# =========================

def ensure_dirs():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def load_song_list_from_txt(txt_path: str) -> List[SongMeta]:
    with open(txt_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    songs = [SongMeta(id=i, youtube_url=url) for i, url in enumerate(urls)]
    return songs


def save_songs_meta(songs: List[SongMeta]):
    ensure_dirs()
    with open(SONGS_META_PATH, "w") as f:
        json.dump([dataclasses.asdict(s) for s in songs], f, indent=2)


def load_songs_meta() -> Optional[List[SongMeta]]:
    if not os.path.exists(SONGS_META_PATH):
        return None
    with open(SONGS_META_PATH, "r") as f:
        raw = json.load(f)
    return [SongMeta(**item) for item in raw]


# =========================
# YouTube download
# =========================

def download_audio_for_song(song: SongMeta) -> SongMeta:
    """
    Download audio with yt_dlp if not already downloaded.
    Saves as <video_id>.m4a in AUDIO_DIR.
    """
    if song.audio_path and os.path.exists(song.audio_path):
        return song

    ensure_dirs()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(AUDIO_DIR, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(song.youtube_url, download=True)
        if info is None:
            st.error(f"Failed to download: {song.youtube_url}")
            return song
        vid_id = info.get("id")
        ext = info.get("ext", "m4a")
        audio_path = os.path.join(AUDIO_DIR, f"{vid_id}.{ext}")
        song.audio_path = audio_path
        return song


# =========================
# Audio feature extraction
# =========================

def extract_features_from_audio(audio_path: str, sr: int = 22050) -> np.ndarray:
    """
    Compute low-level audio features as described in the text.
    Returns a 1D numpy array of features.
    """
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
        beat_reg = 0.5  # fallback

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
    """
    Returns feature matrix [num_songs, D], computing missing features as needed.
    """
    ensure_dirs()
    if os.path.exists(FEATURES_PATH):
        feats = np.load(FEATURES_PATH)
        # If song count changed, recompute.
        if feats.shape[0] == len(songs):
            return feats

    all_feats = []
    for song in songs:
        if not song.audio_path or not os.path.exists(song.audio_path):
            song = download_audio_for_song(song)
        if not song.audio_path or not os.path.exists(song.audio_path):
            # Skip problematic song with NaNs
            all_feats.append(np.zeros(50, dtype=np.float32))
            continue
        f = extract_features_from_audio(song.audio_path)
        all_feats.append(f)

    feats = np.stack(all_feats, axis=0)
    np.save(FEATURES_PATH, feats)
    save_songs_meta(songs)  # update audio_path info
    return feats


# =========================
# Feature autoencoder (JAX)
# =========================

def compute_feature_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def init_feature_ae(rng_key: jax.Array, input_dim: int, latent_dim: int,
                    mean: np.ndarray, std: np.ndarray) -> FeatureAEParams:
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
    x_norm = (x - params.mean) / params.std
    h = jnp.tanh(params.W_enc @ x_norm + params.b_enc)
    return h  # latent_dim


def ae_decode(params: FeatureAEParams, z: jax.Array) -> jax.Array:
    x_norm_hat = params.W_dec @ z + params.b_dec
    x_hat = x_norm_hat * params.std + params.mean
    return x_hat


def ae_batch_loss(params: FeatureAEParams, batch_x: jax.Array) -> jax.Array:
    def single_loss(x):
        z = ae_encode(params, x)
        x_hat = ae_decode(params, z)
        return jnp.mean((x_hat - x) ** 2)

    losses = jax.vmap(single_loss)(batch_x)
    return jnp.mean(losses)


ae_grad = jax.jit(jax.grad(ae_batch_loss))


def train_feature_ae(params: FeatureAEParams, features: np.ndarray,
                     num_epochs: int, lr: float, batch_size: int = BATCH_SIZE) -> FeatureAEParams:
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


def save_feature_ae(params: FeatureAEParams, path: str):
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


def load_feature_ae(path: str) -> Optional[FeatureAEParams]:
    if not os.path.exists(path):
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
    X = jnp.array(features)
    Z = jax.vmap(lambda x: ae_encode(params, x))(X)
    return np.array(Z)


# =========================
# Playlist embedding
# =========================

def compute_playlist_embedding(track_embs: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    track_embs: [num_songs, latent_dim]
    mask: [num_songs] bool, True = accepted in playlist
    Returns playlist embedding [77] or None if no accepted songs.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None

    E = track_embs[idx]  # [M, latent_dim]
    mu = E.mean(axis=0)  # [latent_dim]
    centered = E - mu
    # Sample covariance
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
    k1, k2, k3, k4 = random.split(rng_key, 4)
    W1 = random.normal(k1, (hidden_dim, input_dim)) * 0.05
    b1 = jnp.zeros((hidden_dim,))
    W2 = random.normal(k2, (1, hidden_dim)) * 0.05
    b2 = jnp.zeros((1,))
    return DiscriminatorParams(W1=W1, b1=b1, W2=W2, b2=b2)


def disc_forward(params: DiscriminatorParams, x: jax.Array) -> jax.Array:
    h = jnp.tanh(params.W1 @ x + params.b1)
    logit = params.W2 @ h + params.b2
    return logit[0]


def disc_batch_loss(params: DiscriminatorParams, X: jax.Array, y: jax.Array) -> jax.Array:
    def single_loss(x, y_i):
        logit = disc_forward(params, x)
        # stable BCE
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
    """
    playlist_vec: [P] (77)
    track_embs: [num_examples, latent_dim]
    labels: [num_examples] in {0,1}
    """
    P = jnp.array(playlist_vec)

    def build_input_vec(t_emb):
        return jnp.concatenate([P, t_emb], axis=0)

    inputs = jax.vmap(build_input_vec)(jnp.array(track_embs))  # [N, P+11]
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


def save_discriminator(params: DiscriminatorParams, path: str):
    ensure_dirs()
    np.savez(
        path,
        W1=np.array(params.W1),
        b1=np.array(params.b1),
        W2=np.array(params.W2),
        b2=np.array(params.b2),
    )


def load_discriminator(path: str, input_dim: int, hidden_dim: int) -> Optional[DiscriminatorParams]:
    if not os.path.exists(path):
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
    P = jnp.array(playlist_vec)

    def forward_single(t_emb):
        x = jnp.concatenate([P, t_emb], axis=0)
        logit = disc_forward(params, x)
        prob = jax.nn.sigmoid(logit)
        return prob

    probs = jax.vmap(forward_single)(jnp.array(track_embs))
    return np.array(probs)


# =========================
# Streamlit UI glue
# =========================

def main():
    st.title("Groovy Playlist Sculptor")

    ensure_dirs()

    # ---------- Load / initialise songs ----------
    st.sidebar.header("1. Load song list")
    txt_path = st.sidebar.text_input("Path to YouTube list", "all_songs.txt")

    if "songs" not in st.session_state:
        songs_meta = load_songs_meta()
        if songs_meta is not None:
            st.session_state["songs"] = songs_meta

    if st.sidebar.button("Load from text file"):
        songs = load_song_list_from_txt(txt_path)
        st.session_state["songs"] = songs
        save_songs_meta(songs)

    if "songs" not in st.session_state:
        st.info("Load a song list to start.")
        return

    songs: List[SongMeta] = st.session_state["songs"]

    # ---------- Download + feature extraction ----------
    st.sidebar.header("2. Audio & features")
    if st.sidebar.button("Download missing audio & extract features"):
        feats = build_or_load_feature_matrix(songs)
        st.success(f"Feature matrix shape: {feats.shape}")

    if not os.path.exists(FEATURES_PATH):
        st.warning("No features yet. Click 'Download missing audio & extract features'.")
        return

    features = np.load(FEATURES_PATH)
    num_songs, feat_dim = features.shape
    st.sidebar.write(f"{num_songs} songs, feature dim = {feat_dim}")

    # ---------- Feature AE ----------
    st.sidebar.header("3. Train / load feature model")
    ae_params = load_feature_ae(AE_MODEL_PATH)
    if ae_params is None:
        st.sidebar.write("No feature model found; will initialise randomly on train.")

    ae_epochs = st.sidebar.number_input("AE epochs", min_value=1, max_value=500, value=50)
    ae_lr = st.sidebar.number_input("AE learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")

    if st.sidebar.button("Train feature AE"):
        mean, std = compute_feature_stats(features)
        rng = random.PRNGKey(0)
        if ae_params is None or ae_params.W_enc.shape[1] != feat_dim:
            ae_params = init_feature_ae(rng, feat_dim, LATENT_DIM, mean, std)
        else:
            # update mean/std in existing params
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
        st.success("Feature AE trained and saved.")

    ae_params = load_feature_ae(AE_MODEL_PATH)
    if ae_params is None:
        st.warning("Train the feature AE first.")
        return

    track_embs = compute_track_embeddings(ae_params, features)  # [num_songs, LATENT_DIM]

    # ---------- Playlist & labels ----------
    accepted_mask = np.array([s.accepted for s in songs], dtype=bool)
    rejected_mask = np.array([s.rejected for s in songs], dtype=bool)

    playlist_vec = compute_playlist_embedding(track_embs, accepted_mask)
    if playlist_vec is None:
        st.info("No accepted songs yet; discriminator will be untrained / random.")

    # ---------- Discriminator ----------
    st.sidebar.header("4. Discriminator training")

    input_dim_disc = (LATENT_DIM + (LATENT_DIM * (LATENT_DIM + 1)) // 2) + LATENT_DIM  # 77 + 11 = 88

    disc_params = load_discriminator(DISC_MODEL_PATH, input_dim_disc, HIDDEN_DIM)
    if disc_params is None:
        st.sidebar.write("No discriminator yet; will init when training.")

    disc_epochs = st.sidebar.number_input("Disc epochs", min_value=1, max_value=1000, value=100)
    disc_lr = st.sidebar.number_input("Disc learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")

    if st.sidebar.button("Train discriminator (using current accepted/rejected)"):
        pos_idx = np.where(accepted_mask)[0]
        neg_idx = np.where(rejected_mask)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0 or playlist_vec is None:
            st.error("Need at least one accepted and one rejected song.")
        else:
            X_embs = np.concatenate([track_embs[pos_idx], track_embs[neg_idx]], axis=0)
            y_labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))], axis=0)
            rng = random.PRNGKey(42)
            if disc_params is None:
                disc_params = init_discriminator(rng, input_dim_disc, HIDDEN_DIM)
            disc_params = train_discriminator(disc_params, playlist_vec, X_embs, y_labels,
                                              num_epochs=disc_epochs, lr=disc_lr)
            save_discriminator(disc_params, DISC_MODEL_PATH)
            st.success("Discriminator trained and saved.")

    disc_params = load_discriminator(DISC_MODEL_PATH, input_dim_disc, HIDDEN_DIM)

    # ---------- Main UI: songs ----------
    st.header("Playlist sculpting")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Accepted")
        for s in songs:
            if s.accepted:
                st.write(f"{s.id}: {s.youtube_url}")
    with col2:
        st.subheader("Rejected")
        for s in songs:
            if s.rejected:
                st.write(f"{s.id}: {s.youtube_url}")
    with col3:
        st.subheader("Neutral")
        for s in songs:
            if not s.accepted and not s.rejected:
                st.write(f"{s.id}: {s.youtube_url}")

    # Suggestions
    st.subheader("Suggestions")

    if playlist_vec is not None and disc_params is not None:
        probs = predict_accept_probs(disc_params, playlist_vec, track_embs)
        neutral_idx = [i for i, s in enumerate(songs) if not s.accepted and not s.rejected]
        if neutral_idx:
            neutral_probs = [(i, probs[i]) for i in neutral_idx]
            neutral_probs.sort(key=lambda x: float(x[1]), reverse=True)
        else:
            neutral_probs = []
    else:
        neutral_probs = [(i, 0.5) for i, s in enumerate(songs) if not s.accepted and not s.rejected]

    max_to_show = st.number_input("Max suggestions to show", min_value=1, max_value=50, value=10)
    for i, p in neutral_probs[:max_to_show]:
        s = songs[i]
        st.markdown(f"**Song {s.id}** — {s.youtube_url} (p≈{float(p):.2f})")
        if s.audio_path and os.path.exists(s.audio_path):
            with open(s.audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/m4a")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Accept", key=f"accept_{i}"):
                s.accepted = True
                s.rejected = False
                save_songs_meta(songs)
                st.experimental_rerun()
        with c2:
            if st.button("Reject", key=f"reject_{i}"):
                s.accepted = False
                s.rejected = True
                save_songs_meta(songs)
                st.experimental_rerun()
        with c3:
            st.write(" ")

    st.subheader("Reconsider rejected")
    for i, s in enumerate(songs):
        if s.rejected:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"{s.id}: {s.youtube_url}")
            with c2:
                if st.button("Make neutral", key=f"neutral_{i}"):
                    s.rejected = False
                    save_songs_meta(songs)
                    st.experimental_rerun()


if __name__ == "__main__":
    main()
