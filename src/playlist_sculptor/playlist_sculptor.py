"""
Playlist Sculptor - Core module for sculpting playlists from YouTube URLs.

This module provides functionality to:
- Download audio from YouTube URLs using yt-dlp
- Extract audio features using librosa
- Train an 11D autoencoder with discriminator using JAX (no optax)
- Cluster and organize songs for playlist generation
"""

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import librosa
import numpy as np
import soundfile as sf


# Constants
LATENT_DIM = 11
FEATURE_DIM = 36  # 20 MFCCs + 12 chroma + 1 spectral centroid + 1 rolloff + 1 zcr + 1 tempo
SAMPLE_RATE = 22050
DURATION = 30  # Seconds of audio to analyze


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""

    mfcc: np.ndarray
    chroma: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    tempo: float
    combined: np.ndarray


def get_data_dir() -> Path:
    """Get the data directory path."""
    # Check for data dir relative to this file's location
    module_dir = Path(__file__).parent.parent.parent
    data_dir = module_dir / "data"
    if not data_dir.exists():
        # Fall back to current working directory
        data_dir = Path.cwd() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def url_to_filename(url: str) -> str:
    """Convert URL to a safe filename using hash."""
    return hashlib.sha256(url.encode()).hexdigest()[:32]


def download_audio(url: str, output_dir: Optional[Path] = None) -> Path:
    """
    Download audio from a YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file

    Returns:
        Path to the downloaded audio file
    """
    if output_dir is None:
        output_dir = get_data_dir() / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = url_to_filename(url)
    output_path = output_dir / f"{filename}.wav"

    # Check if already downloaded
    if output_path.exists():
        return output_path

    # Download using yt-dlp
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        str(output_dir / f"{filename}.%(ext)s"),
        url,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download audio: {e.stderr}") from e

    return output_path


def extract_features(audio_path: Path) -> AudioFeatures:
    """
    Extract audio features using librosa.

    Args:
        audio_path: Path to the audio file

    Returns:
        AudioFeatures object containing extracted features
    """
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=DURATION)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Extract spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    # Extract spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = np.mean(spectral_rolloff)

    # Extract zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Extract tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_value = float(tempo)

    # Combine all features into a single vector
    combined = np.concatenate(
        [
            mfcc_mean,  # 20 features
            chroma_mean,  # 12 features
            [spectral_centroid_mean],  # 1 feature
            [spectral_rolloff_mean],  # 1 feature
            [zcr_mean],  # 1 feature
            [tempo_value / 200.0],  # 1 feature (normalized tempo)
        ]
    )

    return AudioFeatures(
        mfcc=mfcc_mean,
        chroma=chroma_mean,
        spectral_centroid=np.array([spectral_centroid_mean]),
        spectral_rolloff=np.array([spectral_rolloff_mean]),
        zero_crossing_rate=np.array([zcr_mean]),
        tempo=tempo_value,
        combined=combined,
    )


# ============================================================================
# JAX-based Autoencoder + Discriminator (no optax)
# ============================================================================


def init_encoder_params(key: jax.Array, input_dim: int, latent_dim: int = LATENT_DIM) -> dict:
    """Initialize encoder parameters."""
    keys = jax.random.split(key, 3)

    # Simple 3-layer encoder
    params = {
        "w1": jax.random.normal(keys[0], (input_dim, 64)) * 0.1,
        "b1": jnp.zeros(64),
        "w2": jax.random.normal(keys[1], (64, 32)) * 0.1,
        "b2": jnp.zeros(32),
        "w3": jax.random.normal(keys[2], (32, latent_dim)) * 0.1,
        "b3": jnp.zeros(latent_dim),
    }
    return params


def init_decoder_params(key: jax.Array, latent_dim: int = LATENT_DIM, output_dim: int = FEATURE_DIM) -> dict:
    """Initialize decoder parameters."""
    keys = jax.random.split(key, 3)

    params = {
        "w1": jax.random.normal(keys[0], (latent_dim, 32)) * 0.1,
        "b1": jnp.zeros(32),
        "w2": jax.random.normal(keys[1], (32, 64)) * 0.1,
        "b2": jnp.zeros(64),
        "w3": jax.random.normal(keys[2], (64, output_dim)) * 0.1,
        "b3": jnp.zeros(output_dim),
    }
    return params


def init_discriminator_params(key: jax.Array, latent_dim: int = LATENT_DIM) -> dict:
    """Initialize discriminator parameters."""
    keys = jax.random.split(key, 3)

    params = {
        "w1": jax.random.normal(keys[0], (latent_dim, 32)) * 0.1,
        "b1": jnp.zeros(32),
        "w2": jax.random.normal(keys[1], (32, 16)) * 0.1,
        "b2": jnp.zeros(16),
        "w3": jax.random.normal(keys[2], (16, 1)) * 0.1,
        "b3": jnp.zeros(1),
    }
    return params


def encoder_forward(params: dict, x: jax.Array) -> jax.Array:
    """Forward pass through encoder."""
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    z = h @ params["w3"] + params["b3"]
    return z


def decoder_forward(params: dict, z: jax.Array) -> jax.Array:
    """Forward pass through decoder."""
    h = jnp.tanh(z @ params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    x_recon = h @ params["w3"] + params["b3"]
    return x_recon


def discriminator_forward(params: dict, z: jax.Array) -> jax.Array:
    """Forward pass through discriminator."""
    h = jnp.tanh(z @ params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    logits = h @ params["w3"] + params["b3"]
    return jax.nn.sigmoid(logits)


def autoencoder_loss(
    enc_params: dict,
    dec_params: dict,
    disc_params: dict,
    x: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    """Compute autoencoder loss with reconstruction and adversarial terms."""
    # Encode and decode
    z = encoder_forward(enc_params, x)
    x_recon = decoder_forward(dec_params, z)

    # Reconstruction loss
    recon_loss = jnp.mean((x - x_recon) ** 2)

    # Adversarial loss (fool discriminator)
    fake_score = discriminator_forward(disc_params, z)
    adv_loss = -jnp.mean(jnp.log(fake_score + 1e-8))

    total_loss = recon_loss + 0.1 * adv_loss

    return total_loss, {"recon_loss": recon_loss, "adv_loss": adv_loss}


def discriminator_loss(
    disc_params: dict,
    enc_params: dict,
    x: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Compute discriminator loss."""
    # Get latent codes from encoder
    z_fake = encoder_forward(enc_params, x)

    # Generate real samples from prior (standard normal)
    z_real = jax.random.normal(key, z_fake.shape)

    # Discriminator scores
    real_score = discriminator_forward(disc_params, z_real)
    fake_score = discriminator_forward(disc_params, z_fake)

    # Binary cross entropy loss
    real_loss = -jnp.mean(jnp.log(real_score + 1e-8))
    fake_loss = -jnp.mean(jnp.log(1 - fake_score + 1e-8))

    return real_loss + fake_loss


def sgd_update(params: dict, grads: dict, learning_rate: float = 0.01) -> dict:
    """Simple SGD update (no optax)."""
    return {k: params[k] - learning_rate * grads[k] for k in params}


def train_autoencoder(
    features: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """
    Train the autoencoder with discriminator using pure JAX SGD.

    Args:
        features: Array of audio features (n_samples, n_features)
        epochs: Number of training epochs
        learning_rate: Learning rate for SGD
        seed: Random seed

    Returns:
        Tuple of (encoder_params, decoder_params, discriminator_params)
    """
    key = jax.random.PRNGKey(seed)
    input_dim = features.shape[1]

    # Initialize parameters
    keys = jax.random.split(key, 4)
    enc_params = init_encoder_params(keys[0], input_dim)
    dec_params = init_decoder_params(keys[1], output_dim=input_dim)
    disc_params = init_discriminator_params(keys[2])
    key = keys[3]

    # Convert features to JAX array
    x = jnp.array(features)

    # Training loop
    for epoch in range(epochs):
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Update autoencoder
        (ae_loss, metrics), ae_grads = jax.value_and_grad(autoencoder_loss, argnums=(0, 1), has_aux=True)(
            enc_params, dec_params, disc_params, x, subkey1
        )
        enc_grads, dec_grads = ae_grads
        enc_params = sgd_update(enc_params, enc_grads, learning_rate)
        dec_params = sgd_update(dec_params, dec_grads, learning_rate)

        # Update discriminator
        disc_loss, disc_grads = jax.value_and_grad(discriminator_loss)(
            disc_params, enc_params, x, subkey2
        )
        disc_params = sgd_update(disc_params, disc_grads, learning_rate)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Recon: {metrics['recon_loss']:.4f}, "
                f"Adv: {metrics['adv_loss']:.4f}, "
                f"Disc: {disc_loss:.4f}"
            )

    return enc_params, dec_params, disc_params


def encode_features(enc_params: dict, features: np.ndarray) -> np.ndarray:
    """Encode features to latent space."""
    z = encoder_forward(enc_params, jnp.array(features))
    return np.array(z)


def compute_similarity(z1: np.ndarray, z2: np.ndarray) -> float:
    """Compute cosine similarity between two latent vectors."""
    z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)
    return float(np.dot(z1_norm, z2_norm))


def save_model(
    enc_params: dict,
    dec_params: dict,
    disc_params: dict,
    path: Optional[Path] = None,
) -> Path:
    """Save model parameters."""
    if path is None:
        path = get_data_dir() / "models" / "autoencoder.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten all params into a single dict for saving
    save_dict = {}
    for name, params in [("enc", enc_params), ("dec", dec_params), ("disc", disc_params)]:
        for k, v in params.items():
            save_dict[f"{name}_{k}"] = np.array(v)

    np.savez(path, **save_dict)
    return path


def load_model(path: Optional[Path] = None) -> tuple[dict, dict, dict]:
    """Load model parameters."""
    if path is None:
        path = get_data_dir() / "models" / "autoencoder.npz"

    data = np.load(path)

    enc_params = {}
    dec_params = {}
    disc_params = {}

    for key in data.files:
        prefix, param_name = key.split("_", 1)
        value = jnp.array(data[key])
        if prefix == "enc":
            enc_params[param_name] = value
        elif prefix == "dec":
            dec_params[param_name] = value
        elif prefix == "disc":
            disc_params[param_name] = value

    return enc_params, dec_params, disc_params


def main():
    """Main entry point for the playlist sculptor."""
    print("Playlist Sculptor - Core Module")
    print("================================")
    print()
    print("This module provides the core functionality for:")
    print("  - Downloading audio from YouTube URLs")
    print("  - Extracting audio features using librosa")
    print("  - Training an 11D autoencoder with discriminator")
    print("  - Computing song similarities for playlist generation")
    print()
    print("To use the Streamlit app, run:")
    print("  streamlit run playlist_sculptor.py")
    print()
    print("Or use the run script:")
    print("  ./scripts/run_app.sh")


if __name__ == "__main__":
    main()
