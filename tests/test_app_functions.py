"""Unit tests for ML processing functions in the Streamlit app module."""

# Import from the main playlist_sculptor.py using its defined functions
import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random


def load_ps_module():
    """Load playlist_sculptor module without running main.

    Note: We use exec() here because the playlist_sculptor.py file contains
    Streamlit-specific code that would fail on regular import. This approach
    allows us to load just the function definitions for testing.
    """
    ps_path = Path(__file__).parent.parent / "playlist_sculptor.py"
    spec = importlib.util.spec_from_file_location("ps", ps_path)
    ps = importlib.util.module_from_spec(spec)
    sys.modules["ps"] = ps
    with open(ps_path, "r") as f:
        code = f.read()
    exec(compile(code, ps_path, "exec"), ps.__dict__)  # noqa: S102
    return ps


ps = load_ps_module()

# Feature dimension used in the Streamlit app (different from src module's 36)
# This matches the extract_features_from_audio function output dimension
APP_FEATURE_DIM = 51
APP_LATENT_DIM = 11


class TestFeatureStats:
    """Tests for feature statistics computation."""

    def test_compute_feature_stats_shapes(self):
        """Test feature stats have correct shapes."""
        features = np.random.randn(10, APP_FEATURE_DIM).astype(np.float32)
        mean, std = ps.compute_feature_stats(features)

        assert mean.shape == (APP_FEATURE_DIM,)
        assert std.shape == (APP_FEATURE_DIM,)

    def test_compute_feature_stats_min_std_threshold(self):
        """Test that std values below threshold are set to 1.0."""
        # Create features with constant column (zero std)
        features = np.ones((10, 5), dtype=np.float32)
        features[:, 0] = np.random.randn(10)  # Only first col has variance

        mean, std = ps.compute_feature_stats(features)

        # Columns with near-zero std should be set to 1.0
        assert std[1] == 1.0
        assert std[2] == 1.0


class TestFeatureAutoencoder:
    """Tests for feature autoencoder functions."""

    @pytest.fixture
    def ae_params(self):
        """Create autoencoder params fixture."""
        rng = random.PRNGKey(42)
        mean = np.zeros(APP_FEATURE_DIM, dtype=np.float32)
        std = np.ones(APP_FEATURE_DIM, dtype=np.float32)
        return ps.init_feature_ae(rng, APP_FEATURE_DIM, APP_LATENT_DIM, mean, std)

    def test_init_feature_ae_shapes(self):
        """Test feature AE initialization produces correct shapes."""
        rng = random.PRNGKey(42)
        mean = np.zeros(APP_FEATURE_DIM, dtype=np.float32)
        std = np.ones(APP_FEATURE_DIM, dtype=np.float32)
        params = ps.init_feature_ae(rng, APP_FEATURE_DIM, APP_LATENT_DIM, mean, std)

        assert params.W_enc.shape == (APP_LATENT_DIM, APP_FEATURE_DIM)
        assert params.b_enc.shape == (APP_LATENT_DIM,)
        assert params.W_dec.shape == (APP_FEATURE_DIM, APP_LATENT_DIM)
        assert params.b_dec.shape == (APP_FEATURE_DIM,)

    def test_ae_encode_output_shape(self, ae_params):
        """Test encoding produces correct shape."""
        x = jnp.ones(APP_FEATURE_DIM)
        z = ps.ae_encode(ae_params, x)
        assert z.shape == (APP_LATENT_DIM,)

    def test_ae_decode_output_shape(self, ae_params):
        """Test decoding produces correct shape."""
        z = jnp.ones(APP_LATENT_DIM)
        x_hat = ps.ae_decode(ae_params, z)
        assert x_hat.shape == (APP_FEATURE_DIM,)

    def test_ae_roundtrip(self, ae_params):
        """Test encoder-decoder roundtrip produces same shape."""
        x = jnp.ones(APP_FEATURE_DIM)
        z = ps.ae_encode(ae_params, x)
        x_hat = ps.ae_decode(ae_params, z)
        assert x_hat.shape == x.shape

    def test_ae_batch_loss_is_finite(self, ae_params):
        """Test batch loss is finite."""
        batch_x = jnp.ones((5, APP_FEATURE_DIM))
        loss = ps.ae_batch_loss(ae_params, batch_x)
        assert jnp.isfinite(loss)


class TestDiscriminator:
    """Tests for discriminator functions."""

    @pytest.fixture
    def disc_params(self):
        """Create discriminator params fixture."""
        rng = random.PRNGKey(42)
        return ps.init_discriminator(rng, 100, 64)

    def test_init_discriminator_shapes(self):
        """Test discriminator initialization produces correct shapes."""
        rng = random.PRNGKey(42)
        params = ps.init_discriminator(rng, 100, 64)

        assert params.W1.shape == (64, 100)
        assert params.b1.shape == (64,)
        assert params.W2.shape == (1, 64)
        assert params.b2.shape == (1,)

    def test_disc_forward_output_shape(self, disc_params):
        """Test discriminator forward produces scalar output."""
        x = jnp.ones(100)
        output = ps.disc_forward(disc_params, x)
        assert output.shape == ()

    def test_disc_batch_loss_is_finite(self, disc_params):
        """Test discriminator batch loss is finite."""
        X = jnp.ones((5, 100))
        y = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
        loss = ps.disc_batch_loss(disc_params, X, y)
        assert jnp.isfinite(loss)


class TestSimilarity:
    """Tests for similarity computation."""

    def test_compute_similarity_identical(self):
        """Test identical vectors have similarity ~1."""
        z = np.array([1.0, 2.0, 3.0])
        sim = ps.compute_similarity(z, z)
        assert np.isclose(sim, 1.0)

    def test_compute_similarity_opposite(self):
        """Test opposite vectors have similarity ~-1."""
        z = np.array([1.0, 2.0, 3.0])
        sim = ps.compute_similarity(z, -z)
        assert np.isclose(sim, -1.0)

    def test_compute_similarity_orthogonal(self):
        """Test orthogonal vectors have similarity ~0."""
        z1 = np.array([1.0, 0.0, 0.0])
        z2 = np.array([0.0, 1.0, 0.0])
        sim = ps.compute_similarity(z1, z2)
        assert np.isclose(sim, 0.0)

    def test_compute_similarity_range(self):
        """Test similarity is in [-1, 1] range."""
        z1 = np.random.randn(APP_LATENT_DIM)
        z2 = np.random.randn(APP_LATENT_DIM)
        sim = ps.compute_similarity(z1, z2)
        assert sim >= -1.0
        assert sim <= 1.0


class TestPlaylistOrder:
    """Tests for playlist ordering function."""

    def test_generate_playlist_order_empty(self):
        """Test empty matrix returns empty list."""
        sim_matrix = np.zeros((0, 0))
        order = ps.generate_playlist_order(sim_matrix)
        assert order == []

    def test_generate_playlist_order_single(self):
        """Test single song matrix returns [0]."""
        sim_matrix = np.array([[1.0]])
        order = ps.generate_playlist_order(sim_matrix)
        assert order == [0]

    def test_generate_playlist_order_visits_all(self):
        """Test all songs are visited exactly once."""
        n = 5
        sim_matrix = np.random.rand(n, n)
        order = ps.generate_playlist_order(sim_matrix)

        assert len(order) == n
        assert set(order) == set(range(n))

    def test_generate_playlist_order_starts_at_zero(self):
        """Test order always starts at index 0."""
        sim_matrix = np.random.rand(5, 5)
        order = ps.generate_playlist_order(sim_matrix)
        assert order[0] == 0


class TestPlaylistEmbedding:
    """Tests for playlist embedding computation."""

    def test_compute_playlist_embedding_none_when_empty(self):
        """Test empty mask returns None."""
        track_embs = np.random.randn(10, APP_LATENT_DIM).astype(np.float32)
        mask = np.zeros(10, dtype=bool)
        result = ps.compute_playlist_embedding(track_embs, mask)
        assert result is None

    def test_compute_playlist_embedding_single_track(self):
        """Test single track embedding."""
        track_embs = np.random.randn(10, APP_LATENT_DIM).astype(np.float32)
        mask = np.zeros(10, dtype=bool)
        mask[0] = True

        result = ps.compute_playlist_embedding(track_embs, mask)

        assert result is not None
        # Expected size: latent_dim (mean) + upper triangular of cov matrix
        expected_size = APP_LATENT_DIM + (APP_LATENT_DIM * (APP_LATENT_DIM + 1)) // 2
        assert result.shape == (expected_size,)

    def test_compute_playlist_embedding_multiple_tracks(self):
        """Test multiple tracks embedding."""
        track_embs = np.random.randn(10, APP_LATENT_DIM).astype(np.float32)
        mask = np.array([True, True, True, False, False, False, False, False, False, False])

        result = ps.compute_playlist_embedding(track_embs, mask)

        assert result is not None
        expected_size = APP_LATENT_DIM + (APP_LATENT_DIM * (APP_LATENT_DIM + 1)) // 2
        assert result.shape == (expected_size,)
        assert result.dtype == np.float32


class TestTrackEmbeddings:
    """Tests for track embedding computation."""

    def test_compute_track_embeddings_shape(self):
        """Test track embeddings have correct shape."""
        rng = random.PRNGKey(42)
        mean = np.zeros(APP_FEATURE_DIM, dtype=np.float32)
        std = np.ones(APP_FEATURE_DIM, dtype=np.float32)
        ae_params = ps.init_feature_ae(rng, APP_FEATURE_DIM, APP_LATENT_DIM, mean, std)

        features = np.random.randn(10, APP_FEATURE_DIM).astype(np.float32)
        embeddings = ps.compute_track_embeddings(ae_params, features)

        assert embeddings.shape == (10, APP_LATENT_DIM)
        assert isinstance(embeddings, np.ndarray)


class TestFeatureExtraction:
    """Tests for audio feature extraction."""

    def test_extract_features_output_shape(self, tmp_path):
        """Test feature extraction returns correct shape (51 dimensions)."""
        import soundfile as sf

        # Create a synthetic audio file (5 seconds of noise)
        sr = 22050
        duration = 5
        audio = np.random.randn(sr * duration).astype(np.float32) * 0.1
        audio_path = tmp_path / "test_audio.wav"
        sf.write(audio_path, audio, sr)

        features = ps.extract_features_from_audio(str(audio_path))

        assert features.shape == (APP_FEATURE_DIM,)
        assert features.dtype == np.float32

    def test_extract_features_tempo_is_scalar(self, tmp_path):
        """Test tempo is converted to scalar (librosa >= 0.10.0 returns array)."""
        import soundfile as sf

        # Create a synthetic audio file
        sr = 22050
        duration = 5
        audio = np.random.randn(sr * duration).astype(np.float32) * 0.1
        audio_path = tmp_path / "test_audio.wav"
        sf.write(audio_path, audio, sr)

        features = ps.extract_features_from_audio(str(audio_path))

        # Tempo is at index 1 in the feature vector
        tempo_value = features[1]
        assert np.isscalar(tempo_value) or tempo_value.shape == ()
        assert np.isfinite(tempo_value)
