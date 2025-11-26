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
    """Load playlist_sculptor module without running main."""
    ps_path = Path(__file__).parent.parent / "playlist_sculptor.py"
    spec = importlib.util.spec_from_file_location("ps", ps_path)
    ps = importlib.util.module_from_spec(spec)
    sys.modules["ps"] = ps
    with open(ps_path, "r") as f:
        code = f.read()
    exec(compile(code, ps_path, "exec"), ps.__dict__)
    return ps


ps = load_ps_module()


class TestFeatureStats:
    """Tests for feature statistics computation."""

    def test_compute_feature_stats_shapes(self):
        """Test feature stats have correct shapes."""
        features = np.random.randn(10, 51).astype(np.float32)
        mean, std = ps.compute_feature_stats(features)

        assert mean.shape == (51,)
        assert std.shape == (51,)

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
        mean = np.zeros(51, dtype=np.float32)
        std = np.ones(51, dtype=np.float32)
        return ps.init_feature_ae(rng, 51, 11, mean, std)

    def test_init_feature_ae_shapes(self):
        """Test feature AE initialization produces correct shapes."""
        rng = random.PRNGKey(42)
        mean = np.zeros(51, dtype=np.float32)
        std = np.ones(51, dtype=np.float32)
        params = ps.init_feature_ae(rng, 51, 11, mean, std)

        assert params.W_enc.shape == (11, 51)
        assert params.b_enc.shape == (11,)
        assert params.W_dec.shape == (51, 11)
        assert params.b_dec.shape == (51,)

    def test_ae_encode_output_shape(self, ae_params):
        """Test encoding produces correct shape."""
        x = jnp.ones(51)
        z = ps.ae_encode(ae_params, x)
        assert z.shape == (11,)

    def test_ae_decode_output_shape(self, ae_params):
        """Test decoding produces correct shape."""
        z = jnp.ones(11)
        x_hat = ps.ae_decode(ae_params, z)
        assert x_hat.shape == (51,)

    def test_ae_roundtrip(self, ae_params):
        """Test encoder-decoder roundtrip produces same shape."""
        x = jnp.ones(51)
        z = ps.ae_encode(ae_params, x)
        x_hat = ps.ae_decode(ae_params, z)
        assert x_hat.shape == x.shape

    def test_ae_batch_loss_is_finite(self, ae_params):
        """Test batch loss is finite."""
        batch_x = jnp.ones((5, 51))
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
        z1 = np.random.randn(11)
        z2 = np.random.randn(11)
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
        track_embs = np.random.randn(10, 11).astype(np.float32)
        mask = np.zeros(10, dtype=bool)
        result = ps.compute_playlist_embedding(track_embs, mask)
        assert result is None

    def test_compute_playlist_embedding_single_track(self):
        """Test single track embedding."""
        track_embs = np.random.randn(10, 11).astype(np.float32)
        mask = np.zeros(10, dtype=bool)
        mask[0] = True

        result = ps.compute_playlist_embedding(track_embs, mask)

        assert result is not None
        # Expected size: 11 (mean) + 66 (upper triangular of 11x11 cov) = 77
        assert result.shape == (77,)

    def test_compute_playlist_embedding_multiple_tracks(self):
        """Test multiple tracks embedding."""
        track_embs = np.random.randn(10, 11).astype(np.float32)
        mask = np.array([True, True, True, False, False, False, False, False, False, False])

        result = ps.compute_playlist_embedding(track_embs, mask)

        assert result is not None
        assert result.shape == (77,)
        assert result.dtype == np.float32


class TestTrackEmbeddings:
    """Tests for track embedding computation."""

    def test_compute_track_embeddings_shape(self):
        """Test track embeddings have correct shape."""
        rng = random.PRNGKey(42)
        mean = np.zeros(51, dtype=np.float32)
        std = np.ones(51, dtype=np.float32)
        ae_params = ps.init_feature_ae(rng, 51, 11, mean, std)

        features = np.random.randn(10, 51).astype(np.float32)
        embeddings = ps.compute_track_embeddings(ae_params, features)

        assert embeddings.shape == (10, 11)
        assert isinstance(embeddings, np.ndarray)
