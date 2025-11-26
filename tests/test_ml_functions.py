"""Unit tests for ML processing functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.playlist_sculptor.playlist_sculptor import (
    FEATURE_DIM,
    LATENT_DIM,
    autoencoder_loss,
    compute_similarity,
    decoder_forward,
    discriminator_forward,
    discriminator_loss,
    encode_features,
    encoder_forward,
    init_decoder_params,
    init_discriminator_params,
    init_encoder_params,
    load_model,
    save_model,
    sgd_update,
    train_autoencoder,
)


class TestParameterInitialization:
    """Tests for parameter initialization functions."""

    def test_init_encoder_params_shape(self):
        """Test that encoder params have correct shapes."""
        key = jax.random.PRNGKey(42)
        params = init_encoder_params(key, input_dim=FEATURE_DIM)

        assert params["w1"].shape == (FEATURE_DIM, 64)
        assert params["b1"].shape == (64,)
        assert params["w2"].shape == (64, 32)
        assert params["b2"].shape == (32,)
        assert params["w3"].shape == (32, LATENT_DIM)
        assert params["b3"].shape == (LATENT_DIM,)

    def test_init_encoder_params_custom_latent_dim(self):
        """Test encoder params with custom latent dimension."""
        key = jax.random.PRNGKey(42)
        latent_dim = 5
        params = init_encoder_params(key, input_dim=20, latent_dim=latent_dim)

        assert params["w3"].shape == (32, latent_dim)
        assert params["b3"].shape == (latent_dim,)

    def test_init_decoder_params_shape(self):
        """Test that decoder params have correct shapes."""
        key = jax.random.PRNGKey(42)
        params = init_decoder_params(key, output_dim=FEATURE_DIM)

        assert params["w1"].shape == (LATENT_DIM, 32)
        assert params["b1"].shape == (32,)
        assert params["w2"].shape == (32, 64)
        assert params["b2"].shape == (64,)
        assert params["w3"].shape == (64, FEATURE_DIM)
        assert params["b3"].shape == (FEATURE_DIM,)

    def test_init_discriminator_params_shape(self):
        """Test that discriminator params have correct shapes."""
        key = jax.random.PRNGKey(42)
        params = init_discriminator_params(key)

        assert params["w1"].shape == (LATENT_DIM, 32)
        assert params["b1"].shape == (32,)
        assert params["w2"].shape == (32, 16)
        assert params["b2"].shape == (16,)
        assert params["w3"].shape == (16, 1)
        assert params["b3"].shape == (1,)


class TestForwardPasses:
    """Tests for forward pass functions."""

    @pytest.fixture
    def encoder_params(self):
        """Create encoder params fixture."""
        key = jax.random.PRNGKey(42)
        return init_encoder_params(key, input_dim=FEATURE_DIM)

    @pytest.fixture
    def decoder_params(self):
        """Create decoder params fixture."""
        key = jax.random.PRNGKey(42)
        return init_decoder_params(key, output_dim=FEATURE_DIM)

    @pytest.fixture
    def discriminator_params(self):
        """Create discriminator params fixture."""
        key = jax.random.PRNGKey(42)
        return init_discriminator_params(key)

    def test_encoder_forward_output_shape(self, encoder_params):
        """Test encoder produces correct output shape."""
        x = jnp.ones((FEATURE_DIM,))
        z = encoder_forward(encoder_params, x)
        assert z.shape == (LATENT_DIM,)

    def test_encoder_forward_batch(self, encoder_params):
        """Test encoder works with batched input."""
        x = jnp.ones((10, FEATURE_DIM))
        z = jax.vmap(lambda xi: encoder_forward(encoder_params, xi))(x)
        assert z.shape == (10, LATENT_DIM)

    def test_decoder_forward_output_shape(self, decoder_params):
        """Test decoder produces correct output shape."""
        z = jnp.ones((LATENT_DIM,))
        x_recon = decoder_forward(decoder_params, z)
        assert x_recon.shape == (FEATURE_DIM,)

    def test_autoencoder_reconstruction(self, encoder_params, decoder_params):
        """Test encoder-decoder roundtrip."""
        x = jnp.ones((FEATURE_DIM,))
        z = encoder_forward(encoder_params, x)
        x_recon = decoder_forward(decoder_params, z)
        assert x_recon.shape == x.shape

    def test_discriminator_forward_output_shape(self, discriminator_params):
        """Test discriminator produces correct output shape."""
        z = jnp.ones((LATENT_DIM,))
        output = discriminator_forward(discriminator_params, z)
        assert output.shape == (1,)

    def test_discriminator_forward_output_range(self, discriminator_params):
        """Test discriminator output is between 0 and 1 (sigmoid)."""
        z = jnp.ones((LATENT_DIM,))
        output = discriminator_forward(discriminator_params, z)
        assert float(output[0]) >= 0.0
        assert float(output[0]) <= 1.0


class TestLossFunctions:
    """Tests for loss functions."""

    @pytest.fixture
    def all_params(self):
        """Create all model params fixtures."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        enc_params = init_encoder_params(keys[0], input_dim=FEATURE_DIM)
        dec_params = init_decoder_params(keys[1], output_dim=FEATURE_DIM)
        disc_params = init_discriminator_params(keys[2])
        return enc_params, dec_params, disc_params

    def test_autoencoder_loss_returns_scalar(self, all_params):
        """Test autoencoder loss returns a scalar."""
        enc_params, dec_params, disc_params = all_params
        x = jnp.ones((5, FEATURE_DIM))
        key = jax.random.PRNGKey(0)

        loss, metrics = autoencoder_loss(enc_params, dec_params, disc_params, x, key)

        assert loss.shape == ()
        assert "recon_loss" in metrics
        assert "adv_loss" in metrics

    def test_autoencoder_loss_is_finite(self, all_params):
        """Test autoencoder loss is finite."""
        enc_params, dec_params, disc_params = all_params
        x = jnp.ones((5, FEATURE_DIM))
        key = jax.random.PRNGKey(0)

        loss, _ = autoencoder_loss(enc_params, dec_params, disc_params, x, key)

        assert jnp.isfinite(loss)

    def test_discriminator_loss_returns_scalar(self, all_params):
        """Test discriminator loss returns a scalar."""
        enc_params, _, disc_params = all_params
        x = jnp.ones((5, FEATURE_DIM))
        key = jax.random.PRNGKey(0)

        loss = discriminator_loss(disc_params, enc_params, x, key)

        assert loss.shape == ()

    def test_discriminator_loss_is_finite(self, all_params):
        """Test discriminator loss is finite."""
        enc_params, _, disc_params = all_params
        x = jnp.ones((5, FEATURE_DIM))
        key = jax.random.PRNGKey(0)

        loss = discriminator_loss(disc_params, enc_params, x, key)

        assert jnp.isfinite(loss)


class TestSGDUpdate:
    """Tests for SGD update function."""

    def test_sgd_update_modifies_params(self):
        """Test that SGD update modifies parameters."""
        params = {"w": jnp.ones((3, 3)), "b": jnp.zeros(3)}
        grads = {"w": jnp.ones((3, 3)), "b": jnp.ones(3)}

        new_params = sgd_update(params, grads, learning_rate=0.1)

        assert not jnp.allclose(params["w"], new_params["w"])
        assert not jnp.allclose(params["b"], new_params["b"])

    def test_sgd_update_correct_direction(self):
        """Test SGD updates in correct direction (subtracting gradient)."""
        params = {"w": jnp.ones((3,))}
        grads = {"w": jnp.ones((3,))}

        new_params = sgd_update(params, grads, learning_rate=0.1)

        expected = jnp.ones((3,)) - 0.1 * jnp.ones((3,))
        assert jnp.allclose(new_params["w"], expected)


class TestTraining:
    """Tests for training function."""

    def test_train_autoencoder_returns_params(self):
        """Test training returns all parameters."""
        features = np.random.randn(10, FEATURE_DIM).astype(np.float32)

        enc_params, dec_params, disc_params = train_autoencoder(
            features, epochs=2, learning_rate=0.01, seed=42
        )

        assert "w1" in enc_params
        assert "w1" in dec_params
        assert "w1" in disc_params

    def test_train_autoencoder_params_modified(self):
        """Test training modifies parameters from initial values."""
        features = np.random.randn(10, FEATURE_DIM).astype(np.float32)
        key = jax.random.PRNGKey(42)
        initial_enc = init_encoder_params(key, input_dim=FEATURE_DIM)

        enc_params, _, _ = train_autoencoder(
            features, epochs=5, learning_rate=0.01, seed=42
        )

        # Parameters should be different after training
        assert not jnp.allclose(initial_enc["w1"], enc_params["w1"])


class TestEncodingAndSimilarity:
    """Tests for encoding and similarity functions."""

    def test_encode_features_output_shape(self):
        """Test encoding features produces correct shape."""
        key = jax.random.PRNGKey(42)
        enc_params = init_encoder_params(key, input_dim=FEATURE_DIM)
        features = np.random.randn(10, FEATURE_DIM).astype(np.float32)

        encoded = encode_features(enc_params, features)

        assert encoded.shape == (10, LATENT_DIM)

    def test_compute_similarity_range(self):
        """Test cosine similarity is between -1 and 1."""
        z1 = np.random.randn(LATENT_DIM)
        z2 = np.random.randn(LATENT_DIM)

        sim = compute_similarity(z1, z2)

        assert sim >= -1.0
        assert sim <= 1.0

    def test_compute_similarity_identical_vectors(self):
        """Test identical vectors have similarity 1."""
        z = np.random.randn(LATENT_DIM)

        sim = compute_similarity(z, z)

        assert np.isclose(sim, 1.0)

    def test_compute_similarity_opposite_vectors(self):
        """Test opposite vectors have similarity -1."""
        z = np.random.randn(LATENT_DIM)

        sim = compute_similarity(z, -z)

        assert np.isclose(sim, -1.0)


class TestModelSaveLoad:
    """Tests for model save/load functions."""

    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading model parameters."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        enc_params = init_encoder_params(keys[0], input_dim=FEATURE_DIM)
        dec_params = init_decoder_params(keys[1], output_dim=FEATURE_DIM)
        disc_params = init_discriminator_params(keys[2])

        model_path = tmp_path / "model.npz"
        save_model(enc_params, dec_params, disc_params, model_path)

        loaded_enc, loaded_dec, loaded_disc = load_model(model_path)

        # Check that loaded params match original
        for key in enc_params:
            assert jnp.allclose(enc_params[key], loaded_enc[key])
        for key in dec_params:
            assert jnp.allclose(dec_params[key], loaded_dec[key])
        for key in disc_params:
            assert jnp.allclose(disc_params[key], loaded_disc[key])
