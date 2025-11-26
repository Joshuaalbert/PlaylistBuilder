"""Unit tests for the database module."""

import numpy as np
import pytest

from src.playlist_sculptor import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Create a temporary database for testing."""
    # Monkeypatch the database path
    test_db_path = tmp_path / "test_playlist_sculptor.db"
    test_data_dir = tmp_path
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    monkeypatch.setattr(db, "DATA_DIR", test_data_dir)

    # Initialize the database
    db.init_database()
    yield test_db_path

    # Cleanup is automatic with tmp_path


class TestSongOperations:
    """Tests for song CRUD operations."""

    def test_add_song(self, temp_db):
        """Test adding a song."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        assert song_id > 0

    def test_add_duplicate_song(self, temp_db):
        """Test adding a duplicate song returns existing id."""
        song_id1 = db.add_song("https://youtube.com/watch?v=test1")
        song_id2 = db.add_song("https://youtube.com/watch?v=test1")
        assert song_id1 == song_id2

    def test_get_song(self, temp_db):
        """Test getting a song by id."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        song = db.get_song(song_id)
        assert song is not None
        assert song.youtube_url == "https://youtube.com/watch?v=test1"
        assert song.id == song_id

    def test_get_nonexistent_song(self, temp_db):
        """Test getting a nonexistent song returns None."""
        song = db.get_song(999)
        assert song is None

    def test_get_song_by_url(self, temp_db):
        """Test getting a song by URL."""
        db.add_song("https://youtube.com/watch?v=test1")
        song = db.get_song_by_url("https://youtube.com/watch?v=test1")
        assert song is not None
        assert song.youtube_url == "https://youtube.com/watch?v=test1"

    def test_get_all_songs(self, temp_db):
        """Test getting all songs."""
        db.add_song("https://youtube.com/watch?v=test1")
        db.add_song("https://youtube.com/watch?v=test2")
        db.add_song("https://youtube.com/watch?v=test3")

        songs = db.get_all_songs()
        assert len(songs) == 3

    def test_update_song_audio_path(self, temp_db):
        """Test updating a song's audio path."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        db.update_song_audio_path(song_id, "/path/to/audio.wav")

        song = db.get_song(song_id)
        assert song.audio_path == "/path/to/audio.wav"

    def test_update_song_features(self, temp_db):
        """Test updating a song's features."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        features = np.random.randn(51).astype(np.float32)
        db.update_song_features(song_id, features)

        song = db.get_song(song_id)
        assert song.features is not None
        assert np.allclose(song.features, features)

    def test_delete_song(self, temp_db):
        """Test deleting a song."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        db.delete_song(song_id)

        song = db.get_song(song_id)
        assert song is None


class TestPlaylistOperations:
    """Tests for playlist CRUD operations."""

    def test_create_playlist(self, temp_db):
        """Test creating a playlist."""
        playlist_id = db.create_playlist("My Playlist", "A test playlist")
        assert playlist_id > 0

    def test_create_duplicate_playlist(self, temp_db):
        """Test creating a duplicate playlist returns existing id."""
        id1 = db.create_playlist("My Playlist")
        id2 = db.create_playlist("My Playlist")
        assert id1 == id2

    def test_get_playlist(self, temp_db):
        """Test getting a playlist by id."""
        playlist_id = db.create_playlist("My Playlist", "A test playlist")
        playlist = db.get_playlist(playlist_id)
        assert playlist is not None
        assert playlist.name == "My Playlist"
        assert playlist.description == "A test playlist"

    def test_get_all_playlists(self, temp_db):
        """Test getting all playlists."""
        db.create_playlist("Playlist 1")
        db.create_playlist("Playlist 2")
        db.create_playlist("Playlist 3")

        playlists = db.get_all_playlists()
        assert len(playlists) == 3

    def test_update_playlist(self, temp_db):
        """Test updating a playlist."""
        playlist_id = db.create_playlist("Old Name")
        db.update_playlist(playlist_id, name="New Name", description="New description")

        playlist = db.get_playlist(playlist_id)
        assert playlist.name == "New Name"
        assert playlist.description == "New description"

    def test_delete_playlist(self, temp_db):
        """Test deleting a playlist."""
        playlist_id = db.create_playlist("My Playlist")
        db.delete_playlist(playlist_id)

        playlist = db.get_playlist(playlist_id)
        assert playlist is None


class TestPlaylistSongOperations:
    """Tests for playlist-song relationship operations."""

    def test_add_song_to_playlist(self, temp_db):
        """Test adding a song to a playlist."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        playlist_id = db.create_playlist("My Playlist")

        db.add_song_to_playlist(playlist_id, song_id)

        songs = db.get_songs_in_playlist(playlist_id)
        assert len(songs) == 1
        assert songs[0][0].id == song_id

    def test_add_song_with_status(self, temp_db):
        """Test adding a song with accepted/rejected status."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        playlist_id = db.create_playlist("My Playlist")

        db.add_song_to_playlist(playlist_id, song_id, accepted=True)

        songs = db.get_songs_in_playlist(playlist_id)
        assert songs[0][1].accepted is True
        assert songs[0][1].rejected is False

    def test_remove_song_from_playlist(self, temp_db):
        """Test removing a song from a playlist."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        playlist_id = db.create_playlist("My Playlist")

        db.add_song_to_playlist(playlist_id, song_id)
        db.remove_song_from_playlist(playlist_id, song_id)

        songs = db.get_songs_in_playlist(playlist_id)
        assert len(songs) == 0

    def test_update_song_status_in_playlist(self, temp_db):
        """Test updating a song's status in a playlist."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        playlist_id = db.create_playlist("My Playlist")

        db.add_song_to_playlist(playlist_id, song_id)
        db.update_song_status_in_playlist(playlist_id, song_id, accepted=True)

        status = db.get_playlist_song_status(playlist_id, song_id)
        assert status.accepted is True
        assert status.rejected is False

    def test_song_in_multiple_playlists(self, temp_db):
        """Test that a song can be in multiple playlists."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        playlist_id1 = db.create_playlist("Playlist 1")
        playlist_id2 = db.create_playlist("Playlist 2")

        db.add_song_to_playlist(playlist_id1, song_id, accepted=True)
        db.add_song_to_playlist(playlist_id2, song_id, rejected=True)

        # Different status in different playlists
        status1 = db.get_playlist_song_status(playlist_id1, song_id)
        status2 = db.get_playlist_song_status(playlist_id2, song_id)

        assert status1.accepted is True
        assert status2.rejected is True

    def test_get_playlist_song_status_nonexistent(self, temp_db):
        """Test getting status for song not in playlist returns None."""
        song_id = db.add_song("https://youtube.com/watch?v=test1")
        playlist_id = db.create_playlist("My Playlist")

        status = db.get_playlist_song_status(playlist_id, song_id)
        assert status is None


class TestModelStorage:
    """Tests for model parameter storage."""

    def test_save_and_load_embedding_model(self, temp_db):
        """Test saving and loading embedding model."""
        params = {
            "W_enc": np.random.randn(11, 51).astype(np.float32),
            "b_enc": np.zeros(11, dtype=np.float32),
            "W_dec": np.random.randn(51, 11).astype(np.float32),
            "b_dec": np.zeros(51, dtype=np.float32),
            "mean": np.zeros(51, dtype=np.float32),
            "std": np.ones(51, dtype=np.float32),
        }

        db.save_embedding_model(params)
        loaded = db.load_embedding_model()

        assert loaded is not None
        for key in params:
            assert np.allclose(params[key], loaded[key])

    def test_load_nonexistent_embedding_model(self, temp_db):
        """Test loading nonexistent embedding model returns None."""
        loaded = db.load_embedding_model()
        assert loaded is None

    def test_save_and_load_discriminator(self, temp_db):
        """Test saving and loading discriminator for a playlist."""
        playlist_id = db.create_playlist("My Playlist")

        params = {
            "W1": np.random.randn(64, 88).astype(np.float32),
            "b1": np.zeros(64, dtype=np.float32),
            "W2": np.random.randn(1, 64).astype(np.float32),
            "b2": np.zeros(1, dtype=np.float32),
        }

        db.save_playlist_discriminator(playlist_id, params)
        loaded = db.load_playlist_discriminator(playlist_id)

        assert loaded is not None
        for key in params:
            assert np.allclose(params[key], loaded[key])

    def test_load_nonexistent_discriminator(self, temp_db):
        """Test loading nonexistent discriminator returns None."""
        playlist_id = db.create_playlist("My Playlist")
        loaded = db.load_playlist_discriminator(playlist_id)
        assert loaded is None

    def test_different_discriminators_per_playlist(self, temp_db):
        """Test that different playlists can have different discriminators."""
        playlist_id1 = db.create_playlist("Playlist 1")
        playlist_id2 = db.create_playlist("Playlist 2")

        params1 = {
            "W1": np.random.randn(64, 88).astype(np.float32),
            "b1": np.zeros(64, dtype=np.float32),
            "W2": np.random.randn(1, 64).astype(np.float32),
            "b2": np.zeros(1, dtype=np.float32),
        }
        params2 = {
            "W1": np.random.randn(64, 88).astype(np.float32) * 2,
            "b1": np.ones(64, dtype=np.float32),
            "W2": np.random.randn(1, 64).astype(np.float32) * 2,
            "b2": np.ones(1, dtype=np.float32),
        }

        db.save_playlist_discriminator(playlist_id1, params1)
        db.save_playlist_discriminator(playlist_id2, params2)

        loaded1 = db.load_playlist_discriminator(playlist_id1)
        loaded2 = db.load_playlist_discriminator(playlist_id2)

        assert not np.allclose(loaded1["W1"], loaded2["W1"])


class TestFeatureMatrix:
    """Tests for feature matrix operations."""

    def test_get_all_songs_feature_matrix(self, temp_db):
        """Test getting feature matrix for all songs."""
        # Add songs with features
        song_id1 = db.add_song("https://youtube.com/watch?v=test1")
        song_id2 = db.add_song("https://youtube.com/watch?v=test2")
        song_id3 = db.add_song("https://youtube.com/watch?v=test3")  # No features

        features1 = np.random.randn(51).astype(np.float32)
        features2 = np.random.randn(51).astype(np.float32)

        db.update_song_features(song_id1, features1)
        db.update_song_features(song_id2, features2)

        matrix, song_ids = db.get_all_songs_feature_matrix()

        assert matrix.shape == (2, 51)  # Only 2 songs have features
        assert len(song_ids) == 2
        assert song_id1 in song_ids
        assert song_id2 in song_ids
        assert song_id3 not in song_ids

    def test_get_playlist_feature_matrix(self, temp_db):
        """Test getting feature matrix for playlist songs."""
        playlist_id = db.create_playlist("My Playlist")

        song_id1 = db.add_song("https://youtube.com/watch?v=test1")
        song_id2 = db.add_song("https://youtube.com/watch?v=test2")

        features1 = np.random.randn(51).astype(np.float32)
        features2 = np.random.randn(51).astype(np.float32)

        db.update_song_features(song_id1, features1)
        db.update_song_features(song_id2, features2)

        db.add_song_to_playlist(playlist_id, song_id1, accepted=True)
        db.add_song_to_playlist(playlist_id, song_id2, rejected=True)

        matrix, song_ids, accepted, rejected = db.get_playlist_feature_matrix(playlist_id)

        assert matrix.shape == (2, 51)
        assert len(song_ids) == 2
        assert accepted == [True, False]
        assert rejected == [False, True]

    def test_empty_feature_matrix(self, temp_db):
        """Test getting empty feature matrix."""
        matrix, song_ids = db.get_all_songs_feature_matrix()
        assert len(matrix) == 0
        assert len(song_ids) == 0
