"""
Database module for Playlist Sculptor.

This module provides SQLite database functionality for storing:
- Songs with their metadata and features
- Playlists with their discriminator models
- Many-to-many relationships between songs and playlists
- Shared embedding model (feature autoencoder)

Uses Streamlit caching for efficient database access.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st

# Database path
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "playlist_sculptor.db"


@dataclass
class Song:
    """Represents a song in the database."""

    id: int
    youtube_url: str
    audio_path: Optional[str] = None
    features: Optional[np.ndarray] = None


@dataclass
class Playlist:
    """Represents a playlist in the database."""

    id: int
    name: str
    description: Optional[str] = None


@dataclass
class PlaylistSong:
    """Represents a song's membership and status in a playlist."""

    playlist_id: int
    song_id: int
    accepted: bool = False
    rejected: bool = False


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection (not cached, for modifications)."""
    ensure_data_dir()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Songs table - stores all songs globally
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            youtube_url TEXT UNIQUE NOT NULL,
            audio_path TEXT,
            features BLOB
        )
    """)

    # Playlists table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS playlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            discriminator_params BLOB
        )
    """)

    # Junction table for many-to-many relationship
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS playlist_songs (
            playlist_id INTEGER NOT NULL,
            song_id INTEGER NOT NULL,
            accepted INTEGER DEFAULT 0,
            rejected INTEGER DEFAULT 0,
            PRIMARY KEY (playlist_id, song_id),
            FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
            FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
        )
    """)

    # Shared embedding model table (only one row)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_model (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            params BLOB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ============================================================================
# Song Operations
# ============================================================================


def add_song(youtube_url: str, audio_path: Optional[str] = None) -> int:
    """Add a song to the database. Returns song id."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO songs (youtube_url, audio_path) VALUES (?, ?)",
            (youtube_url, audio_path),
        )
        conn.commit()
        song_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        # Song already exists, get its id
        cursor.execute("SELECT id FROM songs WHERE youtube_url = ?", (youtube_url,))
        row = cursor.fetchone()
        song_id = row["id"] if row else -1
    finally:
        conn.close()

    return song_id


def get_song(song_id: int) -> Optional[Song]:
    """Get a song by id."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM songs WHERE id = ?", (song_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    features = None
    if row["features"]:
        features = np.frombuffer(row["features"], dtype=np.float32)

    return Song(
        id=row["id"],
        youtube_url=row["youtube_url"],
        audio_path=row["audio_path"],
        features=features,
    )


def get_song_by_url(youtube_url: str) -> Optional[Song]:
    """Get a song by YouTube URL."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM songs WHERE youtube_url = ?", (youtube_url,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    features = None
    if row["features"]:
        features = np.frombuffer(row["features"], dtype=np.float32)

    return Song(
        id=row["id"],
        youtube_url=row["youtube_url"],
        audio_path=row["audio_path"],
        features=features,
    )


def get_all_songs() -> List[Song]:
    """Get all songs from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM songs ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    songs = []
    for row in rows:
        features = None
        if row["features"]:
            features = np.frombuffer(row["features"], dtype=np.float32)
        songs.append(
            Song(
                id=row["id"],
                youtube_url=row["youtube_url"],
                audio_path=row["audio_path"],
                features=features,
            )
        )

    return songs


def update_song_audio_path(song_id: int, audio_path: str):
    """Update a song's audio path."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE songs SET audio_path = ? WHERE id = ?", (audio_path, song_id))
    conn.commit()
    conn.close()


def update_song_features(song_id: int, features: np.ndarray):
    """Update a song's extracted features."""
    conn = get_db_connection()
    cursor = conn.cursor()

    features_blob = features.astype(np.float32).tobytes()
    cursor.execute("UPDATE songs SET features = ? WHERE id = ?", (features_blob, song_id))
    conn.commit()
    conn.close()


def delete_song(song_id: int):
    """Delete a song from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM songs WHERE id = ?", (song_id,))
    conn.commit()
    conn.close()


# ============================================================================
# Playlist Operations
# ============================================================================


def create_playlist(name: str, description: Optional[str] = None) -> int:
    """Create a new playlist. Returns playlist id."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO playlists (name, description) VALUES (?, ?)",
            (name, description),
        )
        conn.commit()
        playlist_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        # Playlist already exists, get its id
        cursor.execute("SELECT id FROM playlists WHERE name = ?", (name,))
        row = cursor.fetchone()
        playlist_id = row["id"] if row else -1
    finally:
        conn.close()

    return playlist_id


def get_playlist(playlist_id: int) -> Optional[Playlist]:
    """Get a playlist by id."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, description FROM playlists WHERE id = ?", (playlist_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return Playlist(
        id=row["id"],
        name=row["name"],
        description=row["description"],
    )


def get_all_playlists() -> List[Playlist]:
    """Get all playlists from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, description FROM playlists ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    return [
        Playlist(id=row["id"], name=row["name"], description=row["description"]) for row in rows
    ]


def update_playlist(playlist_id: int, name: Optional[str] = None, description: Optional[str] = None):
    """Update a playlist's name and/or description."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if name is not None:
        cursor.execute("UPDATE playlists SET name = ? WHERE id = ?", (name, playlist_id))
    if description is not None:
        cursor.execute(
            "UPDATE playlists SET description = ? WHERE id = ?", (description, playlist_id)
        )
    conn.commit()
    conn.close()


def delete_playlist(playlist_id: int):
    """Delete a playlist from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM playlists WHERE id = ?", (playlist_id,))
    conn.commit()
    conn.close()


# ============================================================================
# Playlist-Song Association Operations
# ============================================================================


def add_song_to_playlist(
    playlist_id: int, song_id: int, accepted: bool = False, rejected: bool = False
):
    """Add a song to a playlist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """INSERT INTO playlist_songs (playlist_id, song_id, accepted, rejected)
               VALUES (?, ?, ?, ?)""",
            (playlist_id, song_id, int(accepted), int(rejected)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Association already exists, update it
        cursor.execute(
            """UPDATE playlist_songs SET accepted = ?, rejected = ?
               WHERE playlist_id = ? AND song_id = ?""",
            (int(accepted), int(rejected), playlist_id, song_id),
        )
        conn.commit()
    finally:
        conn.close()


def remove_song_from_playlist(playlist_id: int, song_id: int):
    """Remove a song from a playlist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM playlist_songs WHERE playlist_id = ? AND song_id = ?",
        (playlist_id, song_id),
    )
    conn.commit()
    conn.close()


def update_song_status_in_playlist(
    playlist_id: int, song_id: int, accepted: bool = False, rejected: bool = False
):
    """Update a song's accepted/rejected status in a playlist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """UPDATE playlist_songs SET accepted = ?, rejected = ?
           WHERE playlist_id = ? AND song_id = ?""",
        (int(accepted), int(rejected), playlist_id, song_id),
    )
    conn.commit()
    conn.close()


def get_songs_in_playlist(playlist_id: int) -> List[Tuple[Song, PlaylistSong]]:
    """Get all songs in a playlist with their status."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT s.*, ps.accepted, ps.rejected
           FROM songs s
           JOIN playlist_songs ps ON s.id = ps.song_id
           WHERE ps.playlist_id = ?
           ORDER BY s.id""",
        (playlist_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        features = None
        if row["features"]:
            features = np.frombuffer(row["features"], dtype=np.float32)
        song = Song(
            id=row["id"],
            youtube_url=row["youtube_url"],
            audio_path=row["audio_path"],
            features=features,
        )
        ps = PlaylistSong(
            playlist_id=playlist_id,
            song_id=row["id"],
            accepted=bool(row["accepted"]),
            rejected=bool(row["rejected"]),
        )
        results.append((song, ps))

    return results


def get_playlist_song_status(playlist_id: int, song_id: int) -> Optional[PlaylistSong]:
    """Get a song's status in a playlist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM playlist_songs WHERE playlist_id = ? AND song_id = ?",
        (playlist_id, song_id),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return PlaylistSong(
        playlist_id=row["playlist_id"],
        song_id=row["song_id"],
        accepted=bool(row["accepted"]),
        rejected=bool(row["rejected"]),
    )


# ============================================================================
# Model Storage Operations
# ============================================================================


def save_embedding_model(params_dict: dict):
    """Save the shared embedding model parameters."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Convert params to JSON-serializable format with numpy arrays as base64
    serialized = {}
    for key, value in params_dict.items():
        if isinstance(value, np.ndarray):
            serialized[key] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tobytes().hex(),
            }
        else:
            # Handle JAX arrays by converting to numpy first
            arr = np.asarray(value)
            serialized[key] = {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "data": arr.tobytes().hex(),
            }

    params_blob = json.dumps(serialized).encode("utf-8")

    cursor.execute(
        """INSERT INTO embedding_model (id, params, updated_at)
           VALUES (1, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(id) DO UPDATE SET params = ?, updated_at = CURRENT_TIMESTAMP""",
        (params_blob, params_blob),
    )
    conn.commit()
    conn.close()


def load_embedding_model() -> Optional[dict]:
    """Load the shared embedding model parameters."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT params FROM embedding_model WHERE id = 1")
    row = cursor.fetchone()
    conn.close()

    if row is None or row["params"] is None:
        return None

    serialized = json.loads(row["params"].decode("utf-8"))

    params = {}
    for key, value in serialized.items():
        dtype = np.dtype(value["dtype"])
        shape = tuple(value["shape"])
        data = bytes.fromhex(value["data"])
        params[key] = np.frombuffer(data, dtype=dtype).reshape(shape)

    return params


def save_playlist_discriminator(playlist_id: int, params_dict: dict):
    """Save a playlist's discriminator model parameters."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Convert params to JSON-serializable format
    serialized = {}
    for key, value in params_dict.items():
        if isinstance(value, np.ndarray):
            serialized[key] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tobytes().hex(),
            }
        else:
            # Handle JAX arrays
            arr = np.asarray(value)
            serialized[key] = {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "data": arr.tobytes().hex(),
            }

    params_blob = json.dumps(serialized).encode("utf-8")

    cursor.execute(
        "UPDATE playlists SET discriminator_params = ? WHERE id = ?",
        (params_blob, playlist_id),
    )
    conn.commit()
    conn.close()


def load_playlist_discriminator(playlist_id: int) -> Optional[dict]:
    """Load a playlist's discriminator model parameters."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT discriminator_params FROM playlists WHERE id = ?", (playlist_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None or row["discriminator_params"] is None:
        return None

    serialized = json.loads(row["discriminator_params"].decode("utf-8"))

    params = {}
    for key, value in serialized.items():
        dtype = np.dtype(value["dtype"])
        shape = tuple(value["shape"])
        data = bytes.fromhex(value["data"])
        params[key] = np.frombuffer(data, dtype=dtype).reshape(shape)

    return params


# ============================================================================
# Streamlit Cached Functions
# ============================================================================


@st.cache_resource
def get_cached_db_connection():
    """Get a cached database connection for read operations."""
    init_database()
    return get_db_connection()


def clear_db_cache():
    """Clear the database cache to force a refresh."""
    get_cached_db_connection.clear()


@st.cache_data
def get_cached_songs() -> List[dict]:
    """Get all songs with Streamlit caching."""
    songs = get_all_songs()
    return [
        {
            "id": s.id,
            "youtube_url": s.youtube_url,
            "audio_path": s.audio_path,
            "has_features": s.features is not None,
        }
        for s in songs
    ]


@st.cache_data
def get_cached_playlists() -> List[dict]:
    """Get all playlists with Streamlit caching."""
    playlists = get_all_playlists()
    return [{"id": p.id, "name": p.name, "description": p.description} for p in playlists]


@st.cache_data
def get_cached_playlist_songs(playlist_id: int) -> List[dict]:
    """Get songs in a playlist with Streamlit caching."""
    results = get_songs_in_playlist(playlist_id)
    return [
        {
            "song_id": song.id,
            "youtube_url": song.youtube_url,
            "audio_path": song.audio_path,
            "has_features": song.features is not None,
            "accepted": ps.accepted,
            "rejected": ps.rejected,
        }
        for song, ps in results
    ]


def clear_songs_cache():
    """Clear songs cache."""
    get_cached_songs.clear()


def clear_playlists_cache():
    """Clear playlists cache."""
    get_cached_playlists.clear()


def clear_playlist_songs_cache():
    """Clear playlist songs cache."""
    get_cached_playlist_songs.clear()


def clear_all_caches():
    """Clear all Streamlit data caches."""
    clear_songs_cache()
    clear_playlists_cache()
    clear_playlist_songs_cache()


# ============================================================================
# Feature Matrix Operations
# ============================================================================


def get_all_songs_feature_matrix() -> Tuple[np.ndarray, List[int]]:
    """Get feature matrix for all songs that have features.

    Returns:
        Tuple of (features matrix, list of song ids)
    """
    songs = get_all_songs()
    features_list = []
    song_ids = []

    for song in songs:
        if song.features is not None:
            features_list.append(song.features)
            song_ids.append(song.id)

    if not features_list:
        return np.array([]), []

    return np.stack(features_list, axis=0), song_ids


def get_playlist_feature_matrix(playlist_id: int) -> Tuple[np.ndarray, List[int], List[bool], List[bool]]:
    """Get feature matrix for songs in a playlist.

    Returns:
        Tuple of (features matrix, song ids, accepted flags, rejected flags)
    """
    results = get_songs_in_playlist(playlist_id)
    features_list = []
    song_ids = []
    accepted_flags = []
    rejected_flags = []

    for song, ps in results:
        if song.features is not None:
            features_list.append(song.features)
            song_ids.append(song.id)
            accepted_flags.append(ps.accepted)
            rejected_flags.append(ps.rejected)

    if not features_list:
        return np.array([]), [], [], []

    return np.stack(features_list, axis=0), song_ids, accepted_flags, rejected_flags
