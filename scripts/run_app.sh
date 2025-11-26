#!/bin/bash
# Run the Playlist Sculptor Streamlit app

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Run the Streamlit app
streamlit run playlist_sculptor.py "$@"
