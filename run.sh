#!/bin/bash

# ==============================================================================
# Veriscan Environment Setup and Runner Script
# Usage: 
#   bash run.sh              # Setup (if needed) and run
#   bash run.sh --setup-only # Only setup environment and download data
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_CMD="python3"

echo "========================================"
echo "🛡️  Veriscan System Initialization"
echo "========================================"

# --- 1. Python Check ---
if ! command -v $PYTHON_CMD >/dev/null 2>&1; then
    echo "❌ Error: Python 3 is required but not installed."
    exit 1
fi

# --- 2. Virtual Environment Setup ---
if [ ! -d "$VENV_PATH" ]; then
    echo "📦 Creating virtual environment at $VENV_PATH..."
    $PYTHON_CMD -m venv "$VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# --- 3. Dependency Installation ---
echo "🔄 Installing/Verifying dependencies..."
pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

# --- 4. Directory Structure Check ---
echo "📁 Verifying required directories..."
mkdir -p "$PROJECT_ROOT/dataset/csv_data"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/scripts"

# --- 5. Data Initialization ---
echo "📊 Initializing system datasets..."
python "$PROJECT_ROOT/scripts/setup_data.py"

if [ "$1" == "--setup-only" ]; then
    echo "✅ Setup complete. Exiting."
    exit 0
fi

# ==============================================================================
# Run Services
# ==============================================================================

echo "🚀 Launching Veriscan Services..."

# Start FastAPI Backend in the background
echo "⚡ Starting FastAPI backend on port 8000..."
cd "$PROJECT_ROOT"
uvicorn api.main:app --host 0.0.0.0 --port 8000 > "$PROJECT_ROOT/api.log" 2>&1 &
API_PID=$!

# Wait lightly for API to initialize
sleep 5
echo "✅ Backend started (PID: $API_PID)"

# Start Streamlit Frontend
echo "🖥️ Starting Streamlit dashboard on port 8502..."
streamlit run streamlit_app.py --server.port 8502

# Cleanup function when script is interrupted
function cleanup() {
    echo ""
    echo "🛑 Shutting down backend..."
    kill $API_PID
    echo "Goodbye."
    exit
}

# Trap CTRL+C
trap cleanup SIGINT SIGTERM

wait $API_PID
