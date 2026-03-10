#!/bin/bash
set -e

# --- 1. Data Initialization ---
if [ "$SKIP_SETUP" != "true" ]; then
    echo "========================================"
    echo "📊 Initializing Veriscan Datasets..."
    echo "========================================"
    python scripts/setup_data.py
else
    echo "⏭️ Skipping Data Initialization (SKIP_SETUP=true)"
fi

# --- 2. Execute Command ---
echo "🚀 Starting Service..."
exec "$@"
