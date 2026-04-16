#!/bin/bash

set -e  # stop on error

echo "Starting pipeline setup..."

# Step 1: Check if .venv exists
if [ ! -d ".venv" ]; then
    echo ".venv not found. Creating virtual environment..."
    python3 -m venv .venv

    echo "Activating virtual environment..."
    source .venv/bin/activate

    echo "Installing uv..."
    pip install --upgrade pip
    pip install uv

    echo "Installing dependencies using uv..."
    uv sync

else
    echo ".venv already exists"

    echo "Activating virtual environment..."
    source .venv/bin/activate

    # Optional: ensure uv exists
    if ! command -v uv &> /dev/null; then
        echo "uv not found inside venv. Installing..."
        pip install uv
    fi
fi

# Step 2: Load environment variables
echo "🔹 Loading environment variables..."
if [ -f ".env" ]; then
    echo "Loading environment variables..."
    set -a
    source .env
    set +a
fi

# Step 3: Set DVC root
export DVC_ROOT=$(pwd)

# Step 4: Run pipeline
echo " Running DVC pipeline..."
uv run dvc repro

echo "Pipeline completed successfully!"