#!/usr/bin/env bash
set -euo pipefail

# Simple venv setup script for WSL / Linux
# Usage: bash scripts/setup_venv.sh
# Optional: set PYTHON (e.g. PYTHON=python3.10) and VENV_DIR

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "Using python: $PYTHON"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: $PYTHON not found on PATH"
  exit 2
fi

echo "Creating virtual environment in $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

if [ -f requirements.txt ]; then
  echo "Installing Python dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found in repo root. Skipping pip install."
fi

echo "\nDone. Activate the venv with:\n  source $VENV_DIR/bin/activate"
echo "Run the app: python run.py"
echo "Run tests: pytest -q"
