#!/usr/bin/env bash
set -e

echo "Installing system packages..."
apt-get update -qq
apt-get install -y stockfish

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Build complete!"