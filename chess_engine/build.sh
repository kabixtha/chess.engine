#!/usr/bin/env bash
set -o errexit

pip install -r chess_engine/requirements.txt
python chess_engine/manage.py collectstatic --no-input
python chess_engine/manage.py migrate