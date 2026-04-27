#!/bin/bash
set -e
pip install -r requirements.txt
wget -q "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-avx2.tar"
tar -xf stockfish-ubuntu-x86-64-avx2.tar
cp stockfish/stockfish-ubuntu-x86-64-avx2 /opt/render/project/src/stockfish_bin
chmod +x /opt/render/project/src/stockfish_bin
python manage.py collectstatic --noinput
