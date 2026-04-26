"""
views.py
────────
Two views:
  GET  /          → renders index.html
  POST /analyze/  → accepts image + side, returns JSON with best move

Validation is done in two levels as specified:
  Level 1 (Django)  : wrong file type → 400 | file too large → 400
  Level 2 (OpenCV)  : board not found → 422 | missing king → 422
                      illegal position (board.is_valid()) → 422
"""

import os
import chess
import chess.engine
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render

from .board_detector import detect_board

# ── Config ───────────────────────────────────────────────────────────────
MAX_FILE_BYTES  = 10 * 1024 * 1024   # 10 MB
ALLOWED_TYPES   = {'image/jpeg', 'image/jpg', 'image/png',
                   'image/webp', 'image/gif', 'image/bmp'}
STOCKFISH_PATH  = getattr(settings, 'STOCKFISH_PATH', '/usr/bin/stockfish')
THINK_TIME_SEC  = 1.0                # seconds Stockfish is given to think


# ── Helper ───────────────────────────────────────────────────────────────
def _json_error(msg: str, status: int) -> JsonResponse:
    return JsonResponse({'error': msg}, status=status)


# ── Views ────────────────────────────────────────────────────────────────

def index(request):
    return render(request, 'index.html')


@csrf_exempt
@require_http_methods(["POST"])
def analyze(request):
    # ── Level 1: Django validation ────────────────────────────────────────
    uploaded = request.FILES.get('image')
    if not uploaded:
        return _json_error('No image file provided.', 400)

    # Content-type check
    ct = (uploaded.content_type or '').lower().split(';')[0].strip()
    if ct not in ALLOWED_TYPES:
        return _json_error(
            f'Unsupported file type "{ct}". '
            'Please upload a JPEG, PNG, WebP, GIF, or BMP image.',
            400
        )

    # Size check
    if uploaded.size > MAX_FILE_BYTES:
        return _json_error(
            f'File too large ({uploaded.size // 1024} KB). '
            f'Maximum allowed is {MAX_FILE_BYTES // 1024 // 1024} MB.',
            400
        )

    # Side parameter
    side = request.POST.get('side', 'white').lower().strip()
    if side not in ('white', 'black'):
        return _json_error("'side' must be 'white' or 'black'.", 400)

    img_bytes = uploaded.read()

    # ── Level 2: OpenCV + chess-logic validation ───────────────────────────
    try:
        fen = detect_board(img_bytes, side)
    except ValueError as exc:
        return _json_error(str(exc), 422)
    except Exception as exc:
        return _json_error(f'Board analysis failed: {exc}', 422)

    # python-chess position validity check
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return _json_error(f'Generated FEN is malformed: {exc}', 422)

    if not board.is_valid():
        return _json_error(
            'Detected position is not a legal chess position. '
            'The board may be partially obscured or the wrong side was selected.',
            422
        )

    # ── Stockfish ─────────────────────────────────────────────────────────
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except FileNotFoundError:
        return _json_error(
            'Stockfish binary not found on the server. '
            'Check STOCKFISH_PATH in settings or nixpacks.toml.',
            500
        )
    except Exception as exc:
        return _json_error(f'Failed to start Stockfish: {exc}', 500)

    try:
        # Best move
        result = engine.play(board, chess.engine.Limit(time=THINK_TIME_SEC))
        if result.move is None:
            engine.quit()
            return _json_error('No legal moves available in this position.', 422)

        best_move_san = board.san(result.move)
        best_move_uci = result.move.uci()

        # Evaluation score
        info      = engine.analyse(board, chess.engine.Limit(time=THINK_TIME_SEC))
        score_obj = info.get('score')
        eval_str  = 'N/A'
        if score_obj:
            pov = score_obj.white()
            if pov.is_mate():
                mate_n = pov.mate()
                eval_str = f'Mate in {mate_n}' if mate_n and mate_n > 0 else f'Mated in {abs(mate_n or 0)}'
            else:
                cp = pov.score()
                if cp is not None:
                    eval_str = f'+{cp/100:.2f}' if cp >= 0 else f'{cp/100:.2f}'

    finally:
        engine.quit()

    return JsonResponse({
        'best_move':  best_move_san,    # e.g. "Nf3"
        'move_uci':   best_move_uci,    # e.g. "g1f3"
        'evaluation': eval_str,         # e.g. "+0.35"
        'fen':        fen,
        'side':       side,
    })
