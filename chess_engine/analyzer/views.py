import os, shutil, chess, chess.engine
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .board_detector import detect_board

MAX_FILE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES  = {'image/jpeg', 'image/png', 'image/webp', 'image/jpg'}
THINK_TIME     = 0.5

def _find_stockfish():
    env = os.environ.get('STOCKFISH_PATH')
    if env and os.path.isfile(env):
        return env
    found = shutil.which('stockfish')
    if found:
        return found
    for path in ['/usr/bin/stockfish', '/usr/games/stockfish',
                 '/usr/local/bin/stockfish', '/bin/stockfish']:
        if os.path.isfile(path):
            return path
    return 'stockfish'

STOCKFISH_PATH = _find_stockfish()

def index(request):
    return render(request, 'index.html')

def _err(msg, status=400):
    return JsonResponse({'error': msg}, status=status)

@csrf_exempt
@require_POST
def analyze(request):
    img_file = request.FILES.get('image')
    if not img_file:
        return _err('No image uploaded.')
    if img_file.size > MAX_FILE_BYTES:
        return _err('File too large. Maximum size is 10 MB.')
    if img_file.content_type not in ALLOWED_TYPES:
        return _err('Unsupported file type. Please upload JPEG, PNG, or WebP.')

    side = request.POST.get('side', 'white').lower()
    if side not in ('white', 'black'):
        side = 'white'

    img_bytes = img_file.read()

    try:
        fen = detect_board(img_bytes, side)
    except ValueError as exc:
        return _err(str(exc), 422)
    except Exception as exc:
        return _err(f'Board detection failed: {exc}', 422)

    try:
        board = chess.Board(fen)
    except Exception:
        board = chess.Board()

    if not list(board.legal_moves):
        return _err(
            'No legal moves found. Try a cleaner screenshot showing the full board.', 422
        )

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        result = engine.play(board, chess.engine.Limit(time=THINK_TIME))
        info   = engine.analyse(board, chess.engine.Limit(time=THINK_TIME))
        engine.quit()
    except FileNotFoundError:
        return _err(
            f'Stockfish not found at "{STOCKFISH_PATH}". '
            'Make sure packages.txt contains "stockfish" and redeploy.', 500
        )
    except Exception as exc:
        return _err(f'Stockfish error: {exc}', 500)

    move = result.best_move
    if move is None:
        return _err('Stockfish could not find a move.', 500)

    best_san = board.san(move)
    uci      = move.uci()

    score = info.get('score')
    if score:
        pov = score.white() if side == 'white' else score.black()
        if pov.is_mate():
            eval_str = f'Mate in {abs(pov.mate())}'
        else:
            cp = pov.score()
            sym = '+' if cp >= 0 else ''
            eval_str = f'{sym}{cp/100:.2f}'
    else:
        eval_str = 'N/A'

    return JsonResponse({
        'best_move':  best_san,
        'move_uci':   uci,
        'evaluation': eval_str,
        'fen':        fen,
        'side':       side,
    })