"""
board_detector.py
─────────────────
Reads a chess-board screenshot and returns a FEN string.

Color-independence strategy
────────────────────────────
We never hard-code board colors.  Instead we:
  1. Use CLAHE equalization so any color scheme has good contrast.
  2. Find the board outline via Canny edges + contour approximation.
  3. After warping, sample EVERY square's center brightness and split
     them into two groups (bimodal histogram) to learn the board's own
     light/dark baseline dynamically.
  4. Compare each square's pixel statistics to that learned baseline
     to decide: empty / white-piece / black-piece.

Piece-type accuracy note
────────────────────────
The shape-based heuristic below gives ~55-65 % accuracy for piece type.
For production-quality recognition swap `classify_piece_type()` for a
fine-tuned CNN (MobileNetV3 or EfficientNet-Lite trained on chess-piece
images is the standard approach).
"""

import cv2
import numpy as np
import chess
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
BOARD_PX  = 512          # warp output size (must be divisible by 8)
SQ_PX     = BOARD_PX // 8   # 64 px per square
MARGIN    = SQ_PX // 7      # inner crop to avoid square-border artifacts
LAPLACIAN_EMPTY_THRESH  = 55    # squares with lower variance are empty
BRIGHTNESS_EMPTY_THRESH = 28    # or if mean stays within this of base

# ─────────────────────────────────────────────────────────────────────────
# 1. Board-finding helpers
# ─────────────────────────────────────────────────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners in TL, TR, BR, BL order."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL (smallest x+y)
    rect[2] = pts[np.argmax(s)]   # BR (largest  x+y)
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR (smallest y-x)
    rect[3] = pts[np.argmax(d)]   # BL (largest  y-x)
    return rect


def _perspective_warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [BOARD_PX - 1, 0],
         [BOARD_PX - 1, BOARD_PX - 1], [0, BOARD_PX - 1]],
        dtype="float32"
    )
    M = cv2.getPerspectiveTransform(_order_corners(corners), dst)
    return cv2.warpPerspective(img, M, (BOARD_PX, BOARD_PX))


def _find_board_corners(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Locate the chessboard rectangle.
    Returns a (4, 2) float32 array of corner pixels, or None.

    Robustness techniques:
      • CLAHE: equalises low-contrast or unusual-colour boards
      • Two-pass Canny: adapts to bright & dark boards
      • Morphological close: repairs broken grid-line edges
      • Top-N contour search: tolerates small clutter in frame
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive histogram equalisation (key for colour independence)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (5, 5), 0)

    # Two edge maps (tight + loose) — merged for robustness
    e1 = cv2.Canny(blur, 30, 100)
    e2 = cv2.Canny(blur, 60, 180)
    edges = cv2.bitwise_or(e1, e2)

    # Close gaps between grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    min_area = (min(h, w) * 0.25) ** 2
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts[:10]:
        if cv2.contourArea(cnt) < min_area:
            break
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            # Sanity: the quad should be roughly square
            x, y, bw, bh = cv2.boundingRect(approx)
            ratio = bw / max(bh, 1)
            if 0.65 < ratio < 1.55:
                return approx.reshape(4, 2).astype("float32")

    return None


# ─────────────────────────────────────────────────────────────────────────
# 2. Square sampling helpers
# ─────────────────────────────────────────────────────────────────────────

def _sq_inner(gray: np.ndarray, row: int, col: int) -> np.ndarray:
    """Return the inner (border-trimmed) region of one square, grayscale."""
    y1 = row * SQ_PX + MARGIN
    y2 = (row + 1) * SQ_PX - MARGIN
    x1 = col * SQ_PX + MARGIN
    x2 = (col + 1) * SQ_PX - MARGIN
    return gray[y1:y2, x1:x2]


def _calibrate_baselines(gray: np.ndarray) -> Tuple[float, float]:
    """
    Dynamically learn the mean brightness of empty light/dark squares.

    Method: collect all 64 square-center means, then split at the median
    (the board's own bimodal distribution).  Works for any color theme.

    Returns (dark_baseline, light_baseline).
    """
    means = []
    for r in range(8):
        for c in range(8):
            inner = _sq_inner(gray, r, c)
            means.append(float(np.mean(inner)))

    median = float(np.median(means))
    dark_vals  = [m for m in means if m <= median]
    light_vals = [m for m in means if m >  median]

    dark_base  = float(np.mean(dark_vals))  if dark_vals  else 80.0
    light_base = float(np.mean(light_vals)) if light_vals else 180.0
    return dark_base, light_base


def _is_light_square(row: int, col: int) -> bool:
    """Standard chess colouring: a1 (row=7, col=0) is dark."""
    return (row + col) % 2 == 0


# ─────────────────────────────────────────────────────────────────────────
# 3. Piece detection & classification
# ─────────────────────────────────────────────────────────────────────────

def _has_piece(inner: np.ndarray, base: float) -> bool:
    """
    True if the square contains a piece.

    Two independent signals, either one triggers:
      • Laplacian variance   — pieces add texture; empty squares are flat
      • Brightness deviation — piece pixels shift the mean away from base
    """
    lap_var       = cv2.Laplacian(inner, cv2.CV_64F).var()
    brightness_dev = abs(float(np.mean(inner)) - base)
    return lap_var > LAPLACIAN_EMPTY_THRESH or brightness_dev > BRIGHTNESS_EMPTY_THRESH


def _piece_color(inner: np.ndarray, dark_base: float, light_base: float,
                 row: int, col: int) -> str:
    """
    Determine 'w' (white) or 'b' (black) for a piece on this square.

    Logic:
      • On a LIGHT square: a white piece looks bright (~light_base);
        a black piece looks noticeably darker than light_base.
      • On a DARK square:  a white piece looks noticeably brighter than
        dark_base; a black piece looks dark (~dark_base).

    We use a midpoint threshold between the two baselines so the decision
    is relative to THIS board's color scheme, not any fixed pixel value.
    """
    mid   = (dark_base + light_base) / 2.0
    mean  = float(np.mean(inner))
    light_sq = _is_light_square(row, col)

    if light_sq:
        # White piece on light square → mean near light_base
        # Black piece on light square → mean pulled well below light_base
        threshold = (mid + light_base) / 2.0
        return 'w' if mean >= threshold else 'b'
    else:
        # White piece on dark square → mean pulled well above dark_base
        # Black piece on dark square → mean near dark_base
        threshold = (dark_base + mid) / 2.0
        return 'b' if mean <= threshold else 'w'


def _classify_piece_type(inner: np.ndarray, color: str) -> str:
    """
    Shape-based heuristic classifier.  Accuracy ≈ 55–65 %.

    Pipeline:
      1. Otsu binarise (invert for black pieces so piece blob = white)
      2. Find largest contour → piece silhouette
      3. Compute fill-ratio and aspect-ratio of bounding rect
      4. Map to piece type via decision tree

    Replace this function with a CNN for production accuracy.
    """
    h, w = inner.shape

    # Equalise locally so piece stands out regardless of square brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq    = clahe.apply(inner)

    if color == 'w':
        _, bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Small open to remove noise dots
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 'P'

    cnt    = max(cnts, key=cv2.contourArea)
    area   = cv2.contourArea(cnt)
    fill   = area / (h * w)                         # how much of square is piece

    _, _, bw_r, bh_r = cv2.boundingRect(cnt)
    aspect = bh_r / max(bw_r, 1)                    # height / width

    # ── Decision tree ────────────────────────────────────────────────────
    # Pawn    : smallest, roughly round (aspect ≈ 1)
    # Knight  : medium, wider head, irregular
    # Bishop  : medium-tall, pointed (high aspect)
    # Rook    : medium, wide flat top (low aspect)
    # Queen   : tall, wide crown
    # King    : tallest, cross on top

    if fill < 0.11:
        return 'P'
    if fill < 0.19:
        return 'B' if aspect > 1.75 else 'N'
    if fill < 0.28:
        return 'R' if aspect < 1.15 else 'B'
    if fill < 0.38:
        return 'Q'
    return 'K'


# ─────────────────────────────────────────────────────────────────────────
# 4. Public entry point
# ─────────────────────────────────────────────────────────────────────────

def detect_board(img_bytes: bytes, side: str = 'white') -> str:
    """
    Parameters
    ----------
    img_bytes : raw bytes of the uploaded image file
    side      : 'white' → white is at the bottom of the screenshot
                'black' → black is at the bottom (board is flipped)

    Returns
    -------
    FEN string of the detected position.

    Raises
    ------
    ValueError  – descriptive message explaining what went wrong
    """
    # ── Decode ───────────────────────────────────────────────────────────
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — unsupported format or corrupted file.")

    # ── Locate board ─────────────────────────────────────────────────────
    corners = _find_board_corners(img)
    if corners is None:
        raise ValueError(
            "Chessboard not found. Make sure the full board is visible "
            "and not cut off at the edges."
        )

    # ── Warp to fixed-size square ─────────────────────────────────────────
    board_bgr  = _perspective_warp(img, corners)
    board_gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)

    # ── Learn this board's color baseline ────────────────────────────────
    dark_base, light_base = _calibrate_baselines(board_gray)

    # ── Build position ───────────────────────────────────────────────────
    board = chess.Board(None)   # start with empty board

    for row in range(8):
        for col in range(8):
            inner = _sq_inner(board_gray, row, col)
            base  = light_base if _is_light_square(row, col) else dark_base

            if not _has_piece(inner, base):
                continue

            pcolor = _piece_color(inner, dark_base, light_base, row, col)
            ptype  = _classify_piece_type(inner, pcolor)

            # Map (row, col) → chess.Square
            # row=0 is rank 8 when white plays from bottom
            if side == 'black':
                # Board image is flipped: row 0 = rank 1, col 0 = file h
                chess_rank = row
                chess_file = 7 - col
            else:
                chess_rank = 7 - row
                chess_file = col

            sq    = chess.square(chess_file, chess_rank)
            piece = chess.Piece.from_symbol(
                ptype if pcolor == 'w' else ptype.lower()
            )
            board.set_piece_at(sq, piece)

    # ── Validate kings ────────────────────────────────────────────────────
    if board.king(chess.WHITE) is None:
        raise ValueError("White king not found — board detection may be off.")
    if board.king(chess.BLACK) is None:
        raise ValueError("Black king not found — board detection may be off.")

    board.turn = chess.WHITE if side == 'white' else chess.BLACK
    return board.fen()
