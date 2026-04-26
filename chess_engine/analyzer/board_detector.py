"""
board_detector.py  —  revised
Color-independent chess board detection + FEN generation.

Detection strategy (3-pass):
  Pass 1: Contour-based (fast, works on clean screenshots)
  Pass 2: Hough lines grid (handles boards with coordinate labels/borders)
  Pass 3: Center-crop contour (handles padded/framed board images)
"""

import cv2
import numpy as np
import chess
from typing import Optional, Tuple, List

BOARD_PX  = 512
SQ_PX     = BOARD_PX // 8
MARGIN    = SQ_PX // 7
LAPLACIAN_EMPTY  = 55
BRIGHTNESS_EMPTY = 28

def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def _perspective_warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0,0],[BOARD_PX-1,0],[BOARD_PX-1,BOARD_PX-1],[0,BOARD_PX-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(_order_corners(corners), dst)
    return cv2.warpPerspective(img, M, (BOARD_PX, BOARD_PX))

def _is_valid_quad(approx: np.ndarray, min_area: float) -> bool:
    if len(approx) != 4:
        return False
    area = cv2.contourArea(approx)
    if area < min_area:
        return False
    x, y, w, h = cv2.boundingRect(approx)
    ratio = w / max(h, 1)
    return 0.60 < ratio < 1.65

def _contour_detect(gray: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (5, 5), 0)
    e1 = cv2.Canny(blur, 20, 80)
    e2 = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_or(e1, e2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if _is_valid_quad(approx, min_area):
            return approx.reshape(4, 2).astype("float32")
    return None

def _cluster(values: List[int], gap: int = 15) -> List[int]:
    if not values:
        return []
    values = sorted(values)
    clusters, group = [], [values[0]]
    for v in values[1:]:
        if v - group[-1] <= gap:
            group.append(v)
        else:
            clusters.append(int(np.mean(group)))
            group = [v]
    clusters.append(int(np.mean(group)))
    return clusters

def _hough_detect(gray: np.ndarray, img_h: int, img_w: int) -> Optional[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100)
    min_len = min(img_h, img_w) // 5
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                             minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return None
    h_pos, v_pos = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 12 or angle > 168:
            h_pos.append((y1 + y2) // 2)
        elif 78 < angle < 102:
            v_pos.append((x1 + x2) // 2)
    h_clust = _cluster(h_pos)
    v_clust = _cluster(v_pos)
    if len(h_clust) < 2 or len(v_clust) < 2:
        return None
    top, bot = min(h_clust), max(h_clust)
    lft, rgt = min(v_clust), max(v_clust)
    w, h = rgt - lft, bot - top
    if w < min(img_h, img_w) * 0.2 or h < min(img_h, img_w) * 0.2:
        return None
    if not (0.60 < w / max(h, 1) < 1.65):
        return None
    return np.array([[lft,top],[rgt,top],[rgt,bot],[lft,bot]], dtype="float32")

def _center_crop_detect(img: np.ndarray) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    pad_y, pad_x = int(h * 0.10), int(w * 0.10)
    cropped = img[pad_y:h-pad_y, pad_x:w-pad_x]
    gray_c  = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    min_area = (min(cropped.shape[:2]) * 0.30) ** 2
    corners  = _contour_detect(gray_c, min_area)
    if corners is None:
        return None
    corners[:, 0] += pad_x
    corners[:, 1] += pad_y
    return corners

def _find_board_corners(img: np.ndarray) -> Optional[np.ndarray]:
    h, w   = img.shape[:2]
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_area = (min(h, w) * 0.25) ** 2
    corners = _contour_detect(gray, min_area)
    if corners is not None:
        return corners
    corners = _hough_detect(gray, h, w)
    if corners is not None:
        return corners
    return _center_crop_detect(img)

def _sq_inner(gray: np.ndarray, row: int, col: int) -> np.ndarray:
    y1 = row * SQ_PX + MARGIN
    y2 = (row + 1) * SQ_PX - MARGIN
    x1 = col * SQ_PX + MARGIN
    x2 = (col + 1) * SQ_PX - MARGIN
    return gray[y1:y2, x1:x2]

def _calibrate_baselines(gray: np.ndarray) -> Tuple[float, float]:
    means = [float(np.mean(_sq_inner(gray, r, c)))
             for r in range(8) for c in range(8)]
    median     = float(np.median(means))
    dark_vals  = [m for m in means if m <= median]
    light_vals = [m for m in means if m >  median]
    return (float(np.mean(dark_vals))  if dark_vals  else 80.0,
            float(np.mean(light_vals)) if light_vals else 180.0)

def _is_light_square(row: int, col: int) -> bool:
    return (row + col) % 2 == 0

def _has_piece(inner: np.ndarray, base: float) -> bool:
    lap_var = cv2.Laplacian(inner, cv2.CV_64F).var()
    dev     = abs(float(np.mean(inner)) - base)
    return lap_var > LAPLACIAN_EMPTY or dev > BRIGHTNESS_EMPTY

def _piece_color(inner: np.ndarray, dark_base: float, light_base: float,
                 row: int, col: int) -> str:
    mid      = (dark_base + light_base) / 2.0
    mean     = float(np.mean(inner))
    light_sq = _is_light_square(row, col)
    if light_sq:
        return 'w' if mean >= (mid + light_base) / 2.0 else 'b'
    else:
        return 'b' if mean <= (dark_base + mid) / 2.0 else 'w'

def _classify_piece_type(inner: np.ndarray, color: str) -> str:
    h, w = inner.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq    = clahe.apply(inner)
    flag = cv2.THRESH_BINARY if color == 'w' else cv2.THRESH_BINARY_INV
    _, bw = cv2.threshold(eq, 0, 255, flag + cv2.THRESH_OTSU)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 'P'
    cnt    = max(cnts, key=cv2.contourArea)
    fill   = cv2.contourArea(cnt) / (h * w)
    _, _, bw_r, bh_r = cv2.boundingRect(cnt)
    aspect = bh_r / max(bw_r, 1)
    if fill < 0.11:   return 'P'
    if fill < 0.19:   return 'B' if aspect > 1.75 else 'N'
    if fill < 0.28:   return 'R' if aspect < 1.15 else 'B'
    if fill < 0.38:   return 'Q'
    return 'K'

def detect_board(img_bytes: bytes, side: str = 'white') -> str:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — unsupported format or corrupted file.")
    corners = _find_board_corners(img)
    if corners is None:
        raise ValueError(
            "Chessboard not found. Make sure the full board is visible "
            "and not cut off at the edges."
        )
    board_bgr  = _perspective_warp(img, corners)
    board_gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    dark_base, light_base = _calibrate_baselines(board_gray)
    board = chess.Board(None)
    for row in range(8):
        for col in range(8):
            inner = _sq_inner(board_gray, row, col)
            base  = light_base if _is_light_square(row, col) else dark_base
            if not _has_piece(inner, base):
                continue
            pcolor = _piece_color(inner, dark_base, light_base, row, col)
            ptype  = _classify_piece_type(inner, pcolor)
            if side == 'black':
                chess_rank = row
                chess_file = 7 - col
            else:
                chess_rank = 7 - row
                chess_file = col
            sq    = chess.square(chess_file, chess_rank)
            piece = chess.Piece.from_symbol(ptype if pcolor == 'w' else ptype.lower())
            board.set_piece_at(sq, piece)
    if board.king(chess.WHITE) is None:
        raise ValueError("White king not found — try a cleaner screenshot.")
    if board.king(chess.BLACK) is None:
        raise ValueError("Black king not found — try a cleaner screenshot.")
    board.turn = chess.WHITE if side == 'white' else chess.BLACK
    return board.fen()