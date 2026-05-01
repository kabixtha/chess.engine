from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import chess
from board_to_fen.predict import get_fen_from_image


def _variants(img: Image.Image):
    img = img.convert("RGB")
    items = [img]

    w, h = img.size
    margin = int(min(w, h) * 0.04)
    if margin > 0:
        items.append(img.crop((margin, margin, w - margin, h - margin)))

    gray = ImageOps.grayscale(img).convert("RGB")
    items.append(gray)

    items.append(ImageEnhance.Contrast(img).enhance(1.6))
    items.append(ImageEnhance.Sharpness(img).enhance(1.4))

    resized = img.resize((800, 800))
    items.append(resized)
    items.append(ImageEnhance.Contrast(resized).enhance(1.8))

    return items


def _try_piece_fen(img: Image.Image, black_view: bool) -> str:
    piece_fen = get_fen_from_image(img, black_view=black_view)
    if not piece_fen or "/" not in piece_fen:
        raise ValueError("Could not detect a valid board position from the image.")
    return piece_fen


def detect_board(img_bytes: bytes, side: str = "white") -> str:
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}")

    preferred_black_view = (side == "black")
    errors = []

    for variant in _variants(img):
        for black_view in (preferred_black_view, not preferred_black_view):
            try:
                piece_fen = _try_piece_fen(variant, black_view=black_view)
                fen = f"{piece_fen} {'b' if side == 'black' else 'w'} - - 0 1"
                board = chess.Board(fen)

                if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                    raise ValueError("Both kings were not detected correctly.")

                return fen
            except Exception as exc:
                errors.append(str(exc))

    raise ValueError("Could not detect a valid board position from the image.")