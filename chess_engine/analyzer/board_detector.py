from io import BytesIO
from PIL import Image
import chess
from board_to_fen.predict import get_fen_from_image


def detect_board(img_bytes: bytes, side: str = "white") -> str:
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}")

    try:
        piece_fen = get_fen_from_image(
            img,
            black_view=(side == "black")
        )
    except Exception as exc:
        raise ValueError(
            f"Board detection failed: {exc}. "
            "Use a screenshot that shows only the 8x8 board."
        )

    if not piece_fen or "/" not in piece_fen:
        raise ValueError("Could not detect a valid board position from the image.")

    fen = f"{piece_fen} {'b' if side == 'black' else 'w'} - - 0 1"

    try:
        board = chess.Board(fen)
    except Exception as exc:
        raise ValueError(f"Detected invalid FEN: {fen} ({exc})")

    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        raise ValueError("Both kings were not detected correctly.")

    return fen