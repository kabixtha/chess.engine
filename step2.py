import chess

#harek piece ko value 
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 100000,
}
def evaluate(board):
    # If the game is over by checkmate
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999  # Black won
        else:
            return 99999   # White won

    score = 0
    for piece_type, value in PIECE_VALUES.items():
        # Add points for White's pieces
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        # Subtract points for Black's pieces
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    return score

board = chess.Board()
print("Starting position score:", evaluate(board))