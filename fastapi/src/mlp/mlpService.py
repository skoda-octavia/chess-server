from src.mlp.models import MLP
import torch
import chess

class MlpService:

    def __init__(self, piece_paths: dict, selection_path: str, piece_layers: list, selection_layers: list):
        self.piece_models = {}
        piece_model_in = 7*8*8
        piece_selection_in = 6*8*8
        try:
            for piece, path in piece_paths.items():
                model = MLP(
                    piece_model_in,
                    64,
                    piece_layers,
                    # [piece_model_in, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 100, 64],
                    0)
                model.load_state_dict(torch.load(path, weights_only=True))
                model.eval()
                self.piece_models[piece] = model
            piece = "selection"
            selection_model = MLP(
                piece_selection_in,
                64,
                selection_layers,
                0)
            selection_model.eval()
            selection_model.load_state_dict(torch.load(selection_path, weights_only=True))
            self.selection_model = selection_model
            
        except Exception as e:
            raise ValueError(f"Error while loading model for {piece}: {e}")
        
    def create_tensor(self, board: chess.Board, transform: bool = False):
        matrix_board = torch.zeros((6, 8, 8))
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(i, j))
                if piece is not None:
                    piece_type = piece.piece_type
                    piece_color = piece.color
                    index = piece_type - 1

                    row = 7-j if not transform else j

                    if piece_color == chess.WHITE:
                        matrix_board[index, row, i] = 1
                    else:
                        matrix_board[index, row, i] = -1
        if transform:
            matrix_board *= -1
            matrix_board = torch.where(torch.abs(matrix_board) < 1e-6, torch.zeros_like(matrix_board), matrix_board)
        
        return matrix_board

    
    def is_promotion_move(self, board, from_square, to_square):
        piece = board.piece_at(from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        row = chess.square_rank(to_square)
        if (piece.color == chess.WHITE and row == 7) or (piece.color == chess.BLACK and row == 0):
            return True
        return False
    

    def __call__(self, fen: str):
        base_board = chess.Board(fen)
        turned = False
        if not base_board.turn:
            board = base_board.mirror()
            turned = True
        else:
            board = base_board
        base_legal_moves = list(base_board.legal_moves)
        legal_moves = list(board.legal_moves)
        from_legal = [move.from_square for move in legal_moves]
        piece_squares = [
            square
            for square, piece
            in board.piece_map().items()
            if square in from_legal
            and piece.color == chess.WHITE
        ]
        dd = [chess.square_name(sq) for sq in piece_squares]
        piece_select_mask = [0] * 64
        for square in piece_squares:
            piece_select_mask[square] = 1
        board_matrix = self.create_tensor(board)
        selection_output = self.selection_model(board_matrix.unsqueeze(0), torch.tensor(piece_select_mask))
        selection_output = torch.squeeze(selection_output)
        selected = torch.argmax(selection_output)
        sq = chess.square_name(selected.item())
        
        piece_type = board.piece_at(selected).piece_type
        add_board = torch.zeros(64)
        add_board[selected] = 1
        piece_src = torch.cat((board_matrix, add_board.reshape(8, 8).unsqueeze(0)), dim=0).unsqueeze(0)
        model = self.piece_models[piece_type]
        
        legal_squares = [
            move.to_square
            for move
            in legal_moves
            if move.from_square == selected
        ]
        sq = [chess.square_name(s) for s in legal_squares]
        to_mask = [0] * 64
        for square in legal_squares:
            to_mask[square] = 1
        move_to = model(piece_src, torch.tensor(to_mask))
        square_to = torch.argmax(move_to)

        if turned:
            selected = chess.square_mirror(selected)
            square_to = chess.square_mirror(square_to)
        move = chess.Move(selected, square_to)
        if self.is_promotion_move(base_board, selected, square_to):
            move.promotion = chess.QUEEN
        
        for legal_move in base_legal_moves:
            if move.to_square == legal_move.to_square and move.from_square == legal_move.from_square:
                return move.uci()

        print(f"illegal move generated: {move.uci()}, returning random")
        return base_legal_moves[0].uci()