from src.miniMax.models import Heu
import torch
import chess


class MiniMaxService:

    def __init__(self, model_weights, layers, dropout, depth):
        if depth < 1 or not isinstance(depth, int):
            raise ValueError("depth must be positive int. depth: " + depth)
        self.model = Heu(6*8*8+1, 1, layers, dropout)
        self.model.load_state_dict(torch.load(model_weights, weights_only=True))
        self.model.eval()
        self.model = self.model.to("cuda")
        self.depth = depth

    def __call__(self, fen):
        board = chess.Board(fen)
        _, move = self.minimax(board, 0, board.turn, float('-inf'), float('inf'))
        return move.uci()
    
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
        
        matrix_board = matrix_board.view(6*8*8)
        matrix_board = torch.cat([matrix_board, torch.tensor([int(board.turn)])])
        return matrix_board

    def board_value(self, board: chess.Board):
        outcome = board.outcome()
        winner = outcome.winner if outcome is not None else None
        if winner is None:
            tensor_board = self.create_tensor(board)
            tensor_board = tensor_board.to("cuda")
            out = self.model(tensor_board.unsqueeze(0))
            return out[0].item()
        if winner == chess.WHITE:
            return 1
        else:
            return 0


    def minimax(self, board: chess.Board, depth: int, maximizing_player: bool, alpha: float, beta: float, move: str = None):

        if board.is_game_over() or depth == self.depth:
            return self.board_value(board), move
        
        if maximizing_player:
            best_val = float('-inf')
            best_move = None
            for child_move in board.legal_moves:
                move = child_move if move is None else move
                tmp_board = board.copy()
                tmp_board.push(child_move)
                val, _ = self.minimax(tmp_board, depth+1, tmp_board.turn, alpha, beta, move)

                if val > best_val:
                    best_val = val
                    best_move = child_move
                
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            
            return best_val, best_move
        
        else:
            best_val = float('inf')
            best_move = None
            for child_move in board.legal_moves:
                move = child_move if depth == 0 else move
                tmp_board = board.copy()
                tmp_board.push(child_move)
                val, _ = self.minimax(tmp_board, depth+1, tmp_board.turn, alpha, beta, move)

                if val < best_val:
                    best_val = val
                    best_move = child_move
                
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            
            return best_val, best_move
         

if __name__ == "__main__":
    serv = MiniMaxService("", [23, 23, 23], 0,4)

    move = serv("B7/p4pkp/8/5b2/P2b1ppP/2N1r1n1/1PP2KPR/R4Q2 b - - 0 1")
    print(move)