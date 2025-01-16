import chess
import torch


class Game:

    board: chess.Board = None
    win_bias = 64
    max_evaluation = 63

    layer_to_piece = {
        1: chess.Piece(chess.PAWN, chess.WHITE),
        2: chess.Piece(chess.KNIGHT, chess.WHITE),
        3: chess.Piece(chess.BISHOP, chess.WHITE),
        4: chess.Piece(chess.ROOK, chess.WHITE),
        5: chess.Piece(chess.QUEEN, chess.WHITE),
        6: chess.Piece(chess.KING, chess.WHITE),
        -1: chess.Piece(chess.PAWN, chess.BLACK),
        -2: chess.Piece(chess.KNIGHT, chess.BLACK),
        -3: chess.Piece(chess.BISHOP, chess.BLACK),
        -4: chess.Piece(chess.ROOK, chess.BLACK),
        -5: chess.Piece(chess.QUEEN, chess.BLACK),
        -6: chess.Piece(chess.KING, chess.BLACK),
        0: None,
    }

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        game = Game()
        game.tensor = tensor
        game.board = Game.create_board(tensor)
        return game

    @staticmethod
    def from_board(board: chess.Board):
        tensor = Game.create_tensor(board)
        game = Game()
        game.board = board
        game.tensor = tensor

        return game

    @staticmethod
    def create_tensor(board: chess.Board):
        matrix_board = torch.zeros((6, 8, 8))
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(i, j))
                if piece is not None:
                    piece_type = piece.piece_type
                    piece_color = piece.color
                    index = piece_type - 1
                    row = 7 - j
                    if piece_color == chess.WHITE:
                        matrix_board[index, row, i] = 1
                    else:
                        matrix_board[index, row, i] = -1
        return matrix_board

    @staticmethod
    def create_board(tensor: torch.Tensor):
        nonzero_mask = (tensor != 0).float()
        multiplied_indices = (
            torch.arange(1, 7, device=tensor.device).unsqueeze(-1).unsqueeze(-1)
        )
        result_tensor = tensor * multiplied_indices * nonzero_mask
        tensor = torch.sum(result_tensor, dim=0)
        board = chess.Board.empty()
        for row_idx, row in enumerate(tensor):
            for col_idx, val in enumerate(row):
                piece = Game.layer_to_piece[int(val.item())]
                if piece is not None:
                    square = chess.square(col_idx, 7 - row_idx)
                    board.set_piece_at(square, piece)
        return board

    @staticmethod
    def mask_mates(heus, next_boards: list[chess.Board]):
        for idx, board in enumerate(next_boards):
            outcome = board.outcome()
            winner = outcome.winner if outcome is not None else None
            if winner is None:
                continue
            if winner == chess.WHITE:
                heus[idx] = Game.win_bias
            else:
                heus[idx] = 0
        return heus

    @staticmethod
    def mask_mates(heus, next_boards: list[chess.Board], bin_num: int):
        for idx, board in enumerate(next_boards):
            outcome = board.outcome()
            winner = outcome.winner if outcome is not None else None
            if winner is None:
                continue
            if winner == chess.WHITE:
                heus[idx] = bin_num + Game.win_bias
            else:
                heus[idx] = -Game.win_bias
        return heus

    def state(self):
        return self.tensor

    def valid_moves(self):
        return self.board.legal_moves

    def make_move(self, move):
        new_game = self.simulate_move(move)
        return new_game

    def over(self, timeout=100):
        self.outcome = self.board.outcome()
        if self.outcome is None and len(self.board.move_stack) >= timeout:
            self.outcome = chess.Outcome(chess.Termination.SEVENTYFIVE_MOVES, None)
            return True

        return False if self.outcome is None else True

    def score(self, bin_num):
        self.outcome = self.board.outcome()
        winner = self.outcome.winner if self.outcome is not None else None
        if winner is None:
            return 0.5
        if winner == chess.WHITE:
            return bin_num + self.win_bias
        else:
            return -self.win_bias

    def valid_moves(self):
        return self.board.legal_moves

    def make_move(self, move):
        new_game = self.simulate_move(move)
        return new_game

    def equal_boards(self):
        nonzero_mask = (self.tensor != 0).float()
        multiplied_indices = (
            torch.arange(1, 7, device=self.tensor.device).unsqueeze(-1).unsqueeze(-1)
        )
        result_tensor = self.tensor * multiplied_indices * nonzero_mask
        tensor = torch.sum(result_tensor, dim=0)
        for row_idx, row in enumerate(tensor):
            for col_idx, val in enumerate(row):
                piece_tensor = self.layer_to_piece[int(val.item())]
                square = chess.square(col_idx, 7 - row_idx)
                piece_board = self.board.piece_at(square)
                assert piece_tensor == piece_board

    def simulate_move(self, move: chess.Move):
        new_board = self.board.copy()
        new_board.push(move)
        new_game = Game()
        if self.board.is_en_passant(move) or self.board.is_castling(move):
            new_game.board = new_board
            new_game.tensor = Game.create_tensor(new_board)
            return new_game
        new_tensor = self.tensor.clone()
        idx_beg = self.board.piece_at(move.from_square).piece_type - 1
        idx_end = idx_beg if move.promotion is None else int(move.promotion) - 1
        rank_beg = 7 - chess.square_rank(move.from_square)
        file_beg = chess.square_file(move.from_square)
        rank_end = 7 - chess.square_rank(move.to_square)
        file_end = chess.square_file(move.to_square)
        value = 1 if self.board.turn else -1
        for i in range(len(new_tensor)):
            new_tensor[i][rank_end][file_end] = torch.tensor(0)
        new_tensor[idx_beg][rank_beg][file_beg] = torch.tensor(0)
        new_tensor[idx_end][rank_end][file_end] = torch.tensor(value)
        new_game.tensor = new_tensor
        new_game.board = new_board
        return new_game

    def board_value(self, model):
        outcome = self.board.outcome()
        winner = outcome.winner if outcome is not None else None
        if winner is None:
            tensor_board = self.tensor.view(6 * 8 * 8)
            turn = 1 if self.board.turn else -1
            tensor_board = torch.cat((tensor_board, torch.tensor([turn])))
            tensor_board = tensor_board.to("cuda")
            out = model(tensor_board.unsqueeze(0))
            return out[0].item()
        if winner == chess.WHITE:
            return Game.max_evaluation + 1
        else:
            return -1

    def copy(self):
        copy = self.__class__.__new__(self.__class__)
        copy.tensor = self.tensor.clone()
        copy.board = self.board.copy(stack=True)
        return copy
