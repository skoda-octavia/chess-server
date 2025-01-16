from src.miniMax.models import Heu
import torch
import chess
from src.game.gameEnv import Game
import torch.multiprocessing as mp
from copy import deepcopy
from src.service import Service


class MiniMaxService(Service):

    def __init__(self, device, model_weights, layers, dropout, depth, input, bin_num):
        if depth < 1 or not isinstance(depth, int):
            raise ValueError("depth must be positive int. depth: " + depth)
        self.model = Heu(input, bin_num, layers, dropout)
        self.model.load_state_dict(torch.load(model_weights, weights_only=True))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
        self.depth = depth

    def __call__(self, fen):
        board = chess.Board(fen)
        game = Game.from_board(board)
        _, best_move = self.minimax(
            game, 1, board.turn, float("-inf"), float("inf"), self.model
        )
        current_value = game.board_value(self.model)
        return best_move.uci(), current_value

    def minimax(
        self,
        game: Game,
        depth: int,
        maximizing_player: bool,
        alpha: float,
        beta: float,
        model,
    ):
        if game.board.is_game_over() or depth == self.depth:
            return game.board_value(model), None

        best_move = None

        if maximizing_player:
            best_val = float("-inf")
            for child_move in game.board.legal_moves:
                new_game = game.simulate_move(child_move)
                val, _ = self.minimax(
                    new_game, depth + 1, new_game.board.turn, alpha, beta, model
                )

                if val > best_val:
                    best_val = val
                    if depth == 1:
                        best_move = child_move

                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break

            return best_val, best_move

        else:
            best_val = float("inf")
            for child_move in game.board.legal_moves:
                new_game = game.simulate_move(child_move)
                val, _ = self.minimax(
                    new_game, depth + 1, new_game.board.turn, alpha, beta, model
                )

                if val < best_val:
                    best_val = val
                    if depth == 1:
                        best_move = child_move

                beta = min(beta, best_val)
                if beta <= alpha:
                    break

            return best_val, best_move
