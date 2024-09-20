import torch
import torch.nn as nn
from src.alpha.models import Heu, set_batchnorm_eval
import torch.multiprocessing as mp
import chess
from src.alpha.gameEnv import Game
import gc



class AlphaService:

    def __init__(self, model_weights, layers, playouts, game_timeout, dropout, proc_num=32):
        self.model = Heu(6*8*8, 1, layers, dropout)
        mp.set_start_method('forkserver', force=True)
        self.model.load_state_dict(torch.load(model_weights, weights_only=True))
        self.model.apply(set_batchnorm_eval)
        self.playouts = playouts
        self.game_timeout = game_timeout
        self.device = "cuda"
        self.model = self.model.to(self.device)
        self.proc_num = proc_num

    def __call__(self, fen: str):
        board = chess.Board(fen)
        game = Game.from_board(board)
        moves_eval = self.get_monte_values(
                game,
            )
        if game.board.turn:
            next_move = max(moves_eval, key=lambda x: (x[1], -x[2]))[0]
        else:
            next_move = min(moves_eval, key=lambda x: (x[1], x[2]))[0]
        return next_move.uci()


    def get_monte_values(
            self,
            game: Game,
            ) -> tuple[list[chess.Move], list[float], list[float]]:
        
        
        legal_moves = list(game.board.legal_moves)
        move_eval = []
        with mp.Manager() as manager:
            res_queue = manager.Queue()
            move_queue = manager.Queue()
            processes = []

            for move in legal_moves:
                move_queue.put(move)

            for _ in range(self.proc_num):
                p = mp.Process(
                    target=self.worker,
                    args=(
                        game,
                        move,
                        res_queue,
                        move_queue
                        )
                    )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            while not res_queue.empty():
                move_eval.append(res_queue.get())
        
        return move_eval
    
    def worker(self, game: Game, move, res_queue, move_queue):
        while not move_queue.empty():
            try:
                move = move_queue.get(timeout=1)
            except Exception:
                return
            # print(move_queue.qsize(), move)
            next_game = game.simulate_move(move)
            res, moves = self.monte_carlo_value(next_game)
            eval = move, res, moves
            res_queue.put(eval)

    def monte_carlo_value(self, next_game):
        res = []
        move_nums = []
        for _ in range(self.playouts):
            result, moves = self.playout_value(next_game)
            move_nums.append(moves)
            res.append(result)

        return sum(res)/len(res), sum(move_nums)/len(move_nums)
    
    def playout_value(self, game: Game, moves_num: list[int] = [0]):

        if game.over(self.game_timeout):
            return game.score(), moves_num[0]

        next_states = []
        moves = []
        next_boards = []
        
        for move in game.valid_moves():
            moves.append(move)
            tempGame = game.simulate_move(move)
            next_states.append(tempGame.tensor)
            next_boards.append(tempGame.board)

        final = torch.stack(next_states).to(self.device)
        heus = self.model(final)
        heus = Game.mask_mates(heus, next_boards)
        del final
        best_idx, move, _ = self.get_move(heus, moves, game.board.turn)
        del heus
        next_game = Game()
        next_game.board = next_boards[best_idx]
        next_game.tensor = next_states[best_idx]

        moves_num[0] += 1
        value = self.playout_value(next_game, moves_num)[0]
        
        gc.collect()
        torch.cuda.empty_cache()
        return value, moves_num[0]
    

    def get_move(self, heus_gpu: torch.Tensor, moves: list[chess.Move], white_moves):
        heus = heus_gpu.detach().cpu()
        del heus_gpu
        if white_moves:
            best = torch.argmax(heus).item()
            return best, moves[best], torch.max(heus).item()
        else:
            best = torch.argmin(heus).item()
            return best, moves[best], torch.min(heus).item()


