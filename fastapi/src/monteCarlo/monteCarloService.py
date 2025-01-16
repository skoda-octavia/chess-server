import torch
from src.monteCarlo.models import Heu
import torch.multiprocessing as mp
import chess
from src.game.gameEnv import Game
import gc
from copy import deepcopy
from src.service import Service


class MonteCarloService(Service):

    def __init__(
        self,
        device,
        model_weights,
        layers,
        playouts,
        game_timeout,
        bin_num,
        proc_num=32,
    ):
        self.model = Heu(6 * 8 * 8 + 1, bin_num, layers, 0)
        mp.set_start_method("forkserver", force=True)
        self.model.load_state_dict(torch.load(model_weights, weights_only=True))
        self.model.eval()
        self.playouts = playouts
        self.game_timeout = game_timeout
        self.device = device
        self.model = self.model.to(self.device)
        self.proc_num = proc_num
        self.noise_cooldown = 1
        self.base_noise_std = 0.05
        self.bin_num = bin_num

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

        current = game.board_value(game, self.model)
        return next_move, current

    def get_monte_values(
        self,
        game: Game,
    ) -> tuple[list[chess.Move], list[float], list[float]]:

        legal_moves = list(game.board.legal_moves)
        move_evaluations = {move.uci(): [] for move in legal_moves}
        move_lengths = {move.uci(): [] for move in legal_moves}
        legal_moves = list(game.board.legal_moves)
        move_eval = []
        with mp.Manager() as manager:
            res_queue = manager.Queue()
            move_queue = manager.Queue()
            processes = []

            for move in legal_moves:
                for _ in range(self.playouts):
                    move_queue.put(move)

            for _ in range(self.proc_num):
                p = mp.Process(
                    target=self.worker, args=(game, move, res_queue, move_queue)
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            while not res_queue.empty():
                move, value, moves_num = res_queue.get()
                move_evaluations[move].append(value)
                move_lengths[move].append(moves_num)
                move_eval.append(res_queue.get())

            for move in move_evaluations.keys():
                move_eval.append(
                    (
                        move,
                        sum(move_evaluations[move]) / len(move_evaluations[move]),
                        sum(move_lengths[move]) / len(move_lengths[move]),
                    )
                )

        return move_eval

    def worker(
        self, game: Game, move: chess.Move, res_queue: mp.Queue, move_queue: mp.Queue
    ):
        while not move_queue.empty():
            try:
                move = move_queue.get(timeout=1)
            except Exception:
                return
            # print(move_queue.qsize(), move)
            next_game = game.simulate_move(move)
            model = deepcopy(self.model)

            result, moves = self.playout_value(next_game, model, moves_num=[0])
            eval = move.uci(), result, moves
            res_queue.put(eval)

    def playout_value(
        self,
        game: Game,
        model: Heu,
        moves_num: list[int] = [0],
        current_value: float = None,
    ):

        if game.over(self.game_timeout):
            score = game.score(self.bin_num)
            if score == 0.5:
                score = current_value
            return score, moves_num[0]

        next_states = []
        moves = []
        next_boards = []

        for move in game.valid_moves():
            moves.append(move)
            tempGame = game.simulate_move(move)
            next_states.append(tempGame.tensor)
            next_boards.append(tempGame.board)

        final = torch.stack(next_states).to(self.device)
        final = final.view(final.shape[0], 6 * 8 * 8)
        next_turn = -1 if game.board.turn else 1
        move_column = torch.full((final.shape[0], 1), next_turn).to(self.device)
        final = torch.cat((final, move_column), dim=1)

        heus = model(final)
        heus = heus.float()
        self.apply_noise(heus, moves_num[0])

        heus = Game.mask_mates(heus, next_boards, self.bin_num)
        del final
        best_idx, move, _ = self.get_move(heus, moves, game.board.turn)
        best_heu = heus[best_idx].item()
        del heus
        next_game = Game()
        next_game.board = next_boards[best_idx]
        next_game.tensor = next_states[best_idx]

        moves_num[0] += 1
        value = self.playout_value(next_game, model, moves_num, best_heu)[0]

        gc.collect()
        torch.cuda.empty_cache()
        return value, moves_num[0]

    def apply_noise(self, heus, depth):
        std_dev = self.base_noise_std * (self.noise_cooldown**depth)
        random_tensor = torch.normal(mean=0.0, std=std_dev, size=heus.size()).to(
            self.device
        )
        heus += random_tensor
        return heus

    def get_move(self, heus_gpu: torch.Tensor, moves: list[chess.Move], white_moves):
        heus = heus_gpu.detach().cpu()
        del heus_gpu
        if white_moves:
            best = torch.argmax(heus).item()
            return best, moves[best], torch.max(heus).item()
        else:
            best = torch.argmin(heus).item()
            return best, moves[best], torch.min(heus).item()
