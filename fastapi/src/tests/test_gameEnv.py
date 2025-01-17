from ..game.gameEnv import Game
import torch
import chess

opening_tensor = [
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, -1, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
    ],
    [
        [0, 0, -1, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
    ],
    [
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ],
    [
        [0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
    ],
]


def test_create_board():
    board_tensor = torch.tensor(opening_tensor)
    game = Game.from_tensor(board_tensor)
    assert game.board.board_fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert game.board.is_valid()


def test_from_tensor():
    tensor = torch.zeros((6, 8, 8))
    tensor[0, 7, 0] = 1
    tensor[1, 6, 1] = -1

    game = Game.from_tensor(tensor)

    assert isinstance(game, Game)
    assert game.tensor.equal(tensor)
    assert game.board.piece_at(chess.A1) == chess.Piece(chess.PAWN, chess.WHITE)
    assert game.board.piece_at(chess.B2) == chess.Piece(chess.KNIGHT, chess.BLACK)


def test_from_board():
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))

    game = Game.from_board(board)

    assert isinstance(game, Game)
    assert game.board == board
    assert torch.sum(game.tensor != 0) > 0


def test_create_tensor():
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.C6, chess.Piece(chess.BISHOP, chess.BLACK))

    tensor = Game.create_tensor(board)

    assert tensor.shape == (6, 8, 8)
    assert tensor[4, 4, 4] == 1
    assert tensor[2, 2, 2] == -1


def test_create_board():
    tensor = torch.zeros((6, 8, 8))
    tensor[0, 7, 0] = 1
    tensor[4, 0, 7] = -1

    board = Game.create_board(tensor)

    assert isinstance(board, chess.Board)
    assert board.piece_at(chess.A1) == chess.Piece(chess.PAWN, chess.WHITE)
    assert board.piece_at(chess.H8) == chess.Piece(chess.QUEEN, chess.BLACK)
    assert board.piece_at(chess.C3) is None

def test_mask_mates():
    heus = [0, 0, 0]
    board1 = chess.Board()
    board2 = chess.Board()
    board3 = chess.Board()

    board1.set_fen("8/8/8/8/8/K7/8/k1R5 b - - 0 1")
    board2.set_fen("8/8/3q4/K2q4/3q4/8/8/k7 w - - 0 1")
    board3.set_fen("7k/8/8/8/8/8/8/7K w - - 0 1")

    boards = [board1, board2, board3]
    bin_num = 5

    heus = Game.mask_mates(heus, boards, bin_num)

    assert heus[0] == bin_num + Game.win_bias
    assert heus[1] == -Game.win_bias
    assert heus[2] == 0


def test_state():
    tensor = torch.zeros((6, 8, 8))
    tensor[0, 7, 0] = 1
    game = Game.from_tensor(tensor)

    state = game.state()

    assert torch.equal(state, tensor)


def test_valid_moves():
    board = chess.Board()
    game = Game.from_board(board)

    moves = list(game.valid_moves())

    assert isinstance(moves, list)
    assert all(isinstance(move, chess.Move) for move in moves)
    assert len(moves) > 0


def test_make_move():
    board = chess.Board()
    game = Game.from_board(board)

    move = chess.Move.from_uci("e2e4")

    new_game = game.make_move(move)

    assert isinstance(new_game, Game)
    assert new_game.board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert new_game.board.turn == chess.BLACK
    assert game.board.piece_at(chess.E4) is None

def test_simulate_move():
    board = chess.Board()
    game = Game.from_board(board)

    move = chess.Move.from_uci("e2e4")

    new_game = game.simulate_move(move)

    assert isinstance(new_game, Game)
    assert new_game.board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert new_game.board.turn == chess.BLACK
    assert game.board.piece_at(chess.E4) is None


def test_over():
    board = chess.Board("8/8/R7/3k4/R7/8/8/K1R1R3 b - - 75 1")
    game = Game.from_board(board)
    assert game.over()
    assert game.outcome.termination == chess.Termination.STALEMATE

    board = chess.Board("8/8/8/3k4/8/8/8/K1RRR3 b - - 1 1")
    game = Game.from_board(board)
    assert game.over()
    assert game.outcome.winner == chess.WHITE


def test_score():
    board = chess.Board()
    game = Game.from_board(board)

    assert game.score(bin_num=10) == 0.5

    board.set_fen("8/8/8/k7/8/8/R7/KR6 b - - 1 1")
    game = Game.from_board(board)
    assert game.score(bin_num=10) == 10 + game.win_bias

    board.set_fen("8/8/8/4b3/4b3/k7/8/K7 w - - 1 1")
    game = Game.from_board(board)
    assert game.score(bin_num=10) == -game.win_bias


def test_board_value():
    board = chess.Board()
    game = Game.from_board(board)

    class MockModel:
        def __call__(self, tensor):
            return torch.tensor([1])

    model = MockModel()

    value = game.board_value(model)
    assert value == 1

    board.set_fen("8/8/8/k7/8/8/R7/KR6 b - - 1 1")
    game = Game.from_board(board)
    assert game.board_value(model) == Game.max_evaluation + 1

    board.set_fen("8/8/8/4b3/4b3/k7/8/K7 w - - 1 1")
    game = Game.from_board(board)
    assert game.board_value(model) == -1