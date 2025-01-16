import chess
import chess.engine
from src.service import Service


class StockfishService(Service):
    def __init__(self, engine_path: str, skill: int):
        self.engine_path = engine_path
        self.skill = skill
        self.engine = None

    def start_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self.engine.configure({"Skill Level": self.skill})

    def stop_engine(self):
        if self.engine:
            self.engine.quit()

    def __call__(self, fen: str) -> str:
        board = chess.Board(fen)
        if not self.engine:
            self.start_engine()
        result = self.engine.play(board, chess.engine.Limit(time=0.1))
        return result.move.uci()
