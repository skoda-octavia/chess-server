from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import torch
from contextlib import asynccontextmanager
from src.lstm.lstmService import LstmService
from src.monteCarlo.monteCarloService import MonteCarloService
from src.miniMax.miniMaxService import MiniMaxService
from src.stockfish.stockfishService import StockfishService
import csv
import json
from src.game.gameEnv import Game

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:4200",
]


class ModelResponse(BaseModel):
    model: str
    fen: str
    move: str
    eval: float = None


services = {}
fen_dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    openings_path = "openings.csv"
    service_params = "serviceParams.json"
    with open(openings_path, "r", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        try:
            next(csv_reader)
            for row in csv_reader:
                key, value = row
                fen_dict[key] = value
        except StopIteration:
            pass

    with open(service_params, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device("cpu")
    print("Device: ", device)

    services["LSTM"] = LstmService(device=device, **config["services"]["LSTM"])
    services["Alpha"] = MonteCarloService(device=device, **config["services"]["Alpha"])
    services["Minimax"] = MiniMaxService(device=device, **config["services"]["Minimax"])
    services["Stockfish"] = StockfishService(**config["services"]["Stockfish"])

    Game.max_evaluation = config["game"]["max_evaluation"]
    Game.win_bias = config["game"]["win_bias"]

    print(f"Fen dict len: {len(fen_dict)}")
    for key, item in services.items():
        print(key, item)
    yield
    services.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/get-move/{model}")
def get_move(
    model: str, fen: str = Query(..., description="Forsyth-Edwards Notation string")
):
    try:
        selected_service = services[model]
    except KeyError:
        raise HTTPException(status_code=400, detail="Unsupported model")
    try:
        board = chess.Board(fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid fen: " + fen)

    fen_code = board.board_fen() + str(int(board.turn))
    if fen_code in fen_dict and model != "Stockfish":
        return ModelResponse(model=model, fen=fen, move=fen_dict[fen_code], eval=32.1)
    result = selected_service(fen)
    if len(result) == 2:
        move, eval = result
    else:
        eval = 32.1
        move = result
    with open("debug.txt", "a") as f:
        f.write(f"fen: {fen}, move: {move}, model: {model}")
    return ModelResponse(model=model, fen=fen, move=move, eval=eval)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
