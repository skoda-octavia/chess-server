from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import random
from contextlib import asynccontextmanager
from src.lstm.lstmService import LstmService
from src.alpha.alphaService import AlphaService
from src.miniMax.miniMaxService import MiniMaxService
from src.mlp.mlpService import MlpService

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

services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlp_weights_path = "models/mlp/"
    services["Lstm"] = LstmService()
    services["Alpha"] = AlphaService(
        "models/model_weights_200.pth",
        [384, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 100, 64],
        50,
        50,
        0.0001
        )
    services["Minimax"] = MiniMaxService(
        "models/model_weights_256.pth",
        [384+1, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 100, 64],
        0,
        4
    )

    services["MLP"] = MlpService(
        {
            chess.QUEEN: mlp_weights_path + "queen.pth",
            chess.ROOK: mlp_weights_path + "rook_450.pth",
            chess.BISHOP: mlp_weights_path + "bishop_200.pth",
            chess.KNIGHT: mlp_weights_path + "knight_300.pth",
            chess.PAWN: mlp_weights_path + "pawn_150.pth",
            chess.KING: mlp_weights_path + "king_300.pth"
        },
        mlp_weights_path + "selection_300.pth",
        [7*8*8, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 100, 64],
        [6*8*8, 500, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 100, 64]
    )

    print(services)
    yield
    services.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/get-move/{model}")
def get_move(model: str, fen: str = Query(..., description="Forsyth-Edwards Notation string")):
    try:
        selected_service = services[model]
    except KeyError:
        raise HTTPException(status_code=400, detail="Unsupported model")
    try:
        _ = chess.Board(fen)
    except Exception: 
        raise HTTPException(status_code=400, detail="Invalid fen: " + fen)
    move = selected_service(fen)
    with open("debug.txt", "a") as f:
        f.write(f"fen: {fen}, move: {move}, model: {model}")
    return ModelResponse(model=model, fen=fen, move=move)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


