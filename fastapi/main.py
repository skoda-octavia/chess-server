from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import random

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelResponse(BaseModel):
    model: str
    fen: str
    move: str

@app.get("/get-move/{model}")
def get_move(model: str, fen: str = Query(..., description="Forsyth-Edwards Notation string")):
    if model not in ["LSTM", "MLP", "Alpha-zero"]:
        raise HTTPException(status_code=400, detail="Unsupported model")

    try:
        board = chess.Board(fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Illegal fen")
    move = handle_model(model, board)
    return ModelResponse(model=model, fen=fen, move=move)

def handle_model(model: str, board: chess.Board) -> str:
    moves = list(board.legal_moves)
    move = random.choice(moves) #TODO integrate
    if model == "LSTM":
        return move.uci()
    elif model == "MLP":
        return move.uci()
    elif model == "Alpha-zero":
        return move.uci()

    return "unknown_move"