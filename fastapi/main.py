from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import random
from contextlib import asynccontextmanager
from src.lstm.lstmService import LstmService
from src.alpha.alphaService import AlphaService

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
    services["lstm"] = LstmService()
    services["Alpha"] = AlphaService(
        "models/heu_150.pth",
        [384, 400, 500, 700, 700, 700, 500, 300, 200, 100, 64],
        50,
        50,
        0.0001
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


