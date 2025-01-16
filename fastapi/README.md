## FastApi server

Available endpoints ```get-move/<model>?fen=``` where model:
* Stockfish
* Minimax
* Alpha (not really)
* LSTM

Configure service parameters in [serviceParams.json](./serviceParams.json)

### Run

```
poetry install --no-root
poetry run uvicorn main:app --reload
```