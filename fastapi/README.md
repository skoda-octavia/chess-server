## FastApi server

Available endpoints ```get-move/<model>?fen=``` where model:
* Stockfish
* Minimax
* Alpha (not really but Monte Carlo)
* LSTM

Configure service parameters in [serviceParams.json](./serviceParams.json)

### Stockfish
Download engine from [Stockfish official site](https://stockfishchess.org/download/) and place path in serviceParams.


### Run

```
poetry run pytest
poetry install --no-root
poetry run uvicorn main:app --reload
```
