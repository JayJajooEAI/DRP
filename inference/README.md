# inference: one server for both incident types

```
inference/
├── app.py              the server: routes requests to the right incident type
├── impersonation/      impersonation detector: code + frozen model (screenshot + client in, verdict out)
├── WebPhish/           web-phish detector: code + models (link + client in, verdict out)
├── smoke_test.py       quick check of a running server
├── requirements.txt
└── .env                RDS login and the OpenAI key (git-ignored; copy .env.example)
```

Each incident type is self-contained: its code, its models and its own README live in its folder, and neither imports from the other.
The server only routes. Both read their clients (and, for web phish, the stored pages) from RDS, schema `drp_clients`.

## Run

```bash
cd inference
cp .env.example .env            # fill in RDS_HOST, RDS_USER, RDS_PASSWORD (OPENAI_API_KEY is optional)
uvicorn app:app --host 0.0.0.0 --port 8000
```
Starting takes about a minute (every model loads once). A GPU is used when there is one (`DEVICE=cpu` forces CPU).
`INCIDENT_TYPES=impersonation` or `INCIDENT_TYPES=WebPhish` serves just one. If one fails to load, the other keeps working and the
failed one answers 503 with the reason (see `GET /health`).

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | which incident types are loaded |
| GET | `/impersonation/clients` | clients that can be checked for impersonation |
| POST | `/impersonation/predict` | multipart form: `file` (profile screenshot), `client`, `use_luna` |
| GET | `/webphish/clients` | clients that can be checked for web phishing |
| POST | `/webphish/predict` | JSON: `{"client": "...", "url": "..."}` or `{"client": "...", "content_id": 123}` |

```bash
curl -F file=@impersonation/samples/imp_p1_apple_0.png -F client=apple localhost:8000/impersonation/predict
curl -X POST localhost:8000/webphish/predict -H 'Content-Type: application/json' \
     -d '{"client": "paypal", "url": "http://infosantioana.hi2.ro/php/update/"}'
python smoke_test.py            # sends the samples to both
```
See `impersonation/README.md` and `WebPhish/README.md` for what each one does and returns.

## Where training lives
The code that builds the models is in `../development/` (`impersonation/`, `WebPhish/`). To update a model, retrain there and copy the new
weights into the incident type's `model/` folder.
