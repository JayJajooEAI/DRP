# inference: one server for all incident types

```
inference/
├── app.py              the server: routes requests to the right incident type
├── impersonation/      impersonation detector: code + frozen model (screenshot + client in, verdict out)
├── WebPhish/           web-phish detector: code + models (link + client in, verdict out)
├── PhysicalThreatFacility/    physical-threat detector: code + models (message in, verdict out; no client)
├── smoke_test.py       quick check of a running server
├── requirements.txt
└── .env                RDS login and the OpenAI key (git-ignored; copy .env.example)
```

Each incident type is self-contained: its code, its models and its own README live in its folder, and none imports from another.
The server only routes. Impersonation and web phish read their clients (and, for web phish, the stored pages) from RDS, schema `drp_clients`; physical threat needs no client or database.

## Run

```bash
cd inference
cp .env.example .env            # fill in RDS_HOST, RDS_USER, RDS_PASSWORD (OPENAI_API_KEY is optional)
uvicorn app:app --host 0.0.0.0 --port 8000
```
Starting takes about a minute (every model loads once). A GPU is used when there is one (`DEVICE=cpu` forces CPU).
`INCIDENT_TYPES=impersonation`, or a list like `INCIDENT_TYPES=impersonation,WebPhish`, serves only those. If one fails to load, the other keeps working and the
failed one answers 503 with the reason (see `GET /health`).

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | which incident types are loaded |
| GET | `/impersonation/clients` | clients that can be checked for impersonation |
| POST | `/impersonation/predict` | multipart form: `file` (profile screenshot), `client`, `use_luna` |
| GET | `/webphish/clients` | clients that can be checked for web phishing |
| POST | `/webphish/predict` | JSON: `{"client": "...", "url": "..."}` |
| POST | `/physicalthreat/predict` | JSON: `{"text": "..."}` (optional `use_luna`) |

```bash
curl -F file=@impersonation/samples/imp_p1_apple_0.png -F client=apple localhost:8000/impersonation/predict
curl -X POST localhost:8000/webphish/predict -H 'Content-Type: application/json' \
     -d '{"client": "paypal", "url": "http://infosantioana.hi2.ro/php/update/"}'
curl -X POST localhost:8000/physicalthreat/predict -H 'Content-Type: application/json' \
     -d '{"text": "Shots fired near 5th Ave and Pine St, Seattle"}'
python smoke_test.py            # sends the samples to every incident type
```
See `impersonation/README.md`, `WebPhish/README.md` and `PhysicalThreatFacility/README.md` for what each one does and returns.

## Model weights
The large weight files are not in git (GitHub's 100 MB limit): the web-phish language models and Phishpedia files, and the two physical-threat models. Each `model/README.md` lists them with sizes and SHA-256 checksums; copy them into place (S3 links to be added there).

## Where training lives
The code that builds the models is in `../development/` (`impersonation/`, `WebPhish/`, `PhysicalThreatFacility/`). To update a model, retrain there and copy the new
weights into the incident type's `model/` folder.
