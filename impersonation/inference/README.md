# Impersonation detector: inference API

One profile screenshot in, one verdict out (`impersonation` / `not_impersonation`).
This folder is **self-contained**: it imports nothing from outside itself. It holds the code, the frozen model, and the client profiles.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in the values (the real .env is git-ignored)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Settings are read from the environment or from `inference/.env` (variables already set in the environment win): `RDS_HOST`, `RDS_USER`, `RDS_PASSWORD` (where the client data lives, see "Where the data comes from") and, optionally, `OPENAI_API_KEY` for the Luna step.

Startup loads OCR, CLIP and the model once (a few seconds, about 2 GB of RAM or GPU memory). A GPU is used when available (`DEVICE=cpu` forces CPU).

## Where the data comes from
A request carries only **the screenshot and a client id**. Only the frozen model is stored here (`model/frozen.joblib`). Everything about the client is fetched from RDS (schema `drp_clients`) and cached in memory for 5 minutes, so a change in the database is picked up without a restart:

| Needed | RDS table |
|---|---|
| the client exists and is used by the impersonation pipeline | `customers`, `pipeline_metadata` |
| brand name, official domain | `customers`, `customer_domains` (type `main`) |
| protected executives (name, title) | `executives` |
| the logo and the executive photos | `assets` |
| their precomputed CLIP vectors | `impersonation_embeddings` |

The client id is the RDS `customer_id` (for example `paypal`, `1_1_ionos`). `GET /clients` lists the valid ones.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | status, device, thresholds, whether Luna is available |
| GET | `/clients` | the protected clients you can check against |
| POST | `/predict` | multipart form: `file` = screenshot, `client` = a client id from `/clients`, `use_luna` = true/false (default true) |

```bash
curl -F file=@samples/imp_p1_apple_0.png -F client=apple http://localhost:8000/predict
python smoke_test.py            # sends every screenshot in samples/
```

Example answer (shortened):

```json
{"prediction": "impersonation", "is_impersonation": true, "decided_by": "rules",
 "client": "apple", "rule_score": 0.84, "ml_probability": 0.99, "luna": null,
 "parsed": {"platform": "X", "display_name": "...", "handle": "...", "link": "...", "followers": 1200},
 "checks": {"avatar_matches_client_logo": true, "avatar_matches_executive_photo": false, "verified_badge": false, "link_on_page": "other domain"},
 "timings_ms": {"ocr": 190, "parse": 0.2, "avatar": 20, "signals": 17, "model": 8, "total": 240}}
```

`decided_by` says which stage answered:
- `rules` (hand-written point rules, about 14% of pages): confident rule score.
- `ml` (LightGBM, about 82%): model probability at or below 0.02 or at or above 0.98.
- `luna`: the LLM answered a page that rules and model left unsure (about 4%).
- `ml_fallback`: the page was unsure and Luna was off, unavailable or failed, so the model's best guess (probability >= 0.5) is returned. Treat these as lower confidence.

Errors: `400` not an image, `404` unknown client, `413` file too large, `503` RDS not configured or unreachable.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none | key for the Luna step; without it unsure pages use `ml_fallback` |
| `LUNA_ENABLED` | `1` | `0` turns the Luna step off |
| `LUNA_MODEL` / `LUNA_TIMEOUT_S` | `gpt-6-luna` / `60` | model name and per-call timeout |
| `DEVICE` | `auto` | `cpu` forces CPU |
| `MAX_UPLOAD_MB` | `10` | upload limit |
| `RDS_HOST`, `RDS_USER`, `RDS_PASSWORD` | none | required: where the client data lives |
| `RDS_PORT`, `RDS_DBNAME`, `RDS_SSLMODE`, `RDS_SCHEMA` | `5432`, `customerdata`, `require`, `drp_clients` | connection details |
| `CLIENT_CACHE_TTL_S` | `300` | how long a client's data is cached |
| `MODEL_FILE` | inside this folder | override the model location |

## Folder

```
app.py                 the HTTP API (FastAPI)
pipeline/              ocr.py, layout_parser.py, avatar.py, signals.py, rules.py, features.py, luna.py, engine.py (runs the steps), clients.py (reads RDS), settings.py
model/frozen.joblib    the frozen LightGBM model + the rule and model thresholds
samples/               a few test screenshots + samples.json (used by smoke_test.py)
smoke_test.py          quick check of a running server
```

## Adding a client
Add the client to the RDS tables above (`customers`, a `pipeline_metadata` row for `impersonation`, a main domain, executives, the logo and photos in `assets`, and their vectors in `impersonation_embeddings`). No restart is needed after the cache expires.

## Things to know
- **Input format.** The layout parser is calibrated to 900 x 900 px screenshots of X, Facebook, Instagram, LinkedIn, Telegram and TikTok profile pages as rendered in the benchmark. Other sizes are resized to 900 x 900 (reported as `image_resized_to_900`); real screenshots may need the parser's positions re-tuned.
- **Data.** The model was trained and tested on a synthetic benchmark (templated wording, AI-generated executive faces, fictional executives). Check it on real screenshots before relying on it.
- **Speed.** About 0.3 s per page on a GPU without Luna; about 4 s on CPU only. Luna adds roughly 0.2 to 1.3 s on the few unsure pages. Requests are served one at a time per process (the models are shared); run several processes for more throughput.
- **Frozen.** `model/frozen.joblib` is the model frozen after training on train and checked on test. To retrain, use the batch pipeline in `../code` and copy the new `frozen.joblib` here.
