# Impersonation detector: inference API

One profile screenshot in, one verdict out (`impersonation` / `not_impersonation`).
This folder is **self-contained**: it imports nothing from outside itself. It holds the code, the frozen model, and the client profiles.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # optional: only needed for the Luna step (see below)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Startup loads OCR, CLIP and the model once (a few seconds, about 2 GB of RAM or GPU memory). A GPU is used when available (`DEVICE=cpu` forces CPU).

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | status, device, thresholds, whether Luna is available |
| GET | `/clients` | the protected clients you can check against |
| POST | `/predict` | multipart form: `file` = screenshot, `client` = a client id from `/clients`, `use_luna` = true/false (default true) |

```bash
curl -F file=@samples/imp_p1_apple_0.png -F client=Apple http://localhost:8000/predict
python smoke_test.py            # sends every screenshot in samples/
```

Example answer (shortened):

```json
{"prediction": "impersonation", "is_impersonation": true, "decided_by": "rules",
 "client": "Apple", "rule_score": 0.84, "ml_probability": 0.99, "luna": null,
 "parsed": {"platform": "X", "display_name": "...", "handle": "...", "link": "...", "followers": 1200},
 "checks": {"avatar_matches_client_logo": true, "avatar_matches_executive_photo": false, "verified_badge": false, "link_on_page": "other domain"},
 "timings_ms": {"ocr": 190, "parse": 0.2, "avatar": 20, "signals": 17, "model": 8, "total": 240}}
```

`decided_by` says which stage answered:
- `rules` (hand-written point rules, about 14% of pages): confident rule score.
- `ml` (LightGBM, about 82%): model probability at or below 0.02 or at or above 0.98.
- `luna`: the LLM answered a page that rules and model left unsure (about 4%).
- `ml_fallback`: the page was unsure and Luna was off, unavailable or failed, so the model's best guess (probability >= 0.5) is returned. Treat these as lower confidence.

Errors: `400` not an image, `404` unknown client, `413` file too large.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none | key for the Luna step; without it unsure pages use `ml_fallback` |
| `LUNA_ENABLED` | `1` | `0` turns the Luna step off |
| `LUNA_MODEL` / `LUNA_TIMEOUT_S` | `gpt-6-luna` / `60` | model name and per-call timeout |
| `DEVICE` | `auto` | `cpu` forces CPU |
| `MAX_UPLOAD_MB` | `10` | upload limit |
| `MODEL_FILE`, `CLIENTS_DIR` | inside this folder | override the model / clients locations |

## Folder

```
app.py                 the HTTP API (FastAPI)
pipeline/              ocr.py, layout_parser.py, avatar.py, signals.py, rules.py, features.py, luna.py, engine.py (runs the steps), clients.py, settings.py
model/frozen.joblib    the frozen LightGBM model + the rule and model thresholds
clients/<id>/          client.json, logo.png, executives/*.jpg   (25 demo clients)
samples/               a few test screenshots + samples.json (used by smoke_test.py)
smoke_test.py          quick check of a running server
```

## Adding a client
Create `clients/<id>/` with `client.json` (`id`, `name`, `brand`, `domain`, `executives: [{name, role, photo}]`), `logo.png` and the executive photos, then restart.

## Things to know
- **Input format.** The layout parser is calibrated to 900 x 900 px screenshots of X, Facebook, Instagram, LinkedIn, Telegram and TikTok profile pages as rendered in the benchmark. Other sizes are resized to 900 x 900 (reported as `image_resized_to_900`); real screenshots may need the parser's positions re-tuned.
- **Data.** The model was trained and tested on a synthetic benchmark (templated wording, AI-generated executive faces, fictional executives). Check it on real screenshots before relying on it.
- **Speed.** About 0.3 s per page on a GPU without Luna; about 4 s on CPU only. Luna adds roughly 0.2 to 1.3 s on the few unsure pages. Requests are served one at a time per process (the models are shared); run several processes for more throughput.
- **Frozen.** `model/frozen.joblib` is the model frozen after training on train and checked on test. To retrain, use the batch pipeline in `../code` and copy the new `frozen.joblib` here.
