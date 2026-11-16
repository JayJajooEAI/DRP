# inference/impersonation: the impersonation detector

One profile screenshot in, one verdict out (`impersonation` / `not_impersonation`).
It is served by the shared server in `../app.py` (see `../README.md`); this folder holds the code and the frozen model, and imports nothing from outside `inference/`.

## Run

Start the shared server from `inference/` (see `../README.md`). Settings come from the environment or from `inference/.env` (variables already set in the environment win): `RDS_HOST`, `RDS_USER`, `RDS_PASSWORD` and, optionally, `OPENAI_API_KEY` for the Luna step.

Startup loads OCR, CLIP and the model once (a few seconds, about 2 GB of RAM or GPU memory). A GPU is used when available (`DEVICE=cpu` forces CPU).

## Where the data comes from
A request carries only **the screenshot and a client id**. Only the frozen model is stored here (`model/frozen.joblib`). Everything about the client is fetched from RDS (schema `drp_clients`), on every request, so a change in the database applies to the next call without a restart:

| Needed | RDS table |
|---|---|
| the client exists and is used by the impersonation pipeline | `customers`, `pipeline_metadata` |
| brand name, official domain | `customers`, `customer_domains` (type `main`) |
| protected executives (name, title) | `executives` |
| the logo and the executive photos | `assets` |
| their precomputed CLIP vectors | `impersonation_embeddings` |

The client id is the RDS `customer_id` (for example `paypal`, `1_1_ionos`). `GET /impersonation/clients` lists the valid ones.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` (shared) | status of both incident types; for this one: device, thresholds, whether Luna is available |
| GET | `/impersonation/clients` | the protected clients you can check against |
| POST | `/impersonation/predict` | multipart form: `file` = screenshot, `client` = a client id from `/clients`, `use_luna` = true/false (default true) |

```bash
curl -F file=@impersonation/samples/imp_p1_apple_0.png -F client=apple http://localhost:8000/impersonation/predict
python smoke_test.py            # run from inference/: sends the samples to both incident types
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
| `MODEL_FILE` | inside this folder | override the model location |

## Folder

```
engine.py, ocr.py, layout_parser.py, avatar.py, signals.py, rules.py, features.py, luna.py   the steps (below)
clients.py   reads the clients from RDS
settings.py  paths and switches
model/       frozen.joblib: the frozen LightGBM model + the rule and model thresholds
samples/     a few test screenshots + samples.json (used by ../smoke_test.py; not in git)
```

### The steps

| File | Step |
|---|---|
| `ocr.py` | Reads the text on the screenshot (docTR) |
| `layout_parser.py` | Turns the text lines into fields: platform, name, handle, bio, link, stats, posts |
| `avatar.py` | Crops the profile picture and compares it with the client's logo and executive photos (CLIP vectors from RDS, plus a pixel check) |
| `signals.py` | About 100 numbers from the fields, the avatar scores and the client profile |
| `rules.py` | Hand-written points, scaled to 0 to 1; decides when the score is clear |
| `features.py` | Builds the feature table the frozen LightGBM model expects |
| `luna.py` | Asks Luna (OpenAI Decisions API) about pages the rules and the model leave unsure |
| `clients.py` | Fetches the client's profile, logo, executive photos and stored vectors from RDS |
| `settings.py` | Paths and switches; reads the environment and `inference/.env` |
| `engine.py` | Loads the models once, then runs the steps and builds the answer |

## Adding a client
Add the client to the RDS tables above (`customers`, a `pipeline_metadata` row for `impersonation`, a main domain, executives, the logo and photos in `assets`, and their vectors in `impersonation_embeddings`). No restart is needed.

## Things to know
- **Input format.** The layout parser is calibrated to 900 x 900 px screenshots of X, Facebook, Instagram, LinkedIn, Telegram and TikTok profile pages as rendered in the benchmark. Other sizes are resized to 900 x 900 (reported as `image_resized_to_900`); real screenshots may need the parser's positions re-tuned.
- **Data.** The model was trained and tested on a synthetic benchmark (templated wording, AI-generated executive faces, fictional executives). Check it on real screenshots before relying on it.
- **Speed.** About 0.3 s per page on a GPU without Luna; about 4 s on CPU only. Luna adds roughly 0.2 to 1.3 s on the few unsure pages. Requests are served one at a time per process (the models are shared); run several processes for more throughput.
- **Frozen.** `model/frozen.joblib` is the model frozen after training on train and checked on test. To retrain, use the batch pipeline in `../../development/impersonation/code` and copy the new `frozen.joblib` here.
