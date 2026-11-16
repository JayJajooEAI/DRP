# Impersonation detection

Given a screenshot of a social-media profile and the client it might be impersonating, decide: **impersonation** or **not impersonation**.
The benchmark is synthetic: 25 real brands, 4,470 profile screenshots across six platforms (X, Facebook, Instagram, LinkedIn, Telegram, TikTok).

## What is in this folder

| Folder | What it is |
|---|---|
| `code/` | The batch pipeline: numbered scripts 01 to 10 that build, freeze and test the detector, plus `utilities/` |
| `data/` | The benchmark data: screenshots, train/val/test lists, client assets (not in git, copies are in S3) |
| `outputs/` | Everything the scripts write: intermediate results, the frozen model, reports (created on first run, not in git) |

The HTTP API for this detector is not here: it is in `../../inference/impersonation/` (served together with the web-phish detector).

## How it decides (a cascade)

1. **Read the screenshot.** OCR finds the text, a layout parser turns it into fields (name, handle, bio, link, stats, posts), and the profile picture is compared with the client's logo and executive photos (CLIP and pixel matching). About 100 numbers come out.
2. **Rules.** Hand-written points (brand in the name, logo match, "official" wording, scam wording, verified badge, ...) are added up and scaled to 0 to 1. A clear score decides (about 14% of pages).
3. **LightGBM.** A small model on the same numbers decides when it is very sure, at 2% or less or at 98% or more (about 82% of pages).
4. **Luna** (an LLM, through the OpenAI Decisions API) answers the few pages left unsure (about 4%).

The rule and model thresholds are fitted on train only. The pipeline is frozen with checksums before it is run on test.

## Results (synthetic benchmark)

| Split | Pages | Accuracy |
|---|---|---|
| Train (17 customers) | 3,036 | 99.87% (4 errors) |
| Val (4 customers) | 720 | 100% |
| Test (4 customers) | 714 | 100% (714 of 714) |

Splits are by customer, so no customer appears in two splits. Test has been scored several times on the same frozen pipeline and nothing was tuned on it.
Read the 100% with care: the data is generated (templated wording, AI-generated executive faces, fictional executives), and the parser is calibrated to the way these screenshots are rendered. Check on real screenshots before relying on it.

## Where the data lives

- **Local:** `data/` and `outputs/` (large, not in git).
- **S3** (`drp-pipeline/impersonation/`): the 4,470 screenshots and the metadata (`train.csv`, `val.csv`, `test.csv`, `classes.json`).
- **RDS** (schema `drp_clients`): the client profiles, logos, executive photos and their stored CLIP vectors, and the `content` table that gives every screenshot a number, its client and its split. The inference service (`../../inference/impersonation/`) reads clients from here.

## Quick start

```bash
cd code
python 01_ocr_read_screenshots.py train val test      # then 02, 04, 05, 06 ... see code/README.md
```
The API is started from the shared `inference/` folder (see `../../inference/README.md`).

Each folder has its own README.
