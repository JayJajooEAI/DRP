# Physical Threat Facilities: detection pipeline (development)

The API for this detector is in `../../inference/PhysicalThreatFacility/`. This folder holds the code, data and trained models it was built from (version of 2026-11-16).

The pipeline decides whether a piece of content (a post, alert or article) is a **physical threat**: a danger or disruption to people or places, tied to a real location.

It is a three-rung cost ladder. The cheap models run first, and GPT-6 Luna is called only for the messages the cheap models are unsure about.

```
message ──(has a link? open it, add the page text)──► RUNG 1  names a location?  ── no ──►  not a threat
                                                          │ yes
                                                       RUNG 2  DistilBERT P(threat)   ≥ 0.6 ► threat
                                                          │ 0.3 – 0.6                < 0.3 ► not a threat
                                                       RUNG 3  GPT-6 Luna (prompt v14) decides
```

| Rung | What it does | Model | Time / message | Cost / 1,000 messages |
|---|---|---|---|---|
| 1 | Rule-based filter: does the message name a location? | MiniLMv2 fine-tuned, with place rules as fallback | 0.60 ms | $0.00008 |
| 2 | Threat or not | DistilBERT multilingual, fine-tuned | 2.96 ms | $0.00043 |
| 3 | Decides the unsure band (0.3–0.6) | GPT-6 Luna, prompt v14 | 1,688 ms | $0.297 (no caching) |

Rungs 1–2 are costed as GPU time on an AWS g4dn.xlarge (Tesla T4) at $0.526 per hour. Rung 3 is costed at GPT-6 Luna's billed token prices with no caching discount: $0.10 per 1M input tokens and $0.50 per 1M output tokens.

## Results on the test set (2,850 messages: 1,472 threats, 1,378 non-threats)

| Stage | Messages |
|---|---|
| Rung 1 filters out (no location) | 891 (886 non-threats, 5 threats) |
| Rung 2 decides | 1,956: 1,474 threat · 482 not a threat |
| Rung 3 (GPT-6 Luna) | 3, all correctly not a threat |
| **Final** | **TP 1,467 · FP 7 · FN 5 · TN 1,371** |
| Scores | precision 99.53% · recall 99.66% · F1 99.59% |
| **Cost for the 2,850 messages** | **ladder $0.0020** vs **$0.847** if every message went to GPT-6 Luna (**433× cheaper**) |

- **The 5 missed threats:** the location is written only as a code, hashtag or adjective, so Rung 1 finds no place: `#lebanon24`, `#iawx`, Czech *belgický* ("Belgian"), "KZN" inside a hashtag, and lowercase "black horse pike".
- **The 7 false alarms:** all come from Rung 2, on non-threats that mention a place, such as a news item about police questioning a basketball player, or a sentence about the 1923 Kanto earthquake.
- **Where to find them:** the IDs are in `outputs/shipped_results/test_run_summary.json`, and every message's path through the ladder is in `outputs/shipped_results/test_predictions.csv`.

### Rung 1: GLiNER vs MiniLMv2 fine-tuned (custom location identifier)

These were tested on the 1,472 test threats that name a place (a street, city, state, country or named venue), with GPT-6 Luna's place labels as ground truth (`data/luna_location_labels_test_threats.csv`).

| Model | Found | FP | FN | Precision | Recall | Time / msg | $ / 1k msgs |
|---|---|---|---|---|---|---|---|
| GLiNER (urchade/gliner_multi-v2.1) | 1,460 | 0 | 12 | 100% | 99.18% | 65.9 ms | $0.0096 |
| **MiniLMv2 fine-tuned** | **1,467** | 0 | **5** | 100% | **99.66%** | **0.60 ms** | **$0.00008** |

Precision is 100% for both by construction: every tested message names a place, so a model cannot wrongly find one.

## Data (`data/`)

**Sources**
- **10,000 threats** from a CTI API export: Twitter, surface web, Facebook, weather alerts, travel feeds, Bluesky and web search.
- **9,193 non-threats** from public datasets:

| Source | Count |
|---|---|
| Civil Comments | 2,266 |
| Figurative disaster tweets (target 0) | 2,065 |
| GoEmotions (Reddit) | 1,502 |
| XNLI, 5 languages | 1,494 |
| AG News | 1,403 |
| Synthetic cyber / fraud / outage messages | 463 |

  There were originally 10,000 non-threats. GPT-6 Luna (prompt v13) flagged **807** of them as possibly being threats, and those were removed rather than relabelled.

**Split:** 70 / 15 / 15, stratified by source, seed 42. To make train exactly 50/50, **563** of the 7,000 train threats were set aside (`dropped_for_train_balance = True`). These 563 were never used for training, tuning or testing.

| File | Rows | Contents |
|---|---|---|
| `teacher_labels/` | 2 files | GLiNER's place labels for every train and val message (`teacher_train.jsonl`, `teacher_val.jsonl`), the labels the shipped location model was trained on |
| `dataset_clean.csv` | 19,165 | Every message except the 28 data-quality rows. Columns: `id`, `text`, `label` (1 = threat), `label_name`, `source`, `split`, `dropped_for_train_balance`, `note` |
| `data_quality_issues.csv` | 28 | Test threats whose text names **no place at all**. The location exists only in the CTI analyst's triage note (`cti_location`). Columns add `issue`, `reason`, `cti_location`, `cti_url` |
| `splits/train.csv` | 12,874 | 6,437 threats + 6,437 non-threats, exactly what both models were trained on |
| `splits/val.csv` | 2,878 | 1,500 + 1,378, used to pick the Rung 1 cut-off |
| `splits/test.csv` | 2,850 | 1,472 + 1,378 (the test split minus the 28). Never used for training or tuning |
| `luna_location_labels_test_threats.csv` | 1,500 | GPT-6 Luna's place labels for every test threat: street, city, state, country, other named place |

**The 28 data-quality rows**
- **No usable text (4):** only a link, only @handles, or a scraped login page.
- **No place in the text (24):** live-update timelines, school-lockdown notices, very short messages, headlines naming no place, non-English messages with no place, and messages that give only an agency abbreviation such as "NYPD".
- **Why they are removed:** the triage notes that do have the location are audit evidence only. They are never model input.
- **What was checked:** only test threats were checked for data quality. A blank flag on a train or val row means "not checked", not "clean".

**The 5 location misses stay in the data.** They do name a place, only as a code or hashtag (`note` column), and they count as Rung 1's 5 missed threats.

## Rungs (code and models)

The code is in `code/`; the trained models are in `outputs/` (`02_train_location_model/model`, `03_train_threat_model/model`). Nothing is downloaded at run time except for retraining and step 01. Each module also runs on its own, from the `PhysicalThreatFacility` folder:

```bash
python code/location.py "Shots fired near 5th Ave and Pine St, Seattle"
```

### Rung 1: `code/location.py`
- **Base model:** `nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large` (6 layers, 384 hidden, 107M parameters, multilingual).
- **Task:** fine-tuned as a word tagger with one tag, "place" (B / I / O).
- **Input preparation:** links and @handles are removed, hashtags are kept as words (`#ChicagoScanner` becomes "Chicago Scanner"), and the first 3,000 characters are split into chunks of up to 1,000 characters.
- **Decision:** a message names a location when any word piece has P(B) + P(I) ≥ **0.4**.
- **Fallback:** when the model finds nothing, place rules run on the raw text:
  - street-address shapes, "X County" and ", TX";
  - Japanese and Chinese place suffixes;
  - a gazetteer of countries, US states and cities of 150k+ people.
- **Training labels:** GLiNER (`urchade/gliner_multi-v2.1`) marked the places in every train message, and MiniLMv2 was trained to mark the same words. GLiNER is the teacher and MiniLMv2 the student. GPT-6 Luna's labels are used only to test.
  - `code/01_make_teacher_labels.py` rebuilds those labels.
  - `data/teacher_labels/teacher_train.jsonl` and `teacher_val.jsonl` are the exact labels the shipped model was trained on.
- **Training run:** `code/02_train_location_model.py`, 8 epochs, AdamW, lr 5e-5, batch 32, fp16, cross-entropy per word piece.
- **Cut-off:** chosen on val as the value with the fewest val threats lost, provided it loses no more than GLiNER (17 vs 23) and lets through at most 10% more non-threats than GLiNER (487 vs 456). That gives 0.4.

### Rung 2: `code/threat.py`
- **Model:** `distilbert-base-multilingual-cased`, fine-tuned on `splits/train.csv` (`code/03_train_threat_model.py`): 3 epochs, lr 2e-5, batch 32, max 128 tokens, fp16, seed 42.
- **Output:** P(threat). ≥ 0.6 is a threat, < 0.3 is not a threat, and 0.3–0.6 goes to Rung 3.
- **On its own** (threshold 0.5, the 2,878-message test split before the 28 were removed): precision 99.20%, recall 99.73%, F1 99.47%.

### Rung 3: `code/luna.py`
- **Prompt:** `code/prompt_v14.txt`, the final prompt. Its core rule: a physical threat is a **threat plus an anchor** (a pinned-down location, or a named event that identifies its place). v14 is v13 plus three reviewed fixes:
  - rallies of any kind count;
  - an online or hoax threat of violence counts;
  - a place counts however it is written (hashtag, abbreviation, any language).
- **Answer:** strict JSON (`code/response_schema.json`): `label`, `subtype` (one of 32 CTI subtypes, "other", or "none"), `confidence` and `reason`. Texts are cut to 4,000 characters. Temperature 0.
- **Metadata:** `luna.py` returns the full usage and request metadata for each call.
- **On its own,** on the 2,878-message test split: TP 1,473, FN 27, FP 0, TN 1,378. At least 20 of the 27 misses are among the 28 data-quality rows (messages that name no place).
- **API key:** put `OPENAI_API_KEY=...` in the environment, or in a `.env` file (the package root, or the file named by `OPENAI_ENV_FILE`). No key is included.

### Enrichment: `code/enrich.py`
This runs only when the message itself contains a link:
1. Find the links with `urlextract`.
2. Open each one in headless Chromium with Playwright.
3. Append all the page text: `message + "\n[linked page] " + text`.

There are no lookups and no field picking, and a link that does not open adds nothing. It is off by default in `04_run_ladder.py` (`--enrich` turns it on). The results above were produced without it. It was tried live on a few demo messages only, and old links are often dead.

## How to run

Run from the `PhysicalThreatFacility` folder. Packages are listed in `requirements.txt` (the Python environment is not part of this folder).

| Step | Script | What it does | Writes (in `outputs/`) |
|---|---|---|---|
| 01 | `01_make_teacher_labels.py` | GLiNER marks the places in every train and val message (needs `pip install gliner`; optional, the shipped labels are in `data/teacher_labels/`) | `01_make_teacher_labels/` |
| 02 | `02_train_location_model.py` | Trains the Rung 1 location model from the teacher labels and picks the cut-off on val | `02_train_location_model/model_retrained/` |
| 03 | `03_train_threat_model.py` | Trains the Rung 2 DistilBERT classifier (about 3 minutes on a T4) | `03_train_threat_model/model_retrained/` |
| 04 | `04_run_ladder.py` | Runs the ladder on a CSV with a `text` column | `04_run_ladder/<name>_pipeline.csv` and `_pipeline_summary.json` |

```bash
python code/04_run_ladder.py data/splits/test.csv --no-gpt     # Rungs 1-2, free; reproduces 891 / 1,474 / 482 / 3
python code/04_run_ladder.py data/splits/test.csv              # all rungs (3 GPT-6 Luna calls, about $0.001)
python code/04_run_ladder.py --text "Shots fired near 5th Ave and Pine St, Seattle"
python code/04_run_ladder.py my_messages.csv --enrich          # links are opened (needs Playwright)
```

Retrained models are saved as `model_retrained/` next to the shipped `model/`, which is never overwritten. The ladder reads the shipped `model/` folders.

**Checked on 2026-11-16:** step 04 on the test set reproduces the rung counts above exactly (891 stop at rung 1, 1,956 are decided by rung 2, 3 go to Luna; 5 missed threats), and P(threat) is
identical to the shipped `test_predictions.csv`. The final false-alarm count was 8 instead of 7 because GPT-6 Luna answered one borderline message differently. Step 03 retrained to val F1
0.993 and test precision 99.19%, recall 100%. Steps 02 and 01 ran (step 02 for 1 epoch only; step 01 on 5 messages, with labels identical to the shipped ones).

## Folder layout

```
README.md  requirements.txt
code/      01_make_teacher_labels.py · 02_train_location_model.py · 03_train_threat_model.py · 04_run_ladder.py
           location.py · threat.py · luna.py · enrich.py · prompt_v14.txt · response_schema.json
data/      dataset_clean.csv · data_quality_issues.csv · splits/ · luna_location_labels_test_threats.csv · teacher_labels/
outputs/   02_train_location_model/model · 03_train_threat_model/model   (the trained models, as shipped)
           04_run_ladder/ (written by step 04) · shipped_results/ (deck, test_predictions.csv, test_run_summary.json, location_model_comparison.csv, cost_and_latency.json)
```

## Not included
- **The raw CTI export** (662 MB), the source of the 10,000 threats: `s3://aish-drp-open-source-data/physical-threat/physical_threat_cti_api.csv`.
- **The dataset as one file, with the 28 still in:** `s3://aish-drp-open-source-data/physical-threat-facilities/physical_threat_dataset_2026-11-05.csv`.
- **Earlier work:** prompt versions v1–v13 and their runs, the other models tried (keyword rules, TF-IDF, embeddings, lighter GLiNER variants, spaCy and other location models), and the interactive demo page.

## Caveats
- **Location test:** in the location model comparison, every tested message names a place, so it measures misses (recall) only. GPT-6 Luna did not label the non-threats for places.
- **Training labels:** the Rung 1 model's training labels come from GLiNER, not from GPT-6 Luna, so it learned GLiNER's idea of a place. It still misses fewer test places than GLiNER.
- **Prices:** cloud prices are on-demand list prices (us-east-1) at the time of the work. Check them against current pricing.
