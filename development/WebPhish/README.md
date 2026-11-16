# Web-phish detection (development)

Given a link and the client it might imitate, decide whether the page is a web phish: it presents itself as the client (check 2), it asks
for credentials or payment data, or has a login link (check 3), and it is not on the client's own domain (check 4).
The API for this detector is in `../../inference/WebPhish/`.

| Folder | What it is |
|---|---|
| `code/` | Numbered scripts in run order, plus helpers |
| `data/` | The benchmark: `benchmark/items.csv` (the index of every item), the saved pages and screenshots, `customer_assets/`, `models/phishpedia/` |
| `outputs/` | Everything the scripts write, one folder per step (not in git) |

## Run order (every script reads `data/benchmark/items.csv`)

| Step | Script | Writes (in `outputs/`) |
|---|---|---|
| 01 | `01_parse_html.py` | `01_parse_html/b_html_fields.parquet` |
| 02 | `02_ocr_read_screenshots.py` | `02_ocr_read_screenshots/` (one JSON per item) |
| 03 | `03_build_logo_gallery.py` | `03_logo_similarity/logo_gallery.npz` |
| 04 | `04_detect_and_match_logos.py` | `03_logo_similarity/logo_scores.parquet` and more |
| 05 | `05_check_logo_boxes.py` (optional) | `03_logo_similarity/check/` |
| 06 | `06_build_prompts.py` | `06_build_prompts/check2.parquet`, `check3.parquet` |
| 07 | `07_train_models.py` | `07_train_models/check2/`, `check3/`: 9 epochs each, every epoch saved with its optimizer state |
| 08 | `08_predict.py` | next to the checkpoint |
| 09 | `09_build_error_viewer.py` | `09_build_error_viewer/` |

Helpers: `logo_common.py`, `phishpedia_bit.py`, `model_common.py`, `error_viewer.html`, `setup_env.sh` (builds `.venv` with detectron2 and fetches the Phishpedia weights).
Run from the `WebPhish` folder, for example `python code/01_parse_html.py`.

## Notes
- The `outputs/` folder here holds folders from the earlier Approach B runs (`01_html_parse`, `02_ocr`, `04_prompt_format`, `05_finetune`, ...). The new scripts write to the
  names in the table above, so they do not reuse those.
- The final models (epoch 9 of check 2 and check 3) were copied into `../../inference/WebPhish/model/`.
