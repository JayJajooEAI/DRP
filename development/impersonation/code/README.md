# Impersonation detection: screenshot in, impersonation / not out

Run from this `code/` folder, in order (train and val first; test only once, at the end):

| # | Script | What it does | Writes (outputs/) |
|---|---|---|---|
| 01 | `01_ocr_read_screenshots.py train val test` | docTR OCR on each screenshot | work/ocr_*.json |
| 02 | `02_avatar_matching.py train val test` | avatar vs client logo / executive photos (CLIP + pixel match) | work/avatar_*.json |
| 03 | `03_check_parser_accuracy.py train val` | diagnostic: parser fields vs truth | – |
| 04 | `04_build_signals.py train val test` | parsed fields + avatar scores -> signals | work/signals_*.csv |
| 05 | `05_fit_cascade_train_val.py` | Step 2 rules + Step 3 LightGBM, fit on train, run on val, freeze | work/dev_*_stage.csv, frozen.joblib, reports/frozen_* |
| 06 | `06_luna_unsure_pages.py train val` | Step 4 Luna on the unsure train/val pages | work/luna_*.json |
| 07 | `07_final_metrics_train_val.py` | final confusion matrix on train and val | – |
| 08 | `08_run_test.py` | frozen pipeline on test (checks the freeze checksums first) | work/test_stage.csv, reports/test_* |
| 09 | `09_measure_runtime.py [--luna]` | model sizes and per-page timings | reports/runtime.json |
| 10 | `10_build_report.py` | the step-by-step HTML page | reports/impersonation_pipeline.html |

`utilities/` holds the shared modules (config, layout parser, signals, rules, LightGBM, cascade, Luna client, report sections and template).
Luna needs `OPENAI_API_KEY` in the environment, or in the file named by `OPENAI_ENV_FILE`. Set `IMPERSONATION_OUT` to write outputs elsewhere.

## Notes
- This folder reads **local files only**: the screenshots, lists and assets in `../data/`. It does not use RDS or S3.
- It does use the network in two places: the OCR and CLIP weights are downloaded on first use, and steps 06 and 08 call Luna (OpenAI).
- Step 05 freezes the pipeline with checksums of the code and the fitted model. Step 08 refuses to run on test if any of them changed, so editing a file after step 05 means running 05 again first.
- The inference service in `../inference/` is a separate, self-contained copy of these steps for one screenshot at a time.
