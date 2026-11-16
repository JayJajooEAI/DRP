# model

Two local models. Both load from these folders, with no network access at run time.

| Folder | What it is | Weights |
|---|---|---|
| `rung1_location/` | Rung 1: does the message name a place? MiniLMv2 fine-tuned as a word tagger (P(B) + P(I) at or above 0.4) | `model.safetensors`, 407 MB |
| `rung2_distilbert/` | Rung 2: threat or not. DistilBERT multilingual, fine-tuned (at least 0.6 threat, below 0.3 not a threat, 0.3 to 0.6 goes to Rung 3) | `model.safetensors`, 516 MB |

Rung 3 is the GPT-6 Luna API, so it has no weights; its prompt and answer format are `../prompt_v14.txt` and `../response_schema.json`.

## The weight files are not in git (over GitHub's 100 MB limit)

Copy the two `model.safetensors` files into place (everything else in these folders is in git).

| File | Size (bytes) | SHA-256 | S3 link |
|---|---|---|---|
| `rung1_location/model.safetensors` | 427,401,244 | `7f3736dc24e643a63db3df585fcb832dc698f904dc1f385bed2ef1a6958f8dc0` | to be added once uploaded |
| `rung2_distilbert/model.safetensors` | 541,317,368 | `3bc1b0821c5826f7d38b4440c71028c8bb3e46e6d3737bdb081b53dbf2c7f556` | to be added once uploaded |

Check a download with `sha256sum model/rung1_location/model.safetensors` and compare it with the table.

To replace a model, retrain with `../../../development/PhysicalThreatFacility/code/02_train_location_model.py` or `03_train_threat_model.py` and copy the new `model_retrained/` contents here.
