# model

| Folder | What it is | Size |
|---|---|---|
| `check2/` | Fine-tuned mmBERT, epoch 9: is the page branded as the client? Labels `False`, `True` | 1.2 GB |
| `check3/` | Fine-tuned mmBERT, epoch 9: data capture. Labels `3.1`, `3.2`, `none` | 1.2 GB |
| `phishpedia/` | Phishpedia's logo detector (Faster R-CNN) and logo matcher (siamese BiT-M-R50x1), CC0 | 0.5 GB |
| `url_model/` | Suspicious-URL model (LightGBM, host-name features) | 1.4 MB |

Each `check*/` folder has `model/` (weights and tokenizer), `inference_config.json` (label order, maximum length of 8,192 tokens, field order
and the order in which fields are shortened) and `metrics.json` from training. The `prompt_data` path inside `inference_config.json` is from
training and is not used here.

To replace a model, train again with `../../../development/WebPhish/code/07_train_models.py` and copy the chosen epoch's `model/` and `inference_config.json` here.
The two mmBERT models load from these folders without network access (checked with `HF_HUB_OFFLINE=1`). The only weights fetched at run time are
docTR's text detector and recognizer, downloaded once on first start and cached in `~/.cache/doctr`.

The URL model's file was re-saved so that its helper class is found under this package's name (`WebPhish.url_features`); its predictions are unchanged.

## The four large weight files are not in git (over GitHub's 100 MB limit)

Copy these into place (everything else in this folder is in git).

| File | Size (bytes) | SHA-256 | S3 link |
|---|---|---|---|
| `check2/model/model.safetensors` | 1,230,141,424 | `a5dfc5cfefff4124e9a58a5a407109828c975f1a81e5efeb17b3ed22235a6e73` | to be added once uploaded |
| `check3/model/model.safetensors` | 1,230,144,500 | `44f55a0013604dbdfe64e92ace547d5ee51152fc30afb3d17eaceec47a118328` | to be added once uploaded |
| `phishpedia/rcnn_bet365.pth` | 330,022,096 | `2cf2c3859c519e128e909e18afd8edb673ca306a9ddbb6d52fc3eed5041901c4` | to be added once uploaded |
| `phishpedia/resnetv2_rgb_new.pth.tar` | 192,603,493 | `358040750456f62175e33df2d630286769f2e4ca9d08d356dbafa1ed782010ec` | to be added once uploaded |

Check a download with `sha256sum <file>` and compare it with the table. The two Phishpedia files can also be fetched with `development/WebPhish/code/setup_env.sh`.
