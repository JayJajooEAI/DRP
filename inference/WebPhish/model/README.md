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
