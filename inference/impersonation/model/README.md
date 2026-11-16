# model

`frozen.joblib` (0.5 MB) is the only thing the service stores locally. It holds:
- the trained LightGBM model (300 trees, 102 features),
- the thresholds: rules decide at a score of 0.43 or below (not impersonation) or 0.709 or above (impersonation); the model decides at a probability of 0.02 or below or 0.98 or above,
- the list of feature columns, in the order the model expects.

It is the model frozen by `code/05_fit_cascade_train_val.py` (fitted on train only) and checked once on the test split. To replace it, rerun the batch pipeline in `../../../development/impersonation/code` and copy the new `outputs/work/frozen.joblib` here.

The OCR and CLIP weights are not stored here; they are downloaded and cached on first start.
