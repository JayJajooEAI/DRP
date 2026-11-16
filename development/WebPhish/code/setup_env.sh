#!/usr/bin/env bash
# Approach B: build .venv for the logo step and fetch Phishpedia's detector and matcher weights.
# Run from the repo root:  bash code/setup_env.sh
# The venv reuses the system torch (2.14.1, CUDA 13.0) and adds torchvision (pinned to it) and detectron2 (built from
# source; CPU ops only, the model still runs on the GPU). Weights: github.com/lindsey98/Phishpedia (CC0).
set -euo pipefail

VENV=.venv
MODELS=data/models/phishpedia
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
CONSTRAINTS=$(mktemp)
echo "torch==${TORCH_VERSION}" > "$CONSTRAINTS"

python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -c "$CONSTRAINTS" torchvision --index-url https://download.pytorch.org/whl/cu130
"$VENV/bin/pip" install -c "$CONSTRAINTS" gdown fvcore iopath pycocotools omegaconf hydra-core

SRC=$(mktemp -d)
git clone --depth 1 https://github.com/facebookresearch/detectron2.git "$SRC/detectron2"
FORCE_CUDA=0 "$VENV/bin/pip" install --no-build-isolation -c "$CONSTRAINTS" "$SRC/detectron2"
rm -rf "$SRC" "$CONSTRAINTS"

mkdir -p "$MODELS"
"$VENV/bin/gdown" 1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH -O "$MODELS/rcnn_bet365.pth"          # Faster R-CNN detector
"$VENV/bin/gdown" 1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW -O "$MODELS/faster_rcnn.yaml"         # its config
"$VENV/bin/gdown" 1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS -O "$MODELS/resnetv2_rgb_new.pth.tar" # siamese logo matcher

"$VENV/bin/python" -P -c "import torch, torchvision, detectron2; print('ok', torch.__version__, torchvision.__version__, torch.cuda.is_available())"
