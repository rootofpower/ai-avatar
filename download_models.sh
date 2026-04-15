#!/bin/bash
# Download all required model weights for the avatar pipeline.
# Usage: bash download_models.sh

set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "=== Downloading MuseTalk V1.5 weights ==="
mkdir -p "$MODELS_DIR/musetalkV15"
if [ ! -f "$MODELS_DIR/musetalkV15/unet.pth" ]; then
    huggingface-cli download TMElyralab/MuseTalk --include "musetalkV15/*" --local-dir "$MODELS_DIR/musetalkV15_tmp"
    mv "$MODELS_DIR/musetalkV15_tmp/musetalkV15/"* "$MODELS_DIR/musetalkV15/"
    rm -rf "$MODELS_DIR/musetalkV15_tmp"
    echo "[OK] MuseTalk V1.5"
else
    echo "[SKIP] MuseTalk V1.5 already exists"
fi

echo "=== Downloading sd-vae-ft-mse ==="
mkdir -p "$MODELS_DIR/sd-vae"
if [ ! -f "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin" ]; then
    huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir "$MODELS_DIR/sd-vae"
    echo "[OK] sd-vae-ft-mse"
else
    echo "[SKIP] sd-vae already exists"
fi

echo "=== Downloading whisper-tiny ==="
mkdir -p "$MODELS_DIR/whisper"
if [ ! -f "$MODELS_DIR/whisper/pytorch_model.bin" ]; then
    huggingface-cli download openai/whisper-tiny --local-dir "$MODELS_DIR/whisper"
    echo "[OK] whisper-tiny"
else
    echo "[SKIP] whisper-tiny already exists"
fi

echo "=== Downloading DWPose ==="
mkdir -p "$MODELS_DIR/dwpose"
if [ ! -f "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth" ]; then
    huggingface-cli download yzd-v/DWPose --include "dw-ll_ucoco_384.pth" --local-dir "$MODELS_DIR/dwpose"
    echo "[OK] DWPose"
else
    echo "[SKIP] DWPose already exists"
fi

echo "=== Downloading face-parse-bisenet ==="
mkdir -p "$MODELS_DIR/face-parse-bisent"
if [ ! -f "$MODELS_DIR/face-parse-bisent/79999_iter.pth" ]; then
    echo "[INFO] face-parse-bisenet must be downloaded manually from:"
    echo "  https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view"
    echo "  Place 79999_iter.pth in $MODELS_DIR/face-parse-bisent/"
fi

echo "=== Downloading resnet18 ==="
if [ ! -f "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth" ]; then
    wget -q -O "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth" \
        "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    echo "[OK] resnet18"
else
    echo "[SKIP] resnet18 already exists"
fi

echo "=== Downloading SyncNet ==="
mkdir -p "$MODELS_DIR/syncnet"
if [ ! -f "$MODELS_DIR/syncnet/latentsync_syncnet.pt" ]; then
    huggingface-cli download ByteDance/LatentSync --include "latentsync_syncnet.pt" --local-dir "$MODELS_DIR/syncnet"
    echo "[OK] SyncNet"
else
    echo "[SKIP] SyncNet already exists"
fi

echo ""
echo "=== Model directory structure ==="
find "$MODELS_DIR" -type f -name "*.pth" -o -name "*.bin" -o -name "*.pt" -o -name "*.json" | sort
echo ""
echo "Done. Ensure all models are present before running the avatar pipeline."
