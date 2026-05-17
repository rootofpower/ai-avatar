#!/usr/bin/env python3
"""
Export MuseTalk UNet and VAE decoder to ONNX for TensorRT/CUDA acceleration.

Usage:
    python scripts/export_onnx.py --models_dir ./models --output_dir ./models/onnx

Requirements:
    pip install onnx onnxruntime-gpu
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn


class UNetWrapper(nn.Module):
    """Wraps UNet2DConditionModel for clean ONNX export (returns tensor, not dataclass)."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states).sample


class VAEDecoderWrapper(nn.Module):
    """Wraps VAE decoder for clean ONNX export (post_quant_conv + decoder)."""
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, latent):
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)


def export_unet(models_dir: str, output_dir: str):
    from diffusers import UNet2DConditionModel

    config_path = os.path.join(models_dir, "musetalkV15", "musetalk.json")
    weights_path = os.path.join(models_dir, "musetalkV15", "unet.pth")

    print(f"[Export] Loading UNet from {weights_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    unet = UNet2DConditionModel(**config)
    weights = torch.load(weights_path, map_location='cpu')
    unet.load_state_dict(weights)
    unet = unet.half().eval().cuda()

    wrapper = UNetWrapper(unet).eval().cuda()

    # Use batch=4 dummy inputs so ONNX graph captures batch-aware operations
    EXPORT_BATCH = 4
    dummy_sample = torch.randn(EXPORT_BATCH, 8, 32, 32, dtype=torch.float16, device='cuda')
    dummy_timestep = torch.zeros(EXPORT_BATCH, dtype=torch.long, device='cuda')
    dummy_hidden = torch.randn(EXPORT_BATCH, 50, 384, dtype=torch.float16, device='cuda')

    output_path = os.path.join(output_dir, "unet.onnx")
    print(f"[Export] Exporting UNet to {output_path} (traced with batch={EXPORT_BATCH})...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_sample, dummy_timestep, dummy_hidden),
            output_path,
            input_names=['sample', 'timestep', 'encoder_hidden_states'],
            output_names=['output'],
            dynamic_axes={
                'sample': {0: 'batch'},
                'timestep': {0: 'batch'},
                'encoder_hidden_states': {0: 'batch'},
                'output': {0: 'batch'}
            },
            opset_version=17,
            do_constant_folding=True,
        )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] UNet exported: {output_path} ({file_size:.1f} MB)")


def export_vae_decoder(models_dir: str, output_dir: str):
    from diffusers import AutoencoderKL

    vae_path = os.path.join(models_dir, "sd-vae")
    print(f"[Export] Loading VAE from {vae_path}...")

    vae = AutoencoderKL.from_pretrained(vae_path)
    vae = vae.half().eval().cuda()

    wrapper = VAEDecoderWrapper(vae).eval().cuda()

    EXPORT_BATCH = 4
    dummy_latent = torch.randn(EXPORT_BATCH, 4, 32, 32, dtype=torch.float16, device='cuda')
    output_path = os.path.join(output_dir, "vae_decoder.onnx")
    print(f"[Export] Exporting VAE decoder to {output_path} (traced with batch={EXPORT_BATCH})...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_latent,
            output_path,
            input_names=['latent'],
            output_names=['output'],
            dynamic_axes={
                'latent': {0: 'batch'},
                'output': {0: 'batch'}
            },
            opset_version=17,
            do_constant_folding=True,
        )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] VAE decoder exported: {output_path} ({file_size:.1f} MB)")


def verify_exports(output_dir: str):
    """Quick verification that ONNX models load and run."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Verify] onnxruntime not installed, skipping verification")
        return

    import numpy as np

    # Test with both batch=1 and batch=4 to verify dynamic batching works
    for test_batch in [1, 4]:
        print(f"\n--- Verification with batch={test_batch} ---")
        for name, input_shapes in [
            ("unet.onnx", {
                'sample': (test_batch, 8, 32, 32),
                'timestep': (test_batch,),
                'encoder_hidden_states': (test_batch, 50, 384),
            }),
            ("vae_decoder.onnx", {
                'latent': (test_batch, 4, 32, 32),
            }),
        ]:
            path = os.path.join(output_dir, name)
            if not os.path.exists(path):
                print(f"[Verify] {name} not found, skipping")
                continue

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            try:
                sess = ort.InferenceSession(path, providers=providers)
                active = sess.get_providers()[0]

                inputs = {}
                for inp_name, shape in input_shapes.items():
                    if 'timestep' in inp_name:
                        inputs[inp_name] = np.zeros(shape, dtype=np.int64)
                    else:
                        inputs[inp_name] = np.random.randn(*shape).astype(np.float16)

                import time
                t0 = time.monotonic()
                outputs = sess.run(None, inputs)
                t1 = time.monotonic()
                ms = (t1 - t0) * 1000
                print(f"[Verify] {name} batch={test_batch}: output={outputs[0].shape}, {ms:.1f}ms ({active})")
            except Exception as e:
                print(f"[Verify] {name} batch={test_batch} FAILED: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MuseTalk models to ONNX")
    parser.add_argument("--models_dir", default="./models", help="Path to models directory")
    parser.add_argument("--output_dir", default="./models/onnx", help="Output directory for ONNX files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MuseTalk ONNX Export")
    print("=" * 60)

    export_unet(args.models_dir, args.output_dir)
    export_vae_decoder(args.models_dir, args.output_dir)
    verify_exports(args.output_dir)

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("1. Install: pip install onnxruntime-gpu")
    print("2. Restart the application — it will auto-detect ONNX models")
    print("=" * 60)
