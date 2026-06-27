"""
Model Pre-downloader
--------------------
Downloads all required models before the first run so the pipeline
doesn't stall mid-execution on slow connections.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --whisper medium --clip ViT-L/14
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def download_clip(model_name: str) -> None:
    print(f"[clip] Downloading CLIP model: {model_name} ...")
    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip.load(model_name, device=device)
    print(f"[clip] ✓ {model_name} ready")


def download_whisper(model_size: str) -> None:
    print(f"[whisper] Downloading Whisper model: {model_size} ...")
    import whisper
    whisper.load_model(model_size)
    print(f"[whisper] ✓ {model_size} ready")


def download_resnet() -> None:
    print("[resnet] Downloading ResNet-18 weights ...")
    from torchvision import models
    models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("[resnet] ✓ ResNet-18 ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download all model weights")
    parser.add_argument(
        "--whisper", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to download (default: base)"
    )
    parser.add_argument(
        "--clip", default="ViT-B/32",
        choices=["ViT-B/32", "ViT-L/14"],
        help="CLIP model variant to download (default: ViT-B/32)"
    )
    args = parser.parse_args()

    print("\n=== Downloading AI model weights ===\n")
    download_resnet()
    download_clip(args.clip)
    download_whisper(args.whisper)
    print("\n✅  All models downloaded and cached successfully.\n")


if __name__ == "__main__":
    main()
