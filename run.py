"""
AI Video Highlight Generator
=============================
Entry point — reads config.yaml and orchestrates the full pipeline.

Usage:
    python run.py                        # uses default config.yaml
    python run.py --config my_config.yaml
    python run.py --input clip.mp4 --output my_reel.mp4
"""

import os
import sys
import argparse
import yaml
import cv2
import torch

# ---------------------------------------------------------------------------
# Ensure src/ is on the path regardless of where the script is invoked from
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core.frame_extractor import extract_frames, get_video_fps
from core.model_loader import load_models
from core.frame_scorer import score_frames
from core.frame_selector import select_best_frames
from pipelines.video_assembler import create_highlight_video
from pipelines.caption_pipeline import transcribe_audio, apply_filters_and_captions
from utils.cleanup import cleanup_intermediates


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config and return as a dict."""
    if not os.path.exists(config_path):
        print(f"[error] Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with any CLI arguments the user supplied."""
    if args.input:
        cfg["input_video"] = args.input
    if args.output:
        cfg["final_video_path"] = args.output
    return cfg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n╔══════════════════════════════════════════╗")
    print("║     AI Video Highlight Generator v1.0    ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Input   : {cfg['input_video']}")
    print(f"  Output  : {cfg['final_video_path']}")
    print(f"  Device  : {device}")
    print(f"  Whisper : {cfg['whisper_model_size']}")
    print(f"  CLIP    : {cfg['clip_model_name']}\n")

    # Make sure output directory exists
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # 1 — Extract frames
    all_frame_paths = extract_frames(cfg["input_video"], cfg["frames_dir"])
    source_fps = get_video_fps(cfg["input_video"])

    # 2 — Load models
    resnet_extractor, resnet_transform, clip_model, clip_preprocess = load_models(
        device,
        cfg["clip_model_name"],
        tuple(cfg["resnet_input_size"]),
    )

    # 3 — Score frames
    frame_scores = score_frames(
        all_frame_paths,
        resnet_extractor,
        resnet_transform,
        clip_model,
        clip_preprocess,
        device,
        cfg["clip_text_prompts"],
        cfg["resnet_score_weight"],
        cfg["clip_score_weight"],
    )

    # 4 — Select best frame per second
    selected_paths = select_best_frames(
        frame_scores, source_fps, cfg["selected_frames_dir"]
    )

    # 5 — Assemble silent highlight clip
    frame_width, frame_height = create_highlight_video(
        selected_paths, cfg["temp_video_path"], cfg["highlight_fps"]
    )

    # 6 — Transcribe audio
    caption_segments = transcribe_audio(
        cfg["input_video"], cfg["audio_path"], cfg["whisper_model_size"]
    )

    # 7 — Apply filters and caption overlay → final reel
    apply_filters_and_captions(
        temp_video_path=cfg["temp_video_path"],
        final_video_path=cfg["final_video_path"],
        frame_width=frame_width,
        frame_height=frame_height,
        caption_segments=caption_segments,
        contrast_alpha=cfg["contrast_alpha"],
        brightness_beta=cfg["brightness_beta"],
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=cfg["caption_font_scale"],
        font_thickness=cfg["caption_font_thickness"],
        text_color=tuple(cfg["caption_text_color"]),
        bg_color=tuple(cfg["caption_bg_color"]),
        bar_margin=cfg["caption_bar_margin"],
    )

    # 8 — Cleanup intermediates
    if cfg.get("cleanup_intermediates", True):
        print("\n[cleanup] Removing intermediate files ...")
        cleanup_intermediates(
            frames_dir=cfg["frames_dir"],
            selected_frames_dir=cfg["selected_frames_dir"],
            temp_video_path=cfg["temp_video_path"],
            audio_path=cfg["audio_path"],
            keep_frames=cfg.get("keep_frames_on_cleanup", False),
        )

    print(f"\n✅  Done! Your highlight reel is ready: {cfg['final_video_path']}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Video Highlight Generator"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    parser.add_argument(
        "--input", default=None,
        help="Override input video path from config"
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output video path from config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    try:
        run(cfg)
    except FileNotFoundError as exc:
        print(f"\n[error] {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n[error] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[interrupted] Pipeline cancelled by user.")
        sys.exit(0)
