"""
Batch Video Processor
---------------------
Process multiple input videos in sequence using a single config file.

Usage:
    python scripts/batch_process.py --inputs video1.mp4 video2.mp4 video3.mp4
    python scripts/batch_process.py --inputs_dir ./raw_videos/
"""

import argparse
import os
import sys
import glob
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run import load_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process multiple videos")
    parser.add_argument("--inputs", nargs="+", help="List of input video paths")
    parser.add_argument("--inputs_dir", help="Directory containing .mp4 files")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Collect input files
    video_files = []
    if args.inputs:
        video_files = args.inputs
    elif args.inputs_dir:
        video_files = glob.glob(os.path.join(args.inputs_dir, "*.mp4"))
    else:
        print("[error] Provide --inputs or --inputs_dir")
        sys.exit(1)

    if not video_files:
        print("[error] No video files found.")
        sys.exit(1)

    print(f"\nBatch processing {len(video_files)} video(s) ...\n")

    for i, video_path in enumerate(video_files, 1):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path}")

        batch_cfg = dict(cfg)
        batch_cfg["input_video"] = video_path
        batch_cfg["output_dir"] = f"output/{stem}"
        batch_cfg["frames_dir"] = f"output/{stem}/frames"
        batch_cfg["selected_frames_dir"] = f"output/{stem}/selected_frames"
        batch_cfg["temp_video_path"] = f"output/{stem}/temp_highlight.mp4"
        batch_cfg["final_video_path"] = f"output/{stem}/final_ai_reel.mp4"
        batch_cfg["audio_path"] = f"output/{stem}/audio.wav"

        try:
            run(batch_cfg)
        except Exception as exc:
            print(f"    [error] Failed on {video_path}: {exc}")
            continue

    print("\nBatch processing complete.\n")


if __name__ == "__main__":
    main()
