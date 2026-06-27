"""
Cleanup Utilities
-----------------
Helper functions to remove intermediate files and directories
created during pipeline execution.
"""

import os
import shutil


def remove_dir(path: str) -> None:
    """Remove a directory and all its contents if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"    [cleanup] Removed directory: {path}")


def remove_file(path: str) -> None:
    """Remove a single file if it exists."""
    if os.path.isfile(path):
        os.remove(path)
        print(f"    [cleanup] Removed file: {path}")


def cleanup_intermediates(
    frames_dir: str,
    selected_frames_dir: str,
    temp_video_path: str,
    audio_path: str,
    keep_frames: bool = False,
) -> None:
    """
    Remove all intermediate files produced during the pipeline.

    Set *keep_frames* to True to retain the raw extracted frames for debugging.
    """
    if not keep_frames:
        remove_dir(frames_dir)
        remove_dir(selected_frames_dir)
    remove_file(temp_video_path)
    remove_file(audio_path)
