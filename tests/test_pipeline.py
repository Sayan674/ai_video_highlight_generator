"""
Basic Unit Tests — AI Video Highlight Generator
------------------------------------------------
Tests core logic without requiring GPU or large model downloads.

Run with:
    pytest tests/
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import cv2

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.frame_selector import select_best_frames
from utils.cleanup import remove_file, remove_dir
from pipelines.caption_pipeline import find_caption_at_time


# ---------------------------------------------------------------------------
# Test: find_caption_at_time
# ---------------------------------------------------------------------------

SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "Hello world"},
    {"start": 2.5, "end": 5.0, "text": "This is a test"},
    {"start": 5.0, "end": 8.0, "text": "AI Video Generator"},
]


def test_caption_found_at_valid_timestamp():
    result = find_caption_at_time(1.0, SAMPLE_SEGMENTS)
    assert result == "Hello world"


def test_caption_found_at_boundary():
    result = find_caption_at_time(2.5, SAMPLE_SEGMENTS)
    assert result in ("Hello world", "This is a test")


def test_caption_not_found_outside_range():
    result = find_caption_at_time(99.0, SAMPLE_SEGMENTS)
    assert result == ""


def test_caption_empty_segments():
    result = find_caption_at_time(1.0, [])
    assert result == ""


# ---------------------------------------------------------------------------
# Test: select_best_frames
# ---------------------------------------------------------------------------

def _create_dummy_frames(tmp_dir: str, n: int) -> list[str]:
    """Write n small black JPEG frames to tmp_dir and return their paths."""
    paths = []
    for i in range(n):
        path = os.path.join(tmp_dir, f"frame_{i:06d}.jpg")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(path, dummy)
        paths.append(path)
    return paths


def test_select_best_frames_basic():
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as out_dir:

        frame_paths = _create_dummy_frames(src_dir, 30)
        # Assign ascending scores
        frame_scores = [(p, float(i)) for i, p in enumerate(frame_paths)]

        selected = select_best_frames(frame_scores, source_fps=10, output_dir=out_dir)
        # 30 frames at 10 fps = 3 seconds → 3 selected frames
        assert len(selected) == 3


def test_select_best_frames_raises_on_empty():
    with tempfile.TemporaryDirectory() as out_dir:
        with pytest.raises(RuntimeError, match="No frames were selected"):
            select_best_frames([], source_fps=30, output_dir=out_dir)


# ---------------------------------------------------------------------------
# Test: cleanup utilities
# ---------------------------------------------------------------------------

def test_remove_file_existing():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    assert os.path.exists(path)
    remove_file(path)
    assert not os.path.exists(path)


def test_remove_file_nonexistent():
    remove_file("/tmp/this_file_does_not_exist_xyz.txt")  # should not raise


def test_remove_dir_existing():
    tmp = tempfile.mkdtemp()
    assert os.path.isdir(tmp)
    remove_dir(tmp)
    assert not os.path.isdir(tmp)


def test_remove_dir_nonexistent():
    remove_dir("/tmp/this_dir_does_not_exist_xyz")  # should not raise
