"""
Frame Selection Module
-----------------------
Picks one representative frame per second of source video by taking the
highest-scoring frame within each non-overlapping FPS-sized window.

Grouping by second keeps the highlight reel temporally coherent even
for variable-length videos.
"""

import os
import cv2


def select_best_frames(
    frame_scores: list[tuple[str, float]],
    source_fps: int,
    output_dir: str,
) -> list[str]:
    """
    Select the highest-scoring frame from each 1-second bucket and copy
    it to *output_dir*.

    Returns a list of selected frame paths in temporal order.
    Raises RuntimeError if no frames are selected.
    """
    print("[4/6] Selecting best frames per second ...")

    os.makedirs(output_dir, exist_ok=True)

    # Group frames into per-second buckets based on their original frame index
    second_buckets: dict[int, list[tuple[str, float]]] = {}
    for frame_index, (frame_path, score) in enumerate(frame_scores):
        second_id = frame_index // source_fps
        second_buckets.setdefault(second_id, []).append((frame_path, score))

    selected_frame_paths: list[str] = []

    for bucket_index, bucket_frames in sorted(second_buckets.items()):
        best_frame_path, _ = max(bucket_frames, key=lambda item: item[1])
        dest_path = os.path.join(output_dir, f"sel_{bucket_index:06d}.jpg")
        best_image = cv2.imread(best_frame_path)
        cv2.imwrite(dest_path, best_image)
        selected_frame_paths.append(dest_path)

    if not selected_frame_paths:
        raise RuntimeError(
            "No frames were selected — the video may be empty or corrupt."
        )

    print(
        f"    → Selected {len(selected_frame_paths)} frames "
        f"(one per source second from {len(frame_scores)} total frames)"
    )
    return selected_frame_paths
