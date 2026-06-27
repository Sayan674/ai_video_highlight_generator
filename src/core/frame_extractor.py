"""
Frame Extraction Module
-----------------------
Extracts every frame from the input video and writes them as JPEGs to disk.
"""

import os
import cv2


def extract_frames(video_path: str, output_dir: str) -> list[str]:
    """
    Extract every frame from *video_path* and write them as JPEGs to *output_dir*.

    Returns a list of file paths in display order.
    Raises FileNotFoundError if the video cannot be opened.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    frame_paths: list[str] = []
    frame_index = 0

    print(f"[1/6] Extracting frames from '{video_path}' ...")

    while True:
        success, current_frame = video_capture.read()
        if not success:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
        cv2.imwrite(frame_filename, current_frame)
        frame_paths.append(frame_filename)
        frame_index += 1

    video_capture.release()
    print(f"    → {frame_index} frames saved to '{output_dir}'")
    return frame_paths


def get_video_fps(video_path: str) -> int:
    """Return the FPS of a video file, defaulting to 30 if undetectable."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    cap.release()
    return fps
