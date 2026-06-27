"""
Video Assembly Pipeline
-----------------------
Stitches selected frames into a silent highlight clip, then applies
brightness/contrast enhancement and overlays time-aligned captions.
"""

import cv2


def create_highlight_video(
    selected_frame_paths: list[str],
    output_path: str,
    output_fps: int,
) -> tuple[int, int]:
    """
    Stitch selected frames into a silent video at *output_fps*.

    Returns (width, height) for use by downstream steps.
    Raises RuntimeError if the first frame is unreadable or the writer fails.
    """
    print("[5/6] Assembling highlight clip ...")

    first_frame = cv2.imread(selected_frame_paths[0])
    if first_frame is None:
        raise RuntimeError(f"Could not read frame: {selected_frame_paths[0]}")

    frame_height, frame_width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter could not open output path: {output_path}")

    skipped = 0
    for frame_path in selected_frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"    [warn] Skipping unreadable frame: {frame_path}")
            skipped += 1
            continue
        writer.write(frame)

    writer.release()

    if skipped:
        print(f"    [warn] {skipped} frame(s) were skipped due to read errors.")

    print(f"    → Silent highlight clip saved to '{output_path}'")
    return frame_width, frame_height
