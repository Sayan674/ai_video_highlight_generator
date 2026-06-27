"""
Caption Pipeline
----------------
Transcribes the original audio using OpenAI Whisper and overlays
time-aligned captions onto the highlight reel with a brightness/contrast boost.
"""

import cv2
import whisper
from moviepy.editor import VideoFileClip


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------

def transcribe_audio(
    input_video_path: str,
    audio_output_path: str,
    whisper_model_size: str,
) -> list[dict]:
    """
    Extract audio from the source video and transcribe it with Whisper.

    Returns Whisper's segment list (each segment has 'start', 'end', 'text').
    Returns an empty list if no audio track is found.
    """
    print("[6/6] Transcribing audio with Whisper ...")

    source_clip = VideoFileClip(input_video_path)

    if source_clip.audio is None:
        print("    [warn] No audio track found — captions will be skipped.")
        source_clip.close()
        return []

    source_clip.audio.write_audiofile(audio_output_path, logger=None)
    source_clip.close()

    model = whisper.load_model(whisper_model_size)
    transcription = model.transcribe(audio_output_path)
    segments = transcription["segments"]

    print(f"    → {len(segments)} caption segment(s) detected")
    return segments


# ---------------------------------------------------------------------------
# Caption lookup helper
# ---------------------------------------------------------------------------

def find_caption_at_time(timestamp: float, caption_segments: list[dict]) -> str:
    """
    Return the caption whose time window covers *timestamp*, or '' if none match.
    Linear scan is fine for the typical number of Whisper segments.
    """
    for segment in caption_segments:
        if segment["start"] <= timestamp <= segment["end"]:
            return segment["text"].strip()
    return ""


# ---------------------------------------------------------------------------
# Caption bar renderer
# ---------------------------------------------------------------------------

def draw_caption_bar(
    frame,
    caption_text: str,
    frame_width: int,
    frame_height: int,
    font: int,
    font_scale: float,
    font_thickness: int,
    text_color: tuple,
    bg_color: tuple,
    bar_margin: int,
) -> None:
    """
    Draw a filled rectangle at the bottom of *frame* and render *caption_text*
    on top of it.  Modifies the frame in-place.
    """
    bar_top = frame_height - 100
    bar_bottom = frame_height - bar_margin + 2
    text_origin = (50, frame_height - bar_margin - 2)

    cv2.rectangle(
        frame,
        (bar_margin, bar_top),
        (frame_width - bar_margin, bar_bottom),
        bg_color,
        -1,
    )
    cv2.putText(
        frame,
        caption_text,
        text_origin,
        font,
        font_scale,
        text_color,
        font_thickness,
    )


# ---------------------------------------------------------------------------
# Filter + caption overlay
# ---------------------------------------------------------------------------

def apply_filters_and_captions(
    temp_video_path: str,
    final_video_path: str,
    frame_width: int,
    frame_height: int,
    caption_segments: list[dict],
    contrast_alpha: float,
    brightness_beta: int,
    font: int,
    font_scale: float,
    font_thickness: int,
    text_color: tuple,
    bg_color: tuple,
    bar_margin: int,
) -> None:
    """
    Read the silent highlight clip frame by frame, apply a brightness/contrast
    boost, overlay time-aligned captions, and write the polished final video.
    """
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open temporary video: {temp_video_path}")

    output_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        final_video_path, fourcc, output_fps, (frame_width, frame_height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot write final video to: {final_video_path}")

    rendered = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Mild brightness and contrast enhancement
        enhanced = cv2.convertScaleAbs(frame, alpha=contrast_alpha, beta=brightness_beta)

        # Overlay caption if one matches the current timestamp
        timestamp = rendered / output_fps
        caption = find_caption_at_time(timestamp, caption_segments)
        if caption:
            draw_caption_bar(
                enhanced,
                caption,
                frame_width,
                frame_height,
                font,
                font_scale,
                font_thickness,
                text_color,
                bg_color,
                bar_margin,
            )

        writer.write(enhanced)
        rendered += 1

    cap.release()
    writer.release()
    print(f"    → Final reel saved to '{final_video_path}' ({rendered} frames rendered)")
