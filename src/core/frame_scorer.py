"""
Frame Scoring Module
---------------------
Scores each extracted frame using a weighted combination of:
  - ResNet-18 feature activation  (low-level visual richness)
  - CLIP image-text similarity     (high-level semantic interest)

Using both signals avoids over-selecting blurry-but-busy frames and
under-selecting clean talking-head shots.
"""

import torch
import clip
from PIL import Image
from torchvision import transforms


def compute_resnet_score(
    pil_image: Image.Image,
    resnet_model: torch.nn.Module,
    resnet_transform: transforms.Compose,
    device: str,
) -> float:
    """
    Run the image through the ResNet backbone and return the mean activation
    of the resulting feature vector as a proxy for visual complexity.
    """
    image_tensor = resnet_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = resnet_model(image_tensor)
    return feature_vector.mean().item()


def compute_clip_score(
    frame_path: str,
    clip_model,
    clip_preprocess,
    tokenized_prompts: torch.Tensor,
    device: str,
) -> float:
    """
    Compute the maximum cosine similarity between the frame and a set of
    text prompts via CLIP.  Higher score = frame matches at least one
    'interesting' prompt more closely.
    """
    frame_image = clip_preprocess(Image.open(frame_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(frame_image)
        text_features = clip_model.encode_text(tokenized_prompts)

        # Normalise → cosine similarity becomes a simple dot product
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity_scores = image_features @ text_features.T

    return similarity_scores.max().item()


def score_frames(
    frame_paths: list[str],
    resnet_model: torch.nn.Module,
    resnet_transform: transforms.Compose,
    clip_model,
    clip_preprocess,
    device: str,
    clip_text_prompts: list[str],
    resnet_weight: float,
    clip_weight: float,
) -> list[tuple[str, float]]:
    """
    Score every extracted frame and return a list of (path, combined_score) tuples.

    Combined score = resnet_weight * resnet_score + clip_weight * clip_score
    """
    print("[3/6] Scoring frames ...")

    tokenized_prompts = clip.tokenize(clip_text_prompts).to(device)
    frame_scores: list[tuple[str, float]] = []

    for frame_path in frame_paths:
        pil_image = Image.open(frame_path).convert("RGB")

        r_score = compute_resnet_score(pil_image, resnet_model, resnet_transform, device)
        c_score = compute_clip_score(
            frame_path, clip_model, clip_preprocess, tokenized_prompts, device
        )

        combined = resnet_weight * r_score + clip_weight * c_score
        frame_scores.append((frame_path, combined))

    print(f"    → Scored {len(frame_scores)} frames")
    return frame_scores
