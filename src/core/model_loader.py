"""
Model Loader
------------
Loads and returns all AI models used in the pipeline:
  - ResNet-18 as a visual feature extractor (backbone without final FC layer)
  - CLIP (ViT-B/32) for semantic image-text similarity scoring
"""

import torch
import clip
from torchvision import models, transforms


def load_models(device: str, clip_model_name: str, resnet_input_size: tuple) -> tuple:
    """
    Load and return:
        resnet_feature_extractor, resnet_transform, clip_model, clip_preprocess

    ResNet's final classification layer is stripped so we get raw feature
    vectors rather than ImageNet logits — higher activation magnitude loosely
    correlates with visually richer frames.
    """
    print("[2/6] Loading AI models ...")

    # ResNet-18: drop the final FC layer to use as a feature backbone
    resnet_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet_backbone.eval()
    resnet_feature_extractor = torch.nn.Sequential(*list(resnet_backbone.children())[:-1])
    resnet_feature_extractor.to(device)

    resnet_transform = transforms.Compose([
        transforms.Resize(resnet_input_size),
        transforms.ToTensor(),
    ])

    # CLIP model for semantic scoring
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)

    print(f"    → Models ready  |  device: {device}")
    return resnet_feature_extractor, resnet_transform, clip_model, clip_preprocess
