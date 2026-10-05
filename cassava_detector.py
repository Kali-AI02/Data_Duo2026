from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy cassava leaf",
]
IMAGE_SIZE = 192
MODEL_FILE = Path(__file__).parent / "models" / "cassava_mobilenet_v3_small.pt"


def build_model() -> torch.nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[0].in_features, 512),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.25),
        torch.nn.Linear(512, len(CLASS_NAMES)),
    )
    return model


def make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def prepare_image(image: Image.Image) -> torch.Tensor:
    return make_transform()(image.convert("RGB")).unsqueeze(0)


def extract_embedding(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    features = model.features(batch)
    return model.avgpool(features).flatten(1)


def load_model_bundle(path: Path = MODEL_FILE) -> tuple[torch.nn.Module, torch.Tensor, float, dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    bank = torch.nn.functional.normalize(bundle["embedding_bank"].float(), dim=1)
    return model, bank, float(bundle["ood_threshold"]), bundle.get("metrics", {})
