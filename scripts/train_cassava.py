from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cassava_detector import CLASS_NAMES, IMAGE_SIZE, build_model, extract_embedding


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CassavaImages(Dataset):
    def __init__(self, root: Path, rows: list[tuple[str, int]], transform: transforms.Compose):
        self.root = root
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_name, label = self.rows[index]
        with Image.open(self.root / "train_images" / image_name) as image:
            return self.transform(image.convert("RGB")), label


def stratified_split(rows: list[tuple[str, int]], validation_fraction: float, seed: int):
    grouped: dict[int, list[tuple[str, int]]] = {i: [] for i in range(len(CLASS_NAMES))}
    for row in rows:
        grouped[row[1]].append(row)
    rng = random.Random(seed)
    training, validation = [], []
    for group in grouped.values():
        rng.shuffle(group)
        count = max(1, round(len(group) * validation_fraction))
        validation.extend(group[:count])
        training.extend(group[count:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


@torch.inference_mode()
def encode_dataset(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    embeddings, labels = [], []
    model.eval()
    for images, targets in loader:
        vectors = extract_embedding(model, images.to(device)).cpu()
        embeddings.append(vectors)
        labels.append(targets)
    return torch.cat(embeddings), torch.cat(labels)


def nearest_reference_similarity(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    query = F.normalize(query.float(), dim=1)
    bank = F.normalize(bank.float(), dim=1)
    scores = []
    for start in range(0, len(query), 128):
        scores.append((query[start:start + 128] @ bank.T).max(dim=1).values)
    return torch.cat(scores)


def main():
    parser = argparse.ArgumentParser(description="Train a cassava leaf classifier from the Kaggle images.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "cassava")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "cassava_mobilenet_v3_small.pt")
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    csv_path = args.data_dir / "train.csv"
    image_dir = args.data_dir / "train_images"
    if not csv_path.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Need train.csv and train_images/ under {args.data_dir}. Extract those from the Kaggle ZIP first.")
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = [(row["image_id"], int(row["label"])) for row in csv.DictReader(source)]
    rows = [row for row in rows if (image_dir / row[0]).is_file()]
    training_rows, validation_rows = stratified_split(rows, args.validation_fraction, args.seed)

    # Start with ImageNet visual features, then learn the five cassava-specific classes.
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    backbone = models.mobilenet_v3_small(weights=weights)
    model = build_model()
    model.features.load_state_dict(backbone.features.state_dict())
    model.avgpool.load_state_dict(backbone.avgpool.state_dict())
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.to(device)

    deterministic = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_loader = DataLoader(CassavaImages(args.data_dir, training_rows, deterministic), batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(CassavaImages(args.data_dir, validation_rows, deterministic), batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset: {len(rows):,} images | train {len(training_rows):,} | validation {len(validation_rows):,}", flush=True)
    print("Extracting frozen MobileNet features on CPU…", flush=True)
    train_features, train_labels = encode_dataset(model, train_loader, device)
    valid_features, valid_labels = encode_dataset(model, valid_loader, device)

    model.classifier.train()
    counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).float()
    class_weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    class_weights = class_weights / class_weights.mean()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.003, weight_decay=0.02)
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_features))
        model.classifier.train()
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            logits = model.classifier(train_features[indices])
            loss = F.cross_entropy(logits, train_labels[indices], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            val_logits = torch.cat([model.classifier(valid_features[i:i + args.batch_size]) for i in range(0, len(valid_features), args.batch_size)])
            val_predictions = val_logits.argmax(1)
            accuracy = (val_predictions == valid_labels).float().mean().item()
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | loss {total_loss / len(train_features):.4f} | validation accuracy {accuracy:.4f}", flush=True)

    model.eval()
    with torch.inference_mode():
        val_logits = model.classifier(valid_features)
        val_predictions = val_logits.argmax(1)
        accuracy = (val_predictions == valid_labels).float().mean().item()
        per_class = {}
        for index, name in enumerate(CLASS_NAMES):
            mask = valid_labels.eq(index)
            per_class[name] = round((val_predictions[mask] == valid_labels[mask]).float().mean().item(), 4)

    # A similarity guard helps reject unrelated uploads; its cutoff retains 99% of
    # held-out cassava leaves. It is a safeguard, not a guarantee for lookalike leaves.
    normalized_train = F.normalize(train_features.float(), dim=1)
    valid_similarity = nearest_reference_similarity(valid_features, normalized_train)
    ood_threshold = float(torch.quantile(valid_similarity, 0.01).item())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "embedding_bank": normalized_train.to(torch.float16),
        "ood_threshold": ood_threshold,
        "metrics": {"validation_accuracy": round(accuracy, 4), "per_class_accuracy": per_class,
                    "validation_count": len(validation_rows), "training_count": len(training_rows),
                    "ood_threshold_validation_p01": round(ood_threshold, 4), "seed": args.seed},
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "source": "Kaggle Cassava Leaf Disease Classification; Makerere University AI Lab",
    }, args.output)
    (args.output.with_suffix(".metrics.json")).write_text(json.dumps({"validation_accuracy": round(accuracy, 4), "per_class_accuracy": per_class,
        "training_count": len(training_rows), "validation_count": len(validation_rows), "ood_threshold": round(ood_threshold, 4)}, indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}", flush=True)
    print(json.dumps({"validation_accuracy": round(accuracy, 4), "per_class_accuracy": per_class, "ood_threshold": round(ood_threshold, 4)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
