
import os
import sys
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))

from ct_unet import CTUNet3D

# ---------------- CONFIGURATION ----------------
EPOCHS = 10
PATCH_SIZE = 32
PATCHES_PER_CASE = 4
BASE_CHANNELS = 8
LEARNING_RATE = 0.001
SEED = 42

DATA_DIR = ROOT / "data" / "processed" / "ct"
OUTPUT_DIR = ROOT / "models" / "segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUTPUT_DIR / "ct_unet_best.pt"
REPORT = OUTPUT_DIR / "ct_training_report.json"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

# ---------------- DATA DISCOVERY ----------------
image_files = sorted(DATA_DIR.glob("*_image.pt"))

pairs = []
for image_path in image_files:
    label_path = Path(str(image_path).replace("_image.pt", "_label.pt"))
    if label_path.exists():
        pairs.append((image_path, label_path))

if len(pairs) < 5:
    raise RuntimeError(
        f"Only {len(pairs)} matched CT cases found. "
        "At least 5 are required for this training setup."
    )

if len(pairs) < 20:
    print(
        f"WARNING: Only {len(pairs)} CT cases are available. "
        "The intended dataset size is 20."
    )

random.Random(SEED).shuffle(pairs)

split = max(1, int(len(pairs) * 0.8))
train_pairs = pairs[:split]
val_pairs = pairs[split:]

print(f"Device: {DEVICE}")
print(f"Total CT cases: {len(pairs)}")
print(f"Training cases: {len(train_pairs)}")
print(f"Validation cases: {len(val_pairs)}")
print(f"Epochs: {EPOCHS}")

# ---------------- PATCH SAMPLING ----------------
def load_case(pair):
    image = torch.load(pair[0], map_location="cpu", weights_only=False)
    label = torch.load(pair[1], map_location="cpu", weights_only=False)

    image = image.float()
    label = label.long()

    if image.shape != (1, 96, 96, 96):
        raise ValueError(f"Unexpected image shape: {image.shape}")

    if label.shape != (1, 96, 96, 96):
        raise ValueError(f"Unexpected label shape: {label.shape}")

    if not torch.all((label == 0) | (label == 1)):
        raise ValueError(f"Unexpected label values in {pair[1].name}")

    return image, label


def sample_patch(image, label, tumor_bias=True):
    size = PATCH_SIZE

    if tumor_bias and torch.rand(1).item() < 0.7:
        locations = torch.nonzero(label[0] > 0)

        if len(locations) > 0:
            center = locations[
                torch.randint(len(locations), (1,)).item()
            ].tolist()

            starts = [
                max(0, min(96 - size, c - size // 2))
                for c in center
            ]
        else:
            starts = [
                random.randint(0, 96 - size) for _ in range(3)
            ]
    else:
        starts = [
            random.randint(0, 96 - size) for _ in range(3)
        ]

    d, h, w = starts

    image_patch = image[:, d:d+size, h:h+size, w:w+size]
    label_patch = label[:, d:d+size, h:h+size, w:w+size]

    return image_patch, label_patch


# ---------------- LOSS AND METRICS ----------------
def dice_loss(logits, targets):
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    targets = targets[:, 0].float()

    intersection = (probabilities * targets).sum()
    denominator = probabilities.sum() + targets.sum()

    return 1.0 - (2.0 * intersection + 1e-6) / (
        denominator + 1e-6
    )


def binary_dice(logits, targets):
    predictions = torch.argmax(logits, dim=1) == 1
    actual = targets[:, 0] == 1

    intersection = (predictions & actual).sum().item()
    denominator = predictions.sum().item() + actual.sum().item()

    if denominator == 0:
        return 1.0

    return (2.0 * intersection) / denominator


# ---------------- MODEL ----------------
model = CTUNet3D(
    in_channels=1,
    num_classes=2,
    base_channels=BASE_CHANNELS
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()
best_dice = -1.0
history = []

# ---------------- TRAINING ----------------
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    train_dices = []

    for case_index, pair in enumerate(train_pairs, 1):
        image, label = load_case(pair)

        for patch_number in range(PATCHES_PER_CASE):
            x, y = sample_patch(image, label)

            x = x.unsqueeze(0).to(DEVICE)
            y = y.unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y[:, 0]) + dice_loss(logits, y)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_dices.append(binary_dice(logits.detach(), y))

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"Training case {case_index}/{len(train_pairs)}",
            flush=True
        )

    # ---------------- VALIDATION ----------------
    model.eval()
    val_dices = []

    with torch.no_grad():
        for pair in val_pairs:
            image, label = load_case(pair)

            for _ in range(PATCHES_PER_CASE):
                x, y = sample_patch(image, label)

                logits = model(x.unsqueeze(0).to(DEVICE))
                val_dices.append(binary_dice(logits, y.unsqueeze(0)))

    train_loss = float(np.mean(train_losses))
    train_dice = float(np.mean(train_dices))
    val_dice = float(np.mean(val_dices))

    epoch_result = {
        "epoch": epoch,
        "training_loss": train_loss,
        "training_patch_dice": train_dice,
        "validation_patch_dice": val_dice
    }
    history.append(epoch_result)

    print("\n" + "=" * 45)
    print(f"Epoch: {epoch}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training patch Dice: {train_dice:.4f}")
    print(f"Validation patch Dice: {val_dice:.4f}")
    print("=" * 45, flush=True)

    if val_dice > best_dice:
        best_dice = val_dice

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_dice,
            "train_cases": [p[0].name for p in train_pairs],
            "validation_cases": [p[0].name for p in val_pairs],
            "base_channels": BASE_CHANNELS,
            "num_classes": 2
        }, CHECKPOINT)

        print(f"Best CT checkpoint saved: {CHECKPOINT}")

    report = {
        "dataset": "Task06_Lung",
        "epochs_completed": epoch,
        "target_epochs": EPOCHS,
        "total_cases": len(pairs),
        "training_cases": len(train_pairs),
        "validation_cases": len(val_pairs),
        "patch_size": PATCH_SIZE,
        "patches_per_case": PATCHES_PER_CASE,
        "best_validation_patch_dice": best_dice,
        "history": history,
        "evaluation_note": "Patch-level prototype evaluation only."
    }

    REPORT.write_text(json.dumps(report, indent=2))

print("\n=== CT TRAINING FINISHED ===")
print(f"Best validation patch Dice: {best_dice:.4f}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Report: {REPORT}")
print(f"Elapsed minutes: {(time.time() - start_time) / 60:.1f}")
print("Note: Patch-level prototype evaluation only.")

