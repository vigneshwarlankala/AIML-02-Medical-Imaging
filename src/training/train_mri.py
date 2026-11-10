
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "mri"
MODEL_DIR = ROOT / "models" / "segmentation"
CHECKPOINT = MODEL_DIR / "mri_unet_best.pt"

PATCH_SIZE = 32
PATCHES_PER_CASE = 4
EPOCHS = 30
BASE_CHANNELS = 8
NUM_CLASSES = 4
SEED = 42


def load_case(image_path):
    label_path = image_path.with_name(
        image_path.name.replace("_image.pt", "_label.pt")
    )

    image = torch.load(image_path, map_location="cpu", weights_only=True)
    label = torch.load(label_path, map_location="cpu", weights_only=True)

    if tuple(image.shape) != (4, 96, 96, 96):
        raise ValueError(f"Unexpected image shape: {image.shape}")
    if tuple(label.shape) != (1, 96, 96, 96):
        raise ValueError(f"Unexpected label shape: {label.shape}")

    return image.float(), label.squeeze(0).long()


def extract_patch(image, label, rng, tumor_bias=True):
    size = PATCH_SIZE
    foreground = torch.nonzero(label > 0, as_tuple=False)

    use_tumor_center = (
        tumor_bias and len(foreground) > 0 and rng.random() < 0.7
    )

    if use_tumor_center:
        center = foreground[rng.randrange(len(foreground))]
        starts = [
            max(0, min(int(center[d]) - size // 2, 96 - size))
            for d in range(3)
        ]
    else:
        starts = [rng.randint(0, 96 - size) for _ in range(3)]

    d, h, w = starts
    image_patch = image[:, d:d+size, h:h+size, w:w+size]
    label_patch = label[d:d+size, h:h+size, w:w+size]

    return image_patch.unsqueeze(0), label_patch.unsqueeze(0)


def dice_score(logits, target):
    predictions = logits.argmax(dim=1)
    scores = []

    for class_id in range(1, NUM_CLASSES):
        pred_mask = predictions == class_id
        true_mask = target == class_id

        denominator = pred_mask.sum() + true_mask.sum()
        if denominator.item() == 0:
            continue

        intersection = (pred_mask & true_mask).sum()
        scores.append(
            (2.0 * intersection.float() / denominator.float()).item()
        )

    return sum(scores) / len(scores) if scores else 1.0


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    print("=== MRI 3D U-NET TRAINING ===", flush=True)
    print("Device: CPU")
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    image_files = sorted(DATA_DIR.glob("*_image.pt"))
    if len(image_files) < 3:
        raise RuntimeError(
            "At least 3 preprocessed MRI cases are required."
        )

    # Keep one case separate for validation.
    train_files = image_files[:-1]
    val_file = image_files[-1]

    print(f"Training cases: {len(train_files)}")
    print(f"Validation case: {val_file.name}")

    print("\nLoading training cases...", flush=True)
    train_cases = [load_case(path) for path in train_files]
    val_case = load_case(val_file)

    from models.mri_unet import UNet3D

    model = UNet3D(
        in_channels=4,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    )
    model.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        train_dice = []

        print(f"\nEpoch {epoch + 1}/{EPOCHS}", flush=True)

        total_steps = len(train_cases) * PATCHES_PER_CASE
        step = 0

        for case_index, (image, label) in enumerate(train_cases):
            for _ in range(PATCHES_PER_CASE):
                step += 1
                x, y = extract_patch(image, label, random)
                optimizer.zero_grad(set_to_none=True)

                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_dice.append(dice_score(logits.detach(), y))

                print(
                    f"  Training patch {step}/{total_steps} "
                    f"- loss: {loss.item():.4f}",
                    flush=True,
                )

                del x, y, logits, loss

        model.eval()
        val_dice_scores = []

        print("\nValidating on held-out case...", flush=True)
        with torch.no_grad():
            val_rng = random.Random(SEED)
            for patch_index in range(PATCHES_PER_CASE):
                x, y = extract_patch(
                    val_case[0], val_case[1], val_rng, tumor_bias=True
                )
                logits = model(x)
                score = dice_score(logits, y)
                val_dice_scores.append(score)

                print(
                    f"  Validation patch {patch_index + 1}/"
                    f"{PATCHES_PER_CASE} - Dice: {score:.4f}",
                    flush=True,
                )

                del x, y, logits

        mean_loss = total_loss / total_steps
        mean_train_dice = sum(train_dice) / len(train_dice)
        mean_val_dice = sum(val_dice_scores) / len(val_dice_scores)

        elapsed = time.time() - start_time

        print("\nEpoch summary")
        print(f"  Training loss: {mean_loss:.4f}")
        print(f"  Training patch Dice: {mean_train_dice:.4f}")
        print(f"  Validation patch Dice: {mean_val_dice:.4f}")
        print(f"  Time: {elapsed / 60:.1f} minutes")

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_patch_dice": best_val_dice,
                    "training_loss": mean_loss,
                    "base_channels": BASE_CHANNELS,
                    "num_classes": NUM_CLASSES,
                    "input_modalities": [
                        "FLAIR", "T1w", "t1gd", "T2w"
                    ],
                    "patch_size": PATCH_SIZE,
                },
                CHECKPOINT,
            )
            print(f"  Saved checkpoint: {CHECKPOINT}", flush=True)

    print("\n=== TRAINING FINISHED ===")
    print(f"Best validation patch Dice: {best_val_dice:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Note: Patch-level prototype evaluation only.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
