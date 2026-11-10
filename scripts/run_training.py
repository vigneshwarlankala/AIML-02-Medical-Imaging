# NIH ChestX-ray14 Preprocessing + DenseNet121 Training
# Adapted for: D:/Internship_AI_ML_Projects/AIML-02-Medical-Imaging/
# data/raw/xray/images/  -- 52,120 PNG X-rays
# data/raw/xray/Data_Entry_2017_v2020.csv -- NIH labels
#
# Usage:
#   .venv\Scripts\python.exe scripts\run_training.py              (fresh)
#   .venv\Scripts\python.exe scripts\run_training.py --resume     (resume)
import sys, io
# Fix Windows cp1252 console so unicode emoji don't crash logging
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, time, logging, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ]
)
logger = logging.getLogger(__name__)

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).resolve().parent.parent  # AIML-02-Medical-Imaging/
DATA_DIR   = ROOT / "data" / "raw" / "xray"
OUT_DIR    = ROOT / "data" / "processed" / "classification"
MODEL_DIR  = ROOT / "models" / "classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# â”€â”€â”€ NIH 14 Labels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# â”€â”€â”€ Hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
IMG_SIZE    = 224
BATCH_SIZE  = 8    # Fits in ~6 GB RAM (use 16 if you get OOM)
EPOCHS      = 30
LR          = 1e-4
DROPOUT     = 0.2
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 1: Parse NIH labels
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def parse_nih_csv(csv_path: Path, images_dir: Path, max_samples: int = None):
    """
    Parse Data_Entry_2017_v2020.csv and return a list of dicts:
      { "file": str, "labels": [0/1 Ã— 14] }
    Only includes images that actually exist on disk.
    """
    logger.info(f"Reading NIH CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Total CSV rows: {len(df)}")

    records = []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing CSV"):
        fname = row["Image Index"]
        fpath = images_dir / fname
        if not fpath.exists():
            missing += 1
            continue

        # Parse multi-label string "Atelectasis|Cardiomegaly|No Finding"
        label_str = str(row["Finding Labels"])
        label_vec = [0.0] * len(NIH_LABELS)
        if label_str != "No Finding":
            for condition in label_str.split("|"):
                condition = condition.strip()
                if condition in NIH_LABELS:
                    label_vec[NIH_LABELS.index(condition)] = 1.0

        records.append({"file": fname, "labels": label_vec})

        if max_samples and len(records) >= max_samples:
            break

    logger.info(f"Found {len(records)} valid records ({missing} missing from disk)")
    return records


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 2: Dataset (lazy-load from disk each batch)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class NIHChestDataset(Dataset):
    """Lazy-loading NIH dataset. Low memory footprint."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, records, images_dir, augment=False):
        self.records = records
        self.images_dir = Path(images_dir)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.images_dir / rec["file"]

        # Load PNG and convert to RGB (some NIH images are grayscale RGBA)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]

        # Basic augmentation for training
        if self.augment:
            if np.random.rand() < 0.5:
                arr = arr[:, ::-1, :].copy()           # horizontal flip
            if np.random.rand() < 0.3:
                gamma = np.random.uniform(0.8, 1.2)
                arr = np.clip(arr ** gamma, 0, 1)      # gamma correction

        # ImageNet normalization
        arr = (arr - self.MEAN) / self.STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # [3, H, W]
        label = torch.tensor(rec["labels"], dtype=torch.float32)
        return tensor, label


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 3: Model
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def build_densenet(num_classes=14, dropout=DROPOUT):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 4: Training
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def compute_auc(y_true, y_pred):
    """Compute mean AUC-ROC across all 14 classes."""
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


def train(args):
    # â”€â”€ Parse labels â”€â”€
    csv_path = DATA_DIR / "Data_Entry_2017_v2020.csv"
    images_dir = DATA_DIR / "images"
    all_records = parse_nih_csv(csv_path, images_dir, max_samples=args.max_samples)

    # 70/15/15 split (stratify not possible for multi-label, random split is standard)
    train_r, valtest_r = train_test_split(all_records, test_size=0.30, random_state=42)
    val_r, test_r     = train_test_split(valtest_r,   test_size=0.50, random_state=42)
    logger.info(f"Split â†’ Train: {len(train_r)} | Val: {len(val_r)} | Test: {len(test_r)}")

    # Save manifests
    for split_name, records in [("train", train_r), ("val", val_r), ("test", test_r)]:
        manifest = {"split": split_name, "records": records, "label_names": NIH_LABELS}
        with open(OUT_DIR / f"{split_name}_manifest.json", "w") as f:
            json.dump(manifest, f)

    # â”€â”€ DataLoaders â”€â”€
    train_ds = NIHChestDataset(train_r, images_dir, augment=True)
    val_ds   = NIHChestDataset(val_r,   images_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

    # â”€â”€ Model â”€â”€
    model = build_densenet(num_classes=14, dropout=DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_auc = 0.0
    history = []
    start_epoch = 1

    # â”€â”€ Resume from checkpoint if requested â”€â”€
    ckpt = MODEL_DIR / "densenet121_best.pt"
    if args.resume:
        if not ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
        checkpoint = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_auc = checkpoint.get("best_auc", 0.0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch-1}, best AUC={best_auc:.4f}")
    else:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    # Restore the correct progressive training phase
    # Always freeze everything first, then enable the current phase.
    for param in model.parameters():
        param.requires_grad = False

    if start_epoch <= 5:
        phase_lr = LR
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
        logger.info("Phase 1: Classifier-only training")
    elif start_epoch <= 15:
        phase_lr = LR / 10
        for name, param in model.named_parameters():
            if "denseblock4" in name or "norm5" in name or "classifier" in name:
                param.requires_grad = True
        logger.info("Phase 2: Top-block fine-tuning")
    else:
        phase_lr = LR / 100
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Phase 3: Full fine-tuning")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=phase_lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS - start_epoch + 1, 1)
    )

    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, EPOCHS + 1):
        # Progressive unfreezing
        if epoch == 6:
            logger.info("â–¶ Phase 2: Unfreezing top blocks")
            for name, param in model.named_parameters():
                if "denseblock4" in name or "norm5" in name or "classifier" in name:
                    param.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=LR / 10
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(EPOCHS - epoch + 1, 1)
            )

        if epoch == 16:
            logger.info("â–¶ Phase 3: Full fine-tuning all layers")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR / 100)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(EPOCHS - epoch + 1, 1)
            )

        # â”€â”€ Train epoch â”€â”€
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} Train", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # â”€â”€ Validation epoch â”€â”€
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d} Val", leave=False):
                images = images.to(DEVICE, non_blocking=True)
                labels_d = labels.to(DEVICE, non_blocking=True)
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels_d)
                val_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        mean_auc = compute_auc(all_labels, all_preds)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Mean AUC: {mean_auc:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Per-class AUC log every 5 epochs
        if epoch % 5 == 0:
            for i, name in enumerate(NIH_LABELS):
                col = all_labels[:, i]
                if len(np.unique(col)) > 1:
                    class_auc = roc_auc_score(col, all_preds[:, i])
                    logger.info(f"  {name:<25}: AUC = {class_auc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss/len(train_loader),
                        "val_loss": val_loss/len(val_loader), "mean_auc": mean_auc})

        # Save best model
        if mean_auc > best_auc:
            best_auc = mean_auc
            ckpt = MODEL_DIR / "densenet121_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_auc": best_auc,
                "model_name": "densenet121",
                "num_classes": 14,
                "label_names": NIH_LABELS,
            }, ckpt)
            logger.info(f"  âœ… Saved best checkpoint (AUC={best_auc:.4f})")

    # Save final metrics
    (ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    metrics = {"best_mean_auc": best_auc, "history": history, "label_names": NIH_LABELS}
    with open(ROOT / "metrics" / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Training complete! Best Mean AUC: {best_auc:.4f}")
    logger.info(f"Model saved to: {MODEL_DIR / 'densenet121_best.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (e.g. 5000 for quick test)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from saved checkpoint")
    args = parser.parse_args()
    train(args)

