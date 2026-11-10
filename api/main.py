from contextlib import asynccontextmanager
from pathlib import Path
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.mri_unet import UNet3D
from src.models.ct_unet import CTUNet3D

DEVICE = torch.device("cpu")

X_RAY_CHECKPOINT = ROOT / "models/classification/densenet121_best.pt"
MRI_CHECKPOINT = ROOT / "models/segmentation/mri_unet_best.pt"
CT_CHECKPOINT = ROOT / "models/segmentation/ct_unet_best.pt"

MRI_DIR = ROOT / "data/processed/mri"
CT_DIR = ROOT / "data/processed/ct"

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

MRI_CLASSES = {
    0: "Background",
    1: "Edema",
    2: "Non-enhancing tumour",
    3: "Enhancing tumour"
}

CT_CLASSES = {
    0: "Background",
    1: "Lung lesion"
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

xray_model = None
mri_model = None
ct_model = None
checkpoint_info = {}


def load_models():
    global xray_model, mri_model, ct_model, checkpoint_info

    for path in [X_RAY_CHECKPOINT, MRI_CHECKPOINT, CT_CHECKPOINT]:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    # X-ray classification model
    xray_ckpt = torch.load(
        X_RAY_CHECKPOINT, map_location=DEVICE, weights_only=True
    )

    net = densenet121(weights=None)
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(net.classifier.in_features, 14)
    )
    net.load_state_dict(xray_ckpt["model_state_dict"])
    net.to(DEVICE).eval()
    xray_model = net

    # MRI segmentation model
    # weights_only=False is used for trusted local training checkpoints.
    mri_ckpt = torch.load(
        MRI_CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    mri_net = UNet3D(
        in_channels=4, num_classes=4, base_channels=8
    )
    mri_net.load_state_dict(mri_ckpt["model_state_dict"])
    mri_net.to(DEVICE).eval()
    mri_model = mri_net

    # CT segmentation model
    ct_ckpt = torch.load(
        CT_CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    ct_net = CTUNet3D(
        in_channels=1, num_classes=2, base_channels=8
    )
    ct_net.load_state_dict(ct_ckpt["model_state_dict"])
    ct_net.to(DEVICE).eval()
    ct_model = ct_net

    checkpoint_info.update({
        "xray": {
            "model": xray_ckpt.get("model_name", "DenseNet121"),
            "epoch": xray_ckpt.get("epoch"),
            "best_auc": xray_ckpt.get("best_auc"),
            "labels": xray_ckpt.get("label_names", LABELS)
        },
        "mri": {
            "model": "3D U-Net",
            "epoch": mri_ckpt.get("epoch"),
            "validation_patch_dice": mri_ckpt.get(
                "validation_patch_dice"
            ),
            "classes": MRI_CLASSES
        },
        "ct": {
            "model": "CT 3D U-Net",
            "epoch": ct_ckpt.get("epoch"),
            "validation_patch_dice": ct_ckpt.get("best_val_dice"),
            "classes": CT_CLASSES
        }
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="AI-Assisted Medical Imaging API",
    description=(
        "Educational research prototype for X-ray classification "
        "and MRI/CT segmentation. Not for clinical diagnosis."
    ),
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "project": "AI-Assisted Medical Imaging System",
        "status": "API running",
        "documentation": "/docs",
        "endpoints": [
            "/predict",
            "/predict/mri/{case_id}",
            "/predict/ct/{case_id}"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if all(
            m is not None for m in [xray_model, mri_model, ct_model]
        ) else "model_not_loaded",
        "xray_loaded": xray_model is not None,
        "mri_loaded": mri_model is not None,
        "ct_loaded": ct_model is not None,
        "device": str(DEVICE)
    }


@app.get("/model-info")
def model_info():
    if not checkpoint_info:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    return checkpoint_info


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if xray_model is None:
        raise HTTPException(status_code=503, detail="X-ray model not loaded.")

    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail="Upload a PNG or JPEG chest X-ray image."
        )

    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded file is empty.")

        image = Image.open(__import__("io").BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - MEAN) / STD
        tensor = torch.from_numpy(
            array.transpose(2, 0, 1).copy()
        ).unsqueeze(0)

        with torch.inference_mode():
            probabilities = torch.sigmoid(
                xray_model(tensor.to(DEVICE))
            )[0].cpu().numpy()

        results = [
            {
                "condition": name,
                "probability": round(float(score), 6),
                "percentage": round(float(score) * 100, 2)
            }
            for name, score in zip(
                checkpoint_info["xray"]["labels"], probabilities
            )
        ]
        results.sort(key=lambda item: item["probability"], reverse=True)

        return {
            "filename": file.filename,
            "model": "DenseNet121",
            "predictions": results,
            "disclaimer": "Educational research only. Not a medical diagnosis."
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not process image: {exc}"
        )
    finally:
        await file.close()


def load_processed_tensor(directory: Path, case_id: str, suffix: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise HTTPException(status_code=400, detail="Invalid case ID.")

    path = directory / f"{case_id}_{suffix}.pt"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Processed sample not found: {case_id}"
        )

    try:
        tensor = torch.load(
            path, map_location="cpu", weights_only=True
        )
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Stored data is not a tensor.")
        return tensor.float()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not load processed tensor: {exc}"
        )


def segmentation_summary(prediction, classes):
    prediction = prediction.squeeze(0).cpu()
    total = prediction.numel()

    counts = {}
    for class_id, class_name in classes.items():
        count = int((prediction == class_id).sum().item())
        counts[class_name] = {
            "voxels": count,
            "percentage": round(count / total * 100, 4)
        }

    return {
        "volume_shape": list(prediction.shape),
        "total_voxels": total,
        "class_distribution": counts
    }


@app.post("/predict/mri/{case_id}")
def predict_mri(case_id: str):
    if mri_model is None:
        raise HTTPException(status_code=503, detail="MRI model not loaded.")

    image = load_processed_tensor(MRI_DIR, case_id, "image")

    if tuple(image.shape) != (4, 96, 96, 96):
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected MRI tensor shape: {list(image.shape)}"
        )

    with torch.inference_mode():
        logits = mri_model(image.unsqueeze(0).to(DEVICE))
        prediction = torch.argmax(logits, dim=1, keepdim=True)

    result = segmentation_summary(prediction, MRI_CLASSES)

    return {
        "case_id": case_id,
        "model": "MRI 3D U-Net",
        "checkpoint_epoch": checkpoint_info["mri"]["epoch"],
        "validation_patch_dice": checkpoint_info["mri"][
            "validation_patch_dice"
        ],
        "segmentation": result,
        "disclaimer": "Educational research only. Not a medical diagnosis."
    }


@app.post("/predict/ct/{case_id}")
def predict_ct(case_id: str):
    if ct_model is None:
        raise HTTPException(status_code=503, detail="CT model not loaded.")

    image = load_processed_tensor(CT_DIR, case_id, "image")

    if tuple(image.shape) != (1, 96, 96, 96):
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected CT tensor shape: {list(image.shape)}"
        )

    with torch.inference_mode():
        logits = ct_model(image.unsqueeze(0).to(DEVICE))
        prediction = torch.argmax(logits, dim=1, keepdim=True)

    result = segmentation_summary(prediction, CT_CLASSES)

    return {
        "case_id": case_id,
        "model": "CT 3D U-Net",
        "checkpoint_epoch": checkpoint_info["ct"]["epoch"],
        "validation_patch_dice": checkpoint_info["ct"][
            "validation_patch_dice"
        ],
        "segmentation": result,
        "disclaimer": "Educational research only. Not a medical diagnosis."
    }


# ============================================================
# DOCTOR REVIEW AND AUDIT LOG
# ============================================================

import sqlite3
from datetime import datetime, timezone

AUDIT_DB = Path(__file__).resolve().parents[1] / "data" / "audit_log.db"
AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)


def initialize_audit_database():
    with sqlite3.connect(AUDIT_DB) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS doctor_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                comments TEXT,
                reviewed_at TEXT NOT NULL
            )
        """)
        connection.commit()


initialize_audit_database()


@app.post("/review")
async def submit_doctor_review(payload: dict):
    case_id = str(payload.get("case_id", "")).strip()
    modality = str(payload.get("modality", "")).strip().lower()
    decision = str(payload.get("decision", "")).strip().lower()
    reviewer = str(payload.get("reviewer", "")).strip()
    comments = str(payload.get("comments", "")).strip()

    if not case_id or not reviewer:
        raise HTTPException(
            status_code=400,
            detail="case_id and reviewer are required."
        )

    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise HTTPException(status_code=400, detail="Invalid case ID.")

    if modality not in ["xray", "mri", "ct"]:
        raise HTTPException(
            status_code=400,
            detail="Modality must be xray, mri, or ct."
        )

    if decision not in ["approved", "rejected"]:
        raise HTTPException(
            status_code=400,
            detail="Decision must be approved or rejected."
        )

    if len(reviewer) > 120 or len(comments) > 2000:
        raise HTTPException(status_code=400, detail="Input is too long.")

    reviewed_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(AUDIT_DB) as connection:
        cursor = connection.execute("""
            INSERT INTO doctor_reviews
            (case_id, modality, decision, reviewer, comments, reviewed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            case_id, modality, decision, reviewer, comments, reviewed_at
        ))
        review_id = cursor.lastrowid
        connection.commit()

    return {
        "status": "recorded",
        "review_id": review_id,
        "case_id": case_id,
        "modality": modality,
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "message": "Review decision saved in the audit database."
    }


@app.get("/audit")
def get_audit_log(limit: int = 100):
    if not 1 <= limit <= 500:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 500."
        )

    with sqlite3.connect(AUDIT_DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT id, case_id, modality, decision, reviewer,
                   comments, reviewed_at
            FROM doctor_reviews
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return {
        "total_returned": len(rows),
        "reviews": [dict(row) for row in rows]
    }

