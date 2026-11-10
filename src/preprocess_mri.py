
from pathlib import Path
import gc
import json
import traceback

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "raw" / "mri" / "Task01_BrainTumour"

IMAGES_DIR = DATA_ROOT / "imagesTr"
LABELS_DIR = DATA_ROOT / "labelsTr"

OUTPUT_DIR = ROOT / "data" / "processed" / "mri"
REPORT_PATH = OUTPUT_DIR / "preprocessing_report.json"

TARGET_SHAPE = (96, 96, 96)

MODALITIES = ["FLAIR", "T1w", "t1gd", "T2w"]

NUM_CLASSES = 4

# Increased from 3 to 20 cases
MAX_CASES = 20


# ==========================================
# MRI NORMALIZATION
# ==========================================

def normalize_mri(volume):

    volume = np.asarray(volume, dtype=np.float32)

    if not np.isfinite(volume).all():
        raise ValueError("MRI volume contains invalid values.")

    mask = volume != 0

    if not np.any(mask):
        raise ValueError("MRI modality contains no nonzero voxels.")

    values = volume[mask]

    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-8:
        raise ValueError("MRI modality has near-zero intensity variation.")

    normalized = np.zeros_like(volume, dtype=np.float32)

    normalized[mask] = (values - mean) / std

    normalized = np.clip(normalized, -5.0, 5.0)

    return normalized


# ==========================================
# IMAGE RESIZING
# ==========================================

def resize_mri_volume(volume):

    tensor = (
        torch.from_numpy(np.ascontiguousarray(volume))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    resized = F.interpolate(
        tensor,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False
    )

    result = resized[0, 0].numpy().astype(np.float32)

    del tensor, resized
    gc.collect()

    return result


# ==========================================
# SEGMENTATION MASK RESIZING
# ==========================================

def resize_label_volume(volume):

    tensor = (
        torch.from_numpy(np.ascontiguousarray(volume))
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
    )

    resized = F.interpolate(
        tensor,
        size=TARGET_SHAPE,
        mode="nearest"
    )

    result = resized[0, 0].numpy().astype(np.int64)

    del tensor, resized
    gc.collect()

    return result


# ==========================================
# MAIN PREPROCESSING PIPELINE
# ==========================================

def main():

    print("=== MRI BRAIN TUMOUR PREPROCESSING ===", flush=True)

    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError(
            "MRI imagesTr or labelsTr directory not found."
        )

    images = sorted(IMAGES_DIR.glob("*.nii.gz"))

    labels = {
        path.name: path
        for path in LABELS_DIR.glob("*.nii.gz")
    }

    pairs = [
        (image, labels[image.name])
        for image in images
        if image.name in labels
    ]

    print(f"Matched image-label pairs: {len(pairs)}")
    print(f"Modalities: {MODALITIES}")
    print(f"Target shape: {TARGET_SHAPE}")
    print(f"Testing up to {min(MAX_CASES, len(pairs))} cases")

    if not pairs:
        raise RuntimeError(
            "No matching MRI image-label pairs found."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": "Task01_BrainTumour",
        "modalities": MODALITIES,
        "target_shape": list(TARGET_SHAPE),
        "normalization": "Per-modality nonzero-voxel z-score",
        "matched_pairs": len(pairs),
        "attempted_cases": min(MAX_CASES, len(pairs)),
        "processed": [],
        "failed": [],
    }

    # ======================================
    # PROCESS MRI CASES
    # ======================================

    for index, (image_path, label_path) in enumerate(
        pairs[:MAX_CASES],
        start=1
    ):

        print(
            f"\n[{index}/{min(MAX_CASES, len(pairs))}] "
            f"{image_path.name}",
            flush=True
        )

        try:

            image_nii = nib.load(
                str(image_path),
                mmap=True
            )

            label_nii = nib.load(
                str(label_path),
                mmap=True
            )

            if len(image_nii.shape) != 4:
                raise ValueError(
                    f"Expected 4D MRI image, got {image_nii.shape}"
                )

            if image_nii.shape[3] != len(MODALITIES):
                raise ValueError(
                    "Unexpected number of MRI modalities."
                )

            if len(label_nii.shape) != 3:
                raise ValueError(
                    f"Expected 3D segmentation mask, got {label_nii.shape}"
                )

            if image_nii.shape[:3] != label_nii.shape:
                raise ValueError(
                    "MRI image and segmentation dimensions do not match."
                )

            processed_modalities = []

            # Process all four MRI modalities
            for modality_index, modality_name in enumerate(MODALITIES):

                print(
                    f"  Processing {modality_name}...",
                    flush=True
                )

                source = np.asarray(
                    image_nii.dataobj[..., modality_index],
                    dtype=np.float32
                )

                normalized = normalize_mri(source)

                resized = resize_mri_volume(normalized)

                processed_modalities.append(resized)

                del source, normalized, resized
                gc.collect()

            image_tensor = torch.from_numpy(
                np.stack(processed_modalities, axis=0)
            ).float()

            del processed_modalities
            gc.collect()

            # Process segmentation mask
            print(
                "  Processing segmentation mask...",
                flush=True
            )

            label_source = np.asarray(
                label_nii.dataobj,
                dtype=np.uint8
            )

            label_values = np.unique(label_source)

            if not np.isin(
                label_values,
                list(range(NUM_CLASSES))
            ).all():

                raise ValueError(
                    f"Unexpected segmentation labels: {label_values}"
                )

            label_array = resize_label_volume(label_source)

            label_tensor = (
                torch.from_numpy(label_array)
                .unsqueeze(0)
                .long()
            )

            del label_source, label_array
            del image_nii, label_nii

            gc.collect()

            # Validate tensor dimensions
            expected_image = (4, *TARGET_SHAPE)
            expected_label = (1, *TARGET_SHAPE)

            if tuple(image_tensor.shape) != expected_image:
                raise ValueError(
                    f"Unexpected MRI tensor shape: {image_tensor.shape}"
                )

            if tuple(label_tensor.shape) != expected_label:
                raise ValueError(
                    f"Unexpected label tensor shape: {label_tensor.shape}"
                )

            if not torch.isfinite(image_tensor).all():
                raise ValueError(
                    "MRI tensor contains invalid values."
                )

            output_labels = [
                int(value)
                for value in torch.unique(label_tensor).tolist()
            ]

            # Save output tensors
            base_name = image_path.name.removesuffix(".nii.gz")

            image_out = OUTPUT_DIR / f"{base_name}_image.pt"
            label_out = OUTPUT_DIR / f"{base_name}_label.pt"

            torch.save(image_tensor, image_out)
            torch.save(label_tensor, label_out)

            record = {
                "file": image_path.name,
                "image_output": str(image_out.relative_to(ROOT)),
                "label_output": str(label_out.relative_to(ROOT)),
                "image_shape": list(image_tensor.shape),
                "label_shape": list(label_tensor.shape),
                "image_min": float(image_tensor.min().item()),
                "image_max": float(image_tensor.max().item()),
                "label_values": output_labels,
                "status": "success",
            }

            report["processed"].append(record)

            print(
                f"  Image shape: {tuple(image_tensor.shape)}"
            )

            print(
                f"  Label shape: {tuple(label_tensor.shape)}"
            )

            print(
                f"  Image range: "
                f"{record['image_min']:.3f} to "
                f"{record['image_max']:.3f}"
            )

            print(
                f"  Label values: {output_labels}"
            )

            print("  SUCCESS", flush=True)

            del image_tensor, label_tensor
            gc.collect()

        except Exception as exc:

            report["failed"].append({
                "file": image_path.name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

            print(
                f"  FAILED: {exc!r}",
                flush=True
            )

            traceback.print_exc()

        # Save report after every case
        REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        REPORT_PATH.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8"
        )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print("\n=== MRI PREPROCESSING SUMMARY ===")

    print(
        f"Successful: {len(report['processed'])}"
    )

    print(
        f"Failed: {len(report['failed'])}"
    )

    print(
        f"Report: {REPORT_PATH}"
    )

    if report["failed"]:
        print(
            "Some cases failed. Check the JSON report."
        )
    else:
        print(
            "All MRI test cases processed successfully."
        )


if __name__ == "__main__":
    main()
