
from pathlib import Path
import gc
import json
import traceback

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "raw" / "ct" / "Task06_Lung"
IMAGES_DIR = DATA_ROOT / "imagesTr"
LABELS_DIR = DATA_ROOT / "labelsTr"

OUTPUT_DIR = ROOT / "data" / "processed" / "ct"
REPORT_PATH = OUTPUT_DIR / "preprocessing_report.json"

# ============================================================
# SETTINGS
# ============================================================
TARGET_SHAPE = (96, 96, 96)
CT_MIN = -1000.0
CT_MAX = 400.0
MAX_SCANS = 10


def load_resized_volume(path, is_label=False):
    """
    Load a NIfTI volume using nibabel's proxy and resize it.
    Process slices incrementally to reduce peak memory usage.
    """
    nii = nib.load(str(path), mmap=True)
    proxy = nii.dataobj

    original_shape = nii.shape

    if len(original_shape) != 3:
        raise ValueError(
            f"Expected a 3D volume, got shape {original_shape}"
        )

    # Resize in two stages. First reduce the first two dimensions
    # for each source slice, then resize the depth dimension.
    target_h, target_w, target_d = TARGET_SHAPE
    source_h, source_w, source_d = original_shape

    # Float32 is used for CT; labels retain integer class IDs.
    intermediate = np.empty(
        (target_h, target_w, source_d),
        dtype=np.float32,
    )

    # Process one source slice at a time.
    for z in range(source_d):
        source_slice = np.asarray(proxy[:, :, z], dtype=np.float32)

        tensor = torch.from_numpy(
            np.ascontiguousarray(source_slice)
        )[None, None, :, :]

        tensor = F.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="nearest" if is_label else "bilinear",
            align_corners=None if is_label else False,
        )

        intermediate[:, :, z] = tensor[0, 0].numpy()

        del source_slice, tensor

    del nii, proxy
    gc.collect()

    # Resize depth to the target number of slices.
    volume = torch.from_numpy(
        np.ascontiguousarray(intermediate)
    )[None, None, :, :, :]

    volume = F.interpolate(
        volume,
        size=TARGET_SHAPE,
        mode="nearest" if is_label else "trilinear",
        align_corners=None if is_label else False,
    )

    result = volume[0, 0].numpy()
    del volume, intermediate
    gc.collect()

    if is_label:
        result = np.rint(result).astype(np.int64)
    else:
        # Clip CT values and scale to [0, 1].
        result = np.clip(result, CT_MIN, CT_MAX)
        result = ((result - CT_MIN) / (CT_MAX - CT_MIN)).astype(
            np.float32
        )

    return result


def save_report(report):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def main():
    print("=== MEMORY-EFFICIENT CT PREPROCESSING ===", flush=True)

    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError(
            "Could not find imagesTr or labelsTr under the CT dataset."
        )

    images = sorted(IMAGES_DIR.glob("*.nii.gz"))
    labels = {p.name: p for p in LABELS_DIR.glob("*.nii.gz")}

    pairs = [
        (image_path, labels[image_path.name])
        for image_path in images
        if image_path.name in labels
    ]

    print(f"Matched image-label pairs: {len(pairs)}")
    print(f"Output shape: {TARGET_SHAPE}")
    print(f"Testing up to {min(MAX_SCANS, len(pairs))} scans")

    if not pairs:
        raise RuntimeError("No matching CT image-label pairs found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": "Task06_Lung",
        "target_shape": list(TARGET_SHAPE),
        "intensity_window": [CT_MIN, CT_MAX],
        "matched_pairs": len(pairs),
        "attempted_scans": min(MAX_SCANS, len(pairs)),
        "processed": [],
        "failed": [],
    }

    for index, (image_path, label_path) in enumerate(
        pairs[:MAX_SCANS], start=1
    ):
        print(
            f"\n[{index}/{min(MAX_SCANS, len(pairs))}] "
            f"{image_path.name}",
            flush=True,
        )

        try:
            image_array = load_resized_volume(
                image_path,
                is_label=False,
            )

            label_array = load_resized_volume(
                label_path,
                is_label=True,
            )

            image_tensor = torch.from_numpy(
                np.ascontiguousarray(image_array)
            ).unsqueeze(0).float()

            label_tensor = torch.from_numpy(
                np.ascontiguousarray(label_array)
            ).unsqueeze(0).long()

            expected = (1, *TARGET_SHAPE)

            if tuple(image_tensor.shape) != expected:
                raise ValueError(
                    f"Unexpected image shape: {tuple(image_tensor.shape)}"
                )

            if tuple(label_tensor.shape) != expected:
                raise ValueError(
                    f"Unexpected label shape: {tuple(label_tensor.shape)}"
                )

            if not torch.isfinite(image_tensor).all():
                raise ValueError("Image contains invalid numeric values.")

            # Remove the .nii.gz suffix cleanly.
            base_name = image_path.name.removesuffix(".nii.gz")

            image_out = OUTPUT_DIR / f"{base_name}_image.pt"
            label_out = OUTPUT_DIR / f"{base_name}_label.pt"

            torch.save(image_tensor, image_out)
            torch.save(label_tensor, label_out)

            label_values = [
                int(x) for x in torch.unique(label_tensor).tolist()
            ]

            record = {
                "file": image_path.name,
                "image_output": str(image_out.relative_to(ROOT)),
                "label_output": str(label_out.relative_to(ROOT)),
                "image_shape": list(image_tensor.shape),
                "label_shape": list(label_tensor.shape),
                "image_min": float(image_tensor.min().item()),
                "image_max": float(image_tensor.max().item()),
                "label_values": label_values,
                "status": "success",
            }

            report["processed"].append(record)

            print(f"  Image shape: {tuple(image_tensor.shape)}")
            print(f"  Label shape: {tuple(label_tensor.shape)}")
            print(
                f"  Image range: {record['image_min']:.3f} "
                f"to {record['image_max']:.3f}"
            )
            print(f"  Label values: {label_values}")
            print("  SUCCESS", flush=True)

            del image_array, label_array
            del image_tensor, label_tensor
            gc.collect()

        except Exception as exc:
            report["failed"].append({
                "file": image_path.name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

            print(f"  FAILED: {exc!r}", flush=True)
            traceback.print_exc()

        save_report(report)

    print("\n=== SUMMARY ===")
    print(f"Successful: {len(report['processed'])}")
    print(f"Failed: {len(report['failed'])}")
    print(f"Report: {REPORT_PATH}")

    if report["failed"]:
        print("Some scans failed; see the JSON report for details.")
    else:
        print("All test scans processed successfully.")


if __name__ == "__main__":
    main()
