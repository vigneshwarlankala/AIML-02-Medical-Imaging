import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

class DenseNetGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_index):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            image_tensor = image_tensor.clone().detach().requires_grad_(True)
            output = self.model(image_tensor)
            score = output[0, class_index]
            score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam, output.detach()[0]

def load_model(checkpoint_path, device):
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier.in_features, 14)
    )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        )
    else:
        state = checkpoint

    if isinstance(state, nn.Module):
        model = state
    else:
        state = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state.items()
        }
        model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanation for a chest X-ray."
    )
    parser.add_argument("--image", required=True, help="Input X-ray image path")
    parser.add_argument(
        "--class-index", type=int, default=None,
        help="Optional target class index, from 0 to 13"
    )
    parser.add_argument(
        "--checkpoint",
        default="models/classification/densenet121_best.pt"
    )
    parser.add_argument("--output", default="outputs/gradcam")
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.is_file():
        raise FileNotFoundError(f"X-ray image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if args.class_index is not None and not 0 <= args.class_index < 14:
        raise ValueError("--class-index must be between 0 and 13")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    original = Image.open(image_path).convert("RGB")
    tensor = preprocess(original).unsqueeze(0).to(device)

    gradcam = DenseNetGradCAM(model, model.features.denseblock4)
    predicted_class = args.class_index

    if predicted_class is None:
        with torch.no_grad():
            probabilities = torch.sigmoid(model(tensor))[0]
        predicted_class = int(torch.argmax(probabilities).item())

    cam, logits = gradcam.generate(tensor, predicted_class)
    probability = torch.sigmoid(logits[predicted_class]).item()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap = Image.fromarray(
        np.uint8(np.clip(cam, 0, 1) * 255)
    ).resize(original.size, Image.Resampling.BILINEAR)

    # Red-yellow heatmap blended with the original X-ray.
    heat_array = np.asarray(heatmap).astype(np.float32) / 255.0
    colored = np.zeros(
        (heat_array.shape[0], heat_array.shape[1], 3), dtype=np.uint8
    )
    colored[..., 0] = np.uint8(np.clip(255 * np.minimum(1, 2 * heat_array), 0, 255))
    colored[..., 1] = np.uint8(np.clip(255 * np.maximum(0, 2 * heat_array - 0.5), 0, 255))

    overlay = Image.blend(
        original.convert("RGB"),
        Image.fromarray(colored),
        alpha=0.40
    )

    safe_name = image_path.stem
    overlay_path = output_dir / f"{safe_name}_gradcam.png"
    heatmap_path = output_dir / f"{safe_name}_heatmap.png"

    overlay.save(overlay_path)
    heatmap.save(heatmap_path)

    print("\n===== GRAD-CAM RESULTS =====")
    print("Device:", device)
    print("Target layer: DenseNet121 features.denseblock4")
    print("Target condition:", LABELS[predicted_class])
    print(f"Model score: {probability:.4f}")
    print("Overlay saved:", overlay_path.resolve())
    print("Heatmap saved:", heatmap_path.resolve())
    print("\nEducational visualization only. Not for clinical diagnosis.")

if __name__ == "__main__":
    main()
