"""Shared PCOS inference helpers refactored for Streamlit deployment."""
from __future__ import annotations

import functools
import io
from typing import Dict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import get_model

INPUT_SIZE = (512, 512)


def _overlay_cam_on_image(rgb01: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Blend Grad-CAM heatmap with the input image using a simple jet palette."""
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max - cam_min < 1e-6:
        normalized = np.zeros_like(cam)
    else:
        normalized = (cam - cam_min) / (cam_max - cam_min)

    # Construct an approximate JET colormap without relying on OpenCV.
    r = np.clip(1.5 * normalized - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * normalized - 1.0), 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - normalized) - 0.5, 0.0, 1.0)
    heatmap = np.stack([r, g, b], axis=-1)

    blended = heatmap * alpha + rgb01 * (1.0 - alpha)
    return (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)


@functools.lru_cache(maxsize=1)
def _load_model(model_path: str) -> torch.nn.Module:
    """Load the model once and keep it cached on CPU for subsequent calls."""
    torch.set_num_threads(1)
    model = get_model("InceptionResNetV2")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """Decode the uploaded image into a tensor and keep an RGB copy in [0, 1]."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized = image.resize(INPUT_SIZE, Image.BILINEAR)
    rgb_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0)
    return tensor, rgb_array


def analyze_image_bytes(image_bytes: bytes, model_path: str) -> Dict[str, object]:
    """Predict PCOS probability and return accompanying Grad-CAM overlay."""
    input_tensor, rgb = _preprocess_image(image_bytes)
    model = _load_model(model_path)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    pcos_prob = float(probs[0, 1])
    prediction = "PCOS" if pcos_prob >= 0.5 else "Non-PCOS"

    target_layers = [model.conv2d_7b]
    with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(1)],
        )[0]

    overlay = _overlay_cam_on_image(rgb, grayscale_cam)
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")

    return {
        "probability": pcos_prob,
        "prediction": prediction,
        "overlay_png": buffer.getvalue(),
        "logits": logits[0].tolist(),
    }
