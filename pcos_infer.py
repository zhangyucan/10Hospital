"""Shared PCOS inference utilities for notebooks, Streamlit and other frontends."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Optional

import cv2
import dlib
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from model import get_model


def _detect_primary_face(detector: dlib.fhog_object_detector, image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest detected face crop; fall back to the full image if none."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if not faces:
        return None

    largest = max(faces, key=lambda f: f.width() * f.height())
    x, y, w, h = largest.left(), largest.top(), largest.width(), largest.height()
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x0 + w, image_rgb.shape[1]), min(y0 + h, image_rgb.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return image_rgb[y0:y1, x0:x1]


@functools.lru_cache(maxsize=1)
def _load_model(model_path: str, device_str: str) -> torch.nn.Module:
    """Load and cache the classification model on the desired device."""
    model = get_model("InceptionResNetV2")
    state = torch.load(model_path, map_location=device_str)
    model.load_state_dict(state)
    model.to(device_str)
    model.eval()
    return model


def analyze_image_bytes(
    image_bytes: bytes,
    model_path: str,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Predict PCOS probability from raw image bytes and return Grad-CAM overlay."""
    raw_array = np.frombuffer(image_bytes, dtype=np.uint8)
    raw = cv2.imdecode(raw_array, cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError("Unable to decode the uploaded image.")

    image_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    face = _detect_primary_face(detector, image_rgb) or image_rgb
    face = face.astype(np.float32) / 255.0
    face_resized = cv2.resize(face, (512, 512), interpolation=cv2.INTER_LINEAR)

    preprocess = transforms.Compose([transforms.ToTensor()])
    tensor = preprocess(face_resized).unsqueeze(0)

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_str)
    model = _load_model(model_path, device_str)
    input_tensor = tensor.to(torch_device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    pcos_prob = float(probs[0, 1].cpu())
    prediction = "PCOS" if pcos_prob >= 0.5 else "Non-PCOS"

    target_layers = [model.conv2d_7b]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0]

    overlay = show_cam_on_image(face_resized, grayscale_cam, use_rgb=True)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
    success, overlay_buf = cv2.imencode(".png", overlay_bgr)
    if not success:
        raise RuntimeError("Failed to encode Grad-CAM overlay as PNG.")

    return {
        "probability": pcos_prob,
        "prediction": prediction,
        "overlay_png": overlay_buf.tobytes(),
        "face": face_resized,
    }
