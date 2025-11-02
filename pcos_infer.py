"""Minimal, deployment-friendly Grad-CAM & inference helpers for Streamlit."""
from __future__ import annotations

import io
import logging
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_model

try:  # Optional preprocessing dependencies
    import cv2
    import dlib
except ImportError:  # pragma: no cover - optional path
    cv2 = None  # type: ignore
    dlib = None  # type: ignore

# WEIGHTS_PATH = "weights/epoch006_0.00005_0.29149_0.8864.pth"
WEIGHTS_PATH = r"/home/yucan/NewDisk/10Hospital/code/regressor/InceptionResNetV2_PCOS2nd/weights_clf/epoch006_0.00005_0.29149_0.8864.pth"
INPUT_SIZE = (512, 512)
LOGGER = logging.getLogger(__name__)


def _overlay_cam_on_image(rgb01: np.ndarray, cam01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    cam = np.clip(cam01, 0.0, 1.0)
    r = np.clip(1.5 * cam - 0.5, 0, 1)
    g = np.clip(1.5 - np.abs(2 * cam - 1.0), 0, 1)
    b = np.clip(1.5 * (1 - cam) - 0.5, 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)
    out = heatmap * alpha + np.clip(rgb01, 0, 1) * (1 - alpha)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _detect_primary_face(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Try to crop the most prominent face; fall back to the full frame if unavailable."""

    if cv2 is None or dlib is None:
        return None

    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if not faces:
        return None

    largest = max(faces, key=lambda f: f.width() * f.height())
    x, y, w, h = largest.left(), largest.top(), largest.width(), largest.height()
    x0, y0 = max(x, 0), max(y, 0)
    x1 = min(x0 + w, image_rgb.shape[1])
    y1 = min(y0 + h, image_rgb.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return image_rgb[y0:y1, x0:x1]


class GradCAMMinimal:
    """Small Grad-CAM helper that records activations and gradients via hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        try:
            self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
        except AttributeError:
            self._backward_handle = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inp, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        grad = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        self.gradients = grad.detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def _normalize_cam(self, cam: torch.Tensor) -> np.ndarray:
        cam = cam - cam.min()
        maxv = cam.max()
        if float(maxv) > 0:
            cam = cam / maxv
        return cam.cpu().numpy()

    def __call__(self, x: torch.Tensor, target_index: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if target_index is None:
            target_index = int(torch.argmax(logits, dim=1).item())

        loss = logits[:, target_index].sum()
        loss.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks failed to capture activations/gradients for Grad-CAM")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        return self._normalize_cam(cam)


def _decode_resize_to01(img_bytes: bytes, size_hw=INPUT_SIZE) -> np.ndarray:
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)

    face = _detect_primary_face(rgb)
    if face is None:
        if cv2 is None or dlib is None:
            LOGGER.debug("Face detection skipped (cv2/dlib unavailable); using full frame.")
        else:
            LOGGER.debug("Face detector found no faces; using full frame.")
        face = rgb

    face_image = Image.fromarray(face)
    resized = face_image.resize(size_hw, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb01).permute(2, 0, 1).unsqueeze(0).float()


_model_cache: Optional[nn.Module] = None


def load_model() -> nn.Module:
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    torch.set_num_threads(1)
    model = get_model("InceptionResNetV2")
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    _model_cache = model
    return model


@torch.no_grad()
def predict(rgb01: np.ndarray):
    model = load_model()
    x = _to_tensor(rgb01)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred = int(torch.argmax(probs, dim=1).item())
    return logits[0].tolist(), probs[0].tolist(), pred


def analyze_image_bytes(img_bytes: bytes, make_cam: bool = True, target_index: int = 1) -> Dict[str, object]:
    rgb01 = _decode_resize_to01(img_bytes, size_hw=INPUT_SIZE)
    logits, probs, pred = predict(rgb01)

    overlay_img = None
    if make_cam:
        model = load_model()
        if hasattr(model, "conv2d_7b"):
            target_layer = getattr(model, "conv2d_7b")
        elif hasattr(model, "block8"):
            target_layer = getattr(model, "block8")
        else:
            target_layer = next((m for m in reversed(list(model.modules())) if isinstance(m, nn.Conv2d)), None)

        if target_layer is not None:
            cam_helper = GradCAMMinimal(model, target_layer)
            cam_map = cam_helper(_to_tensor(rgb01), target_index=target_index)
            cam_helper.remove_hooks()
            overlay_img = Image.fromarray(_overlay_cam_on_image(rgb01, cam_map))

    return {
        "pred": pred,
        "probs": probs,
        "logits": logits,
        "overlay": overlay_img,
    }
