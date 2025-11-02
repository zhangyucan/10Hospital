"""Minimal, deployment-friendly Grad-CAM & inference helpers for Streamlit."""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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

_BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE_DIR / "weights" / "epoch006_0.00005_0.29149_0.8864.pth"
# WEIGHTS_PATH = r"/home/yucan/NewDisk/10Hospital/code/regressor/InceptionResNetV2_PCOS2nd/weights_clf/epoch006_0.00005_0.29149_0.8864.pth"
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


def _decode_resize_to01(
    img_bytes: bytes, size_hw=INPUT_SIZE, return_preview: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Image.Image]]:
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
    rgb01 = np.asarray(resized, dtype=np.float32) / 255.0

    if return_preview:
        return rgb01, resized.copy()
    return rgb01


def _to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb01).permute(2, 0, 1).unsqueeze(0).float()


_model_cache: Optional[nn.Module] = None


def load_model() -> nn.Module:
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    torch.set_num_threads(1)
    model = get_model("InceptionResNetV2")
    state = torch.load(str(WEIGHTS_PATH), map_location="cpu")
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


def batch_detect_and_crop_faces(root_folder: str, output_folder: Optional[str] = None) -> Dict[str, int]:
    """
    批量检测并裁剪文件夹中所有图像的人脸。
    
    Args:
        root_folder: 根文件夹路径
        output_folder: 输出文件夹路径（可选）。如果为None，则覆盖原文件
        
    Returns:
        包含处理统计信息的字典：{'processed': int, 'faces_found': int, 'no_faces': int}
    """
    if cv2 is None or dlib is None:
        raise ImportError("需要安装 cv2 和 dlib 才能使用人脸检测功能")
    
    import os
    
    detector = dlib.get_frontal_face_detector()
    stats = {'processed': 0, 'faces_found': 0, 'no_faces': 0}
    
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        # 判断是否是文件夹
        if os.path.isdir(folder_path):
            # 如果指定了输出文件夹，创建对应的子文件夹
            if output_folder is not None:
                output_subfolder = os.path.join(output_folder, folder_name)
                os.makedirs(output_subfolder, exist_ok=True)
            
            # 遍历子文件夹
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # 判断是否是文件且为jpg格式
                if os.path.isfile(file_path) and file_path.lower().endswith(".jpg"):
                    try:
                        # 读取输入图片
                        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if img is None:
                            LOGGER.warning(f"无法读取图像: {file_path}")
                            continue
                        
                        # 将图像转换为灰度图（人脸检测器要求输入为灰度图）
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # 使用人脸检测器检测人脸
                        faces = detector(gray)
                        
                        stats['processed'] += 1
                        
                        if len(faces) == 0:
                            stats['no_faces'] += 1
                            LOGGER.info(f"未检测到人脸: {file_path}")
                            continue
                        
                        # 选择最大的人脸（如果检测到多个）
                        face = max(faces, key=lambda f: f.width() * f.height())
                        stats['faces_found'] += 1
                        
                        # 获取人脸的坐标
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        
                        # 裁剪人脸，确保坐标不越界
                        x = max(x, 0)
                        y = max(y, 0)
                        x_end = min(x + w, img.shape[1])
                        y_end = min(y + h, img.shape[0])
                        
                        face_crop = img[y:y_end, x:x_end]
                        
                        # 保存裁剪后的人脸
                        if output_folder is not None:
                            save_path = os.path.join(output_subfolder, file_name)
                        else:
                            save_path = file_path
                        
                        # 使用 imencode 和 tofile 以支持中文路径
                        is_success, buffer = cv2.imencode(".jpg", face_crop)
                        if is_success:
                            buffer.tofile(save_path)
                            LOGGER.info(f"已处理并保存: {save_path}")
                        else:
                            LOGGER.error(f"保存失败: {save_path}")
                            
                    except Exception as e:
                        LOGGER.error(f"处理图像时出错 {file_path}: {str(e)}")
                        continue
    
    LOGGER.info(f"批量处理完成 - 总处理: {stats['processed']}, 检测到人脸: {stats['faces_found']}, 未检测到人脸: {stats['no_faces']}")
    return stats


def analyze_image_bytes(img_bytes: bytes, make_cam: bool = True, target_index: int = 1) -> Dict[str, object]:
    rgb01, preview = _decode_resize_to01(img_bytes, size_hw=INPUT_SIZE, return_preview=True)
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
        "crop": preview,
    }
