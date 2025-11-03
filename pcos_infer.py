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

try:  # MediaPipe - preferred for cloud deployment
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore

try:  # Optional preprocessing dependencies (for local dev)
    import cv2
    import dlib
except ImportError:  # pragma: no cover - optional path
    cv2 = None  # type: ignore
    dlib = None  # type: ignore

_BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE_DIR / "weights" / "epoch006_0.00005_0.29149_0.8864.pth"
SHAPE_PREDICTOR_PATH = _BASE_DIR / "weights" / "shape_predictor_68_face_landmarks.dat"
# WEIGHTS_PATH = r"/home/yucan/NewDisk/10Hospital/code/regressor/InceptionResNetV2_PCOS2nd/weights_clf/epoch006_0.00005_0.29149_0.8864.pth"
INPUT_SIZE = (512, 512)
LOGGER = logging.getLogger(__name__)

# 缓存 shape predictor 以避免重复加载
_shape_predictor_cache: Optional[object] = None
# 缓存 MediaPipe face detection
_mediapipe_face_detection: Optional[object] = None
# 记录最近使用的人脸检测方法: 'mediapipe' | 'dlib' | 'haar' | None
_last_detection_method: Optional[str] = None


def _overlay_cam_on_image(rgb01: np.ndarray, cam01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    cam = np.clip(cam01, 0.0, 1.0)
    r = np.clip(1.5 * cam - 0.5, 0, 1)
    g = np.clip(1.5 - np.abs(2 * cam - 1.0), 0, 1)
    b = np.clip(1.5 * (1 - cam) - 0.5, 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)
    out = heatmap * alpha + np.clip(rgb01, 0, 1) * (1 - alpha)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _get_mediapipe_detector():
    """延迟加载并缓存 MediaPipe face detection"""
    global _mediapipe_face_detection
    if _mediapipe_face_detection is not None:
        return _mediapipe_face_detection
    
    if mp is None:
        return None
    
    try:
        _mediapipe_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 for full range (0-5 meters), 0 for short range
            min_detection_confidence=0.5
        )
        LOGGER.info("成功加载 MediaPipe face detection")
        return _mediapipe_face_detection
    except Exception as e:
        LOGGER.error(f"加载 MediaPipe 失败: {e}")
        return None


def _get_shape_predictor():
    """延迟加载并缓存 shape predictor"""
    global _shape_predictor_cache
    if _shape_predictor_cache is not None:
        return _shape_predictor_cache
    
    if dlib is None:
        return None
    
    if not SHAPE_PREDICTOR_PATH.exists():
        LOGGER.warning(f"Shape predictor 文件不存在: {SHAPE_PREDICTOR_PATH}")
        return None
    
    try:
        _shape_predictor_cache = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
        LOGGER.info(f"成功加载 shape predictor: {SHAPE_PREDICTOR_PATH}")
        return _shape_predictor_cache
    except Exception as e:
        LOGGER.error(f"加载 shape predictor 失败: {e}")
        return None


def _detect_primary_face(image_rgb: np.ndarray, use_alignment: bool = True) -> Optional[np.ndarray]:
    """
    Try to crop the most prominent face; fall back to the full frame if unavailable.
    Priority: MediaPipe > dlib > OpenCV Haar > None
    
    Args:
        image_rgb: RGB 图像数组
        use_alignment: 是否使用 shape predictor 进行人脸对齐（需要 shape_predictor_68_face_landmarks.dat）
    """

    global _last_detection_method

    # Try MediaPipe first (lightweight, cloud-friendly, no compilation needed)
    if mp is not None:
        try:
            detector = _get_mediapipe_detector()
            if detector is not None:
                # MediaPipe expects RGB uint8
                results = detector.process(image_rgb)
                if results.detections:
                    # Get the first/largest detection
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    h, w = image_rgb.shape[:2]
                    # Convert relative to absolute coordinates
                    x0 = max(0, int(bbox.xmin * w))
                    y0 = max(0, int(bbox.ymin * h))
                    x1 = min(w, int((bbox.xmin + bbox.width) * w))
                    y1 = min(h, int((bbox.ymin + bbox.height) * h))
                    
                    if x1 > x0 and y1 > y0:
                        LOGGER.info(f"使用 MediaPipe 检测到人脸，区域: ({x0}, {y0}) -> ({x1}, {y1})")
                        _last_detection_method = "mediapipe"
                        return image_RGB_slice(image_rgb, x0, y0, x1, y1)
                else:
                    LOGGER.info("MediaPipe 未检测到人脸，尝试其他方法")
        except Exception as e:
            LOGGER.warning(f"MediaPipe 检测出错: {e}，尝试其他方法")

    # Fallback to dlib if available (best quality but requires compilation)
    gray = None
    if cv2 is None and dlib is None and mp is None:
        LOGGER.info("人脸检测模块未安装 (mediapipe/cv2/dlib)，将使用完整图像")
        _last_detection_method = None
        return None

    # Try dlib path
    if dlib is not None and cv2 is not None:
        try:
            detector = dlib.get_frontal_face_detector()
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = detector(gray, 1)  # 上采样1次提高检测率
            if faces:
                largest = max(faces, key=lambda f: f.width() * f.height())
                # 如果启用对齐且 shape predictor 可用，使用关键点扩展边界框
                if use_alignment:
                    predictor = _get_shape_predictor()
                    if predictor is not None:
                        try:
                            shape = predictor(gray, largest)
                            points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
                            x_min, y_min = points.min(axis=0)
                            x_max, y_max = points.max(axis=0)
                            margin_x = int((x_max - x_min) * 0.1)
                            margin_y = int((y_max - y_min) * 0.1)
                            x0 = max(0, x_min - margin_x)
                            y0 = max(0, y_min - margin_y)
                            x1 = min(image_rgb.shape[1], x_max + margin_x)
                            y1 = min(image_rgb.shape[0], y_max + margin_y)
                            LOGGER.info(f"使用 shape predictor 检测到人脸关键点，区域: ({x0}, {y0}) -> ({x1}, {y1})")
                            _last_detection_method = "dlib"
                            return image_RGB_slice(image_rgb, x0, y0, x1, y1)
                        except Exception as e:
                            LOGGER.warning(f"Shape predictor 处理失败，回退到基础 dlib 框: {e}")
                # 基础 dlib 框
                x, y, w, h = largest.left(), largest.top(), largest.width(), largest.height()
                x0, y0 = max(x, 0), max(y, 0)
                x1 = min(x0 + w, image_rgb.shape[1])
                y1 = min(y0 + h, image_rgb.shape[0])
                if x1 > x0 and y1 > y0:
                    LOGGER.info(f"成功使用 dlib 检测到人脸，区域: ({x0}, {y0}) -> ({x1}, {y1})")
                    _last_detection_method = "dlib"
                    return image_RGB_slice(image_rgb, x0, y0, x1, y1)
        except Exception as e:
            LOGGER.warning(f"dlib 检测路径出错，尝试使用 OpenCV Haar 回退: {e}")

    # dlib not available or failed -> try OpenCV Haar cascades if available
    if cv2 is not None:
        try:
            if gray is None:
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            cascade_path = getattr(cv2.data, "haarcascades", None)
            if cascade_path is None:
                cascade_path = cv2.__file__  # fallback
            haar_file = None
            try:
                haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            except Exception:
                # best effort: try common locations
                possible = ["haarcascade_frontalface_default.xml"]
                for p in possible:
                    if os.path.exists(p):
                        haar_file = p
                        break

            if haar_file and os.path.exists(haar_file):
                cascade = cv2.CascadeClassifier(haar_file)
                rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(rects) > 0:
                    # choose largest
                    areas = [w * h for (x, y, w, h) in rects]
                    idx = int(np.argmax(areas))
                    x, y, w, h = rects[idx]
                    x0 = max(0, int(x))
                    y0 = max(0, int(y))
                    x1 = min(image_rgb.shape[1], int(x + w))
                    y1 = min(image_rgb.shape[0], int(y + h))
                    LOGGER.info(f"使用 OpenCV Haar 检测到人脸，区域: ({x0}, {y0}) -> ({x1}, {y1})")
                    _last_detection_method = "haar"
                    return image_RGB_slice(image_rgb, x0, y0, x1, y1)
                else:
                    LOGGER.info("OpenCV Haar 未检测到人脸，回退到使用完整图像")
                    _last_detection_method = None
                    return None
            else:
                LOGGER.info("未找到 OpenCV Haar 模型文件，无法使用 Haar 检测")
        except Exception as e:
            LOGGER.warning(f"OpenCV Haar 检测出错: {e}")

    # 最后的回退：不裁剪
    _last_detection_method = None
    return None


def image_RGB_slice(image_rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    # Helper to slice and ensure integer indices
    x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
    return image_rgb[y0i:y1i, x0i:x1i]


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
    
    LOGGER.info(f"原始图像尺寸: {rgb.shape}")

    face = _detect_primary_face(rgb)
    if face is None:
        LOGGER.info("使用完整图像进行预测")
        face = rgb
    else:
        LOGGER.info(f"使用检测到的人脸区域进行预测，尺寸: {face.shape}")

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
    # Use the unified _detect_primary_face() which already handles dlib/haar/none
    if cv2 is None and dlib is None:
        raise ImportError("需要安装 cv2 或 dlib 才能使用批量人脸检测功能")

    import os

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
                        # 读取输入图片为 BGR
                        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR) if cv2 is not None else None

                        if img is None:
                            LOGGER.warning(f"无法读取图像: {file_path}")
                            continue

                        # 转换为 RGB 并调用统一检测函数
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        stats['processed'] += 1
                        face_crop_rgb = _detect_primary_face(img_rgb)
                        if face_crop_rgb is None:
                            stats['no_faces'] += 1
                            LOGGER.info(f"未检测到人脸: {file_path}")
                            continue

                        stats['faces_found'] += 1

                        # 将裁剪结果从 RGB 转回 BGR 以保存
                        face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)

                        # 保存裁剪后的人脸
                        if output_folder is not None:
                            save_path = os.path.join(output_subfolder, file_name)
                        else:
                            save_path = file_path

                        # 使用 imencode 和 tofile 以支持中文路径
                        is_success, buffer = cv2.imencode(".jpg", face_crop_bgr)
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
        "detector": _last_detection_method,
    }
