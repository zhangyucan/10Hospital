"""极简人脸检测 + PCOS 推理模块（基于 PyTorch 生态）"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_model

# 可选人脸检测（基于 PyTorch MTCNN，无需编译）
try:
    from face_detect import crop_face_or_full
    HAVE_FACE_DETECT = True
except Exception:
    HAVE_FACE_DETECT = False

_BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE_DIR / "weights" / "epoch006_0.00005_0.29149_0.8864.pth"
INPUT_SIZE = (512, 512)
LOGGER = logging.getLogger(__name__)

# 模型缓存
_model_cache: Optional[nn.Module] = None


def _overlay_cam_on_image(rgb01: np.ndarray, cam01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """将 Grad-CAM 热力图叠加到原始图像上"""
    cam = np.clip(cam01, 0.0, 1.0)
    r = np.clip(1.5 * cam - 0.5, 0, 1)
    g = np.clip(1.5 - np.abs(2 * cam - 1.0), 0, 1)
    b = np.clip(1.5 * (1 - cam) - 0.5, 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)
    out = heatmap * alpha + np.clip(rgb01, 0, 1) * (1 - alpha)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _to_tensor(rgb01: np.ndarray) -> torch.Tensor:
    """将 RGB 数组转换为 PyTorch tensor"""
    return torch.from_numpy(rgb01).permute(2, 0, 1).unsqueeze(0).float()


def load_model() -> nn.Module:
    """加载并缓存 PCOS 分类模型"""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    
    torch.set_num_threads(1)  # Streamlit Cloud CPU 优化
    model = get_model("InceptionResNetV2")
    state = torch.load(str(WEIGHTS_PATH), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    _model_cache = model
    LOGGER.info("成功加载 PCOS 分类模型")
    return model


@torch.no_grad()
def predict(rgb01: np.ndarray):
    """
    对归一化图像进行预测
    
    Args:
        rgb01: 归一化的 RGB 数组 [0-1]
        
    Returns:
        tuple: (logits, probs, pred)
    """
    model = load_model()
    x = _to_tensor(rgb01)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred = int(torch.argmax(probs, dim=1).item())
    return logits[0].tolist(), probs[0].tolist(), pred


class GradCAMMinimal:
    """轻量级 Grad-CAM 实现"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
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


def analyze_image_bytes(img_bytes: bytes, use_face: bool = True, make_cam: bool = True, target_index: int = 1) -> Dict[str, object]:
    """
    分析图像字节流，返回预测结果和可视化
    
    Args:
        img_bytes: 图像字节流
        use_face: 是否启用人脸检测
        make_cam: 是否生成 Grad-CAM 热力图
        target_index: Grad-CAM 目标类别索引
        
    Returns:
        dict: 包含预测结果、概率、热力图等信息
    """
    # 读取图像
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 人脸检测 + 裁剪（或回退到完整图像）
    # 使用与 dlib 训练一致的预处理参数：
    # - margin=0.0: 不扩展边距（与训练时一致）
    # - force_square=True: 强制正方形（dlib 检测框扩展为正方形）
    bbox = None
    if use_face and HAVE_FACE_DETECT:
        try:
            rgb01, bbox = crop_face_or_full(
                img, 
                out_size=INPUT_SIZE, 
                margin=0.0,  # 不扩展边距，与 dlib 训练一致
                force_square=True  # 强制正方形，与 dlib 训练一致
            )
            detector_method = "MTCNN (PyTorch)" if bbox else "完整图像（未检测到人脸）"
            LOGGER.info(f"人脸检测: {detector_method}")
        except Exception as e:
            LOGGER.warning(f"人脸检测失败，使用完整图像: {e}")
            rgb01 = np.asarray(img.resize(INPUT_SIZE)).astype("float32") / 255.0
            detector_method = "完整图像（检测出错）"
    else:
        rgb01 = np.asarray(img.resize(INPUT_SIZE)).astype("float32") / 255.0
        detector_method = "完整图像（未启用检测）"
    
    # 预测
    logits, probs, pred = predict(rgb01)
    
    # Grad-CAM 可视化
    overlay_img = None
    if make_cam:
        model = load_model()
        # 找到最后一个卷积层
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
    
    # 生成预览图像
    preview_img = Image.fromarray((rgb01 * 255).astype(np.uint8))
    
    return {
        "pred": pred,
        "probs": probs,
        "logits": logits,
        "overlay": overlay_img,
        "crop": preview_img,
        "detector": detector_method,
        "bbox": bbox,
    }
