"""极简人脸检测 + PCOS 推理模块（基于 PyTorch 生态）"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_model

_BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE_DIR / "weights" / "epoch006_0.00005_0.29149_0.8864.pth"
INPUT_SIZE = (512, 512)
LOGGER = logging.getLogger(__name__)

# 模型缓存
_model_cache: Optional[nn.Module] = None

# 人脸检测器缓存
_face_detector = None
_detector_type = None


def _get_face_detector():
    """
    延迟初始化人脸检测器，优先使用 dlib，不可用则使用 MTCNN
    
    Returns:
        tuple: (detector, detector_type) 或 (None, None)
    """
    global _face_detector, _detector_type
    
    if _face_detector is not None:
        return _face_detector, _detector_type
    
    # 尝试使用 dlib（优先）
    try:
        import dlib
        predictor_path = _BASE_DIR / "weights" / "shape_predictor_68_face_landmarks.dat"
        if predictor_path.exists():
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(str(predictor_path))
            _face_detector = (detector, predictor)
            _detector_type = "dlib"
            LOGGER.info("✅ 使用 dlib 人脸检测（68 关键点）")
            return _face_detector, _detector_type
        else:
            LOGGER.warning(f"⚠️ dlib 模型文件不存在: {predictor_path}")
    except ImportError:
        LOGGER.info("ℹ️ dlib 不可用，尝试使用 MTCNN")
    except Exception as e:
        LOGGER.warning(f"⚠️ dlib 初始化失败: {e}")
    
    # 回退到 MTCNN
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)
        _face_detector = mtcnn
        _detector_type = "mtcnn"
        LOGGER.info("✅ 使用 MTCNN (PyTorch) 人脸检测")
        return _face_detector, _detector_type
    except ImportError:
        LOGGER.error("❌ MTCNN 不可用")
    except Exception as e:
        LOGGER.error(f"❌ MTCNN 初始化失败: {e}")
    
    return None, None


def detect_face_box(pil_img: Image.Image, conf_thres: float = 0.6) -> Optional[Tuple[int, int, int, int]]:
    """
    使用可用的检测器检测图像中的人脸（优先 dlib，未检测到则尝试 MTCNN）
    
    Args:
        pil_img: PIL Image 对象
        conf_thres: 置信度阈值（仅 MTCNN）
        
    Returns:
        (x1, y1, x2, y2) 人脸框坐标，或 None（未检测到）
    """
    detector, detector_type = _get_face_detector()
    
    if detector is None:
        LOGGER.error("❌ 没有可用的人脸检测器")
        return None
    
    result = None
    
    try:
        if detector_type == "dlib":
            # dlib 检测
            import dlib
            detector_obj, predictor = detector
            img_array = np.array(pil_img)
            gray = np.mean(img_array, axis=2).astype(np.uint8) if img_array.ndim == 3 else img_array
            faces = detector_obj(gray, 1)
            
            if len(faces) > 0:
                # 取最大的人脸
                face = max(faces, key=lambda rect: rect.width() * rect.height())
                result = (face.left(), face.top(), face.right(), face.bottom())
                LOGGER.info("✅ dlib 检测到人脸")
            else:
                LOGGER.info("⚠️ dlib 未检测到人脸，尝试使用 MTCNN...")
                # dlib 未检测到，尝试 MTCNN
                result = _try_mtcnn_detection(pil_img, conf_thres)
        
        elif detector_type == "mtcnn":
            # MTCNN 检测
            boxes, probs = detector.detect(pil_img)
            if boxes is not None and probs is not None and len(boxes) > 0:
                i = int(np.nanargmax(probs))
                if probs[i] is not None and probs[i] >= conf_thres:
                    x1, y1, x2, y2 = boxes[i]
                    result = tuple(map(int, [x1, y1, x2, y2]))
    
    except Exception as e:
        LOGGER.warning(f"{detector_type} 检测失败: {e}")
    
    return result


def _try_mtcnn_detection(pil_img: Image.Image, conf_thres: float = 0.6) -> Optional[Tuple[int, int, int, int]]:
    """
    尝试使用 MTCNN 检测人脸（当 dlib 失败时的后备方案）
    
    Args:
        pil_img: PIL Image 对象
        conf_thres: 置信度阈值
        
    Returns:
        (x1, y1, x2, y2) 人脸框坐标，或 None（未检测到）
    """
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)
        boxes, probs = mtcnn.detect(pil_img)
        
        if boxes is not None and probs is not None and len(boxes) > 0:
            i = int(np.nanargmax(probs))
            if probs[i] is not None and probs[i] >= conf_thres:
                x1, y1, x2, y2 = boxes[i]
                LOGGER.info("✅ MTCNN 检测到人脸（后备方案）")
                return tuple(map(int, [x1, y1, x2, y2]))
        
        LOGGER.info("⚠️ MTCNN 也未检测到人脸")
    except Exception as e:
        LOGGER.warning(f"MTCNN 后备检测失败: {e}")
    
    return None


def crop_face_square(pil_img: Image.Image, out_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    检测并裁剪人脸为正方形（与 dlib 训练预处理一致）
    Args:
        pil_img: 输入 PIL Image
        out_size: 输出尺寸
        
    Returns:
        tuple: (rgb01_array, box_or_None)
            - rgb01_array: 归一化后的 RGB 数组 [0-1]，如果未检测到人脸返回 None
            - box_or_None: 检测到的人脸框 (x1, y1, x2, y2) 或 None
    """
    box = detect_face_box(pil_img)
    
    if box is None:
        # 未检测到人脸，返回 None
        return None, None
    
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    # 扩展到正方形（以较大边为准，与 dlib 训练一致）
    max_side = max(w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # 计算正方形框的坐标（不扩展边距，margin=0.0）
    half_size = max_side / 2
    x1_new = int(max(0, cx - half_size))
    x2_new = int(min(pil_img.width, cx + half_size))
    y1_new = int(max(0, cy - half_size))
    y2_new = int(min(pil_img.height, cy + half_size))
    
    face = pil_img.crop((x1_new, y1_new, x2_new, y2_new))
    
    # Resize 到目标尺寸（使用 LANCZOS 高质量插值）
    face = face.resize(out_size, Image.Resampling.LANCZOS)
    arr = np.asarray(face).astype("float32") / 255.0
    
    return arr, box


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


def analyze_image_bytes(img_bytes: bytes, make_cam: bool = True, target_index: int = 1) -> Dict[str, object]:
    """
    分析图像字节流，返回预测结果和可视化
    必须检测到人脸才进行预测
    
    Args:
        img_bytes: 图像字节流
        make_cam: 是否生成 Grad-CAM 热力图
        target_index: Grad-CAM 目标类别索引
        
    Returns:
        dict: 包含预测结果、概率、热力图等信息
              如果未检测到人脸，返回错误信息
    """
    # 读取图像
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 人脸检测 + 裁剪（必须检测到人脸）
    try:
        rgb01, bbox = crop_face_square(img, out_size=INPUT_SIZE)
        
        if rgb01 is None or bbox is None:
            # 未检测到人脸，返回错误
            _, detector_type = _get_face_detector()
            LOGGER.warning("未检测到人脸")
            return {
                "error": "未检测到人脸",
                "message": "请上传包含清晰人脸的照片。建议：正面拍摄、光线充足、避免遮挡。",
                "pred": None,
                "probs": None,
                "logits": None,
                "overlay": None,
                "crop": None,
                "detector": f"{detector_type.upper() if detector_type else 'Unknown'} (未检测到)",
                "bbox": None,
            }
        
        _, detector_type = _get_face_detector()
        LOGGER.info(f"成功检测到人脸: {bbox} (使用 {detector_type})")
    except Exception as e:
        _, detector_type = _get_face_detector()
        LOGGER.error(f"人脸检测出错: {e}")
        return {
            "error": "人脸检测失败",
            "message": f"检测过程出错: {str(e)}",
            "pred": None,
            "probs": None,
            "logits": None,
            "overlay": None,
            "crop": None,
            "detector": f"{detector_type.upper() if detector_type else 'Unknown'} (出错)",
            "bbox": None,
        }
    
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
    
    # 获取检测器类型
    _, detector_type = _get_face_detector()
    detector_name = "dlib (68-point)" if detector_type == "dlib" else "MTCNN (PyTorch)" if detector_type == "mtcnn" else "Unknown"
    
    return {
        "pred": pred,
        "probs": probs,
        "logits": logits,
        "overlay": overlay_img,
        "crop": preview_img,
        "detector": detector_name,
        "bbox": bbox,
    }
