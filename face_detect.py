# face_detect.py
"""
基于 facenet-pytorch 的轻量级人脸检测模块
使用 MTCNN 模型，纯 PyTorch 实现，适合 Streamlit Cloud CPU 环境
"""
from PIL import Image
import numpy as np

_MTCNN = None

def _get_mtcnn():
    """延迟初始化 MTCNN，避免影响其他功能"""
    global _MTCNN
    if _MTCNN is None:
        from facenet_pytorch import MTCNN  # 延迟导入，若未安装也不影响其它功能
        _MTCNN = MTCNN(keep_all=False, device='cpu', post_process=False)
    return _MTCNN

def detect_first_face_box(pil_img: Image.Image, conf_thres=0.6):
    """
    检测图像中置信度最高的人脸
    
    Args:
        pil_img: PIL Image 对象
        conf_thres: 置信度阈值
        
    Returns:
        (x1, y1, x2, y2) 人脸框坐标，或 None（未检测到）
    """
    mtcnn = _get_mtcnn()
    boxes, probs = mtcnn.detect(pil_img)
    if boxes is None or probs is None or len(boxes) == 0:
        return None
    i = int(np.nanargmax(probs))
    if probs[i] is None or probs[i] < conf_thres:
        return None
    x1, y1, x2, y2 = boxes[i]
    return tuple(map(int, [x1, y1, x2, y2]))

def crop_face_or_full(pil_img: Image.Image, out_size=(512, 512), margin=0.15):
    """
    优先裁剪人脸；失败则回退到使用整图
    
    Args:
        pil_img: 输入 PIL Image
        out_size: 输出尺寸
        margin: 人脸框扩展边距（相对于框宽高的比例）
        
    Returns:
        tuple: (rgb01_array, box_or_None)
            - rgb01_array: 归一化后的 RGB 数组 [0-1]
            - box_or_None: 检测到的人脸框 (x1, y1, x2, y2) 或 None
    """
    box = None
    try:
        box = detect_first_face_box(pil_img)
    except Exception as e:
        print(f"人脸检测失败，使用完整图像: {e}")
        box = None
    
    if box:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        m = margin
        # 扩展边距
        x1 = int(max(0, cx - (1 + m) * w / 2))
        x2 = int(min(pil_img.width,  cx + (1 + m) * w / 2))
        y1 = int(max(0, cy - (1 + m) * h / 2))
        y2 = int(min(pil_img.height, cy + (1 + m) * h / 2))
        face = pil_img.crop((x1, y1, x2, y2))
    else:
        face = pil_img
    
    face = face.resize(out_size)
    arr = np.asarray(face).astype("float32") / 255.0
    return arr, box
