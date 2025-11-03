# face_detect.py
"""
基于 facenet-pytorch 的轻量级人脸检测模块
使用 MTCNN 模型，纯 PyTorch 实现，适合 Streamlit Cloud CPU 环境

优先级：MTCNN (云端) > dlib (本地开发)
"""
from PIL import Image
import numpy as np

_MTCNN = None
_DLIB_DETECTOR = None

def _get_mtcnn():
    """延迟初始化 MTCNN，避免影响其他功能"""
    global _MTCNN
    if _MTCNN is None:
        try:
            from facenet_pytorch import MTCNN  # 延迟导入，若未安装也不影响其它功能
            _MTCNN = MTCNN(keep_all=False, device='cpu', post_process=False)
            print("✅ 使用 MTCNN (PyTorch) 人脸检测")
        except Exception as e:
            print(f"⚠️ MTCNN 不可用: {e}")
            _MTCNN = False  # 标记为不可用
    return _MTCNN if _MTCNN is not False else None

def _get_dlib_detector():
    """延迟初始化 dlib 检测器（备选方案）"""
    global _DLIB_DETECTOR
    if _DLIB_DETECTOR is None:
        try:
            import dlib
            import cv2
            _DLIB_DETECTOR = {
                'detector': dlib.get_frontal_face_detector(),
                'cv2': cv2
            }
            print("✅ 使用 dlib 人脸检测（本地开发）")
        except Exception as e:
            print(f"⚠️ dlib 不可用: {e}")
            _DLIB_DETECTOR = False  # 标记为不可用
    return _DLIB_DETECTOR if _DLIB_DETECTOR is not False else None

def detect_first_face_box(pil_img: Image.Image, conf_thres=0.6):
    """
    检测图像中置信度最高的人脸
    优先使用 MTCNN，失败则尝试 dlib
    
    Args:
        pil_img: PIL Image 对象
        conf_thres: 置信度阈值（仅用于 MTCNN）
        
    Returns:
        (x1, y1, x2, y2) 人脸框坐标，或 None（未检测到）
    """
    # 优先尝试 MTCNN
    mtcnn = _get_mtcnn()
    if mtcnn is not None:
        try:
            boxes, probs = mtcnn.detect(pil_img)
            if boxes is not None and probs is not None and len(boxes) > 0:
                i = int(np.nanargmax(probs))
                if probs[i] is not None and probs[i] >= conf_thres:
                    x1, y1, x2, y2 = boxes[i]
                    return tuple(map(int, [x1, y1, x2, y2]))
        except Exception as e:
            print(f"MTCNN 检测失败: {e}")
    
    # 备选方案：dlib（本地开发）
    dlib_detector = _get_dlib_detector()
    if dlib_detector is not None:
        try:
            cv2 = dlib_detector['cv2']
            detector = dlib_detector['detector']
            
            # 转换为 numpy 数组并转为灰度图
            img_array = np.array(pil_img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # dlib 检测
            faces = detector(gray, 1)
            if len(faces) > 0:
                # 使用第一个检测到的人脸
                face = faces[0]
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                return (x1, y1, x2, y2)
        except Exception as e:
            print(f"dlib 检测失败: {e}")
    
    return None

def crop_face_or_full(pil_img: Image.Image, out_size=(512, 512), margin=0.0, force_square=True):
    """
    优先裁剪人脸；失败则回退到使用整图
    
    Args:
        pil_img: 输入 PIL Image
        out_size: 输出尺寸
        margin: 人脸框扩展边距（相对于框宽高的比例），默认 0（与 dlib 训练一致）
        force_square: 是否强制裁剪为正方形（与 dlib 训练一致）
        
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
        
        # 与 dlib 训练预处理保持一致：
        # 1. 使用原始检测框（不扩展边距，或使用极小边距）
        # 2. 如果需要正方形，扩展到最大边
        if force_square:
            # 扩展到正方形（以较大边为准）
            max_side = max(w, h)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 计算正方形框的坐标
            half_size = max_side * (1 + margin) / 2
            x1_new = int(max(0, cx - half_size))
            x2_new = int(min(pil_img.width, cx + half_size))
            y1_new = int(max(0, cy - half_size))
            y2_new = int(min(pil_img.height, cy + half_size))
            
            face = pil_img.crop((x1_new, y1_new, x2_new, y2_new))
        else:
            # 直接使用检测框（可能不是正方形）
            if margin > 0:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                x1 = int(max(0, cx - (1 + margin) * w / 2))
                x2 = int(min(pil_img.width, cx + (1 + margin) * w / 2))
                y1 = int(max(0, cy - (1 + margin) * h / 2))
                y2 = int(min(pil_img.height, cy + (1 + margin) * h / 2))
            else:
                # 边界检查
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(pil_img.width, x2)
                y2 = min(pil_img.height, y2)
            
            face = pil_img.crop((x1, y1, x2, y2))
    else:
        face = pil_img
    
    # Resize 到目标尺寸（使用 LANCZOS 高质量插值，与训练时保持一致）
    face = face.resize(out_size, Image.Resampling.LANCZOS)
    arr = np.asarray(face).astype("float32") / 255.0
    return arr, box
