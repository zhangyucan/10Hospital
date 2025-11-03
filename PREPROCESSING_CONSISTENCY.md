# 人脸检测预处理一致性说明

## 🎯 问题分析

你的模型是用 **dlib 裁剪的人脸图像**训练的，因此推理时必须保持预处理一致性，否则会影响准确率。

## 📊 训练时的预处理（dlib）

```python
# 你的训练预处理代码
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
faces = detector(gray)  # dlib 检测
face = faces[0]
x, y, w, h = face.left(), face.top(), face.width(), face.height()
face_crop = img[y:y+h, x:x+w]  # 直接裁剪，不扩展边距
# 注意：这里裁剪的是 BGR 图像（OpenCV）
```

### 特点：
1. ✅ 使用灰度图检测
2. ✅ 直接裁剪人脸框（**不扩展边距**）
3. ✅ 裁剪结果可能**不是正方形**（取决于人脸朝向）
4. ✅ BGR 格式（OpenCV）

## 🔄 推理时的预处理（MTCNN）

```python
# 当前的推理预处理
rgb01, bbox = crop_face_or_full(
    img,  # PIL Image (RGB)
    out_size=(512, 512),
    margin=0.0,        # ✅ 不扩展边距（与训练一致）
    force_square=True  # ✅ 强制正方形（补偿 dlib 可能的非正方形）
)
```

### 特点：
1. ✅ 使用 RGB 图像检测（MTCNN 特性）
2. ✅ 不扩展边距（`margin=0.0`）
3. ✅ 强制扩展为正方形（`force_square=True`）
4. ✅ RGB 格式（PIL）

## 🔑 关键参数说明

### `margin=0.0`（不扩展边距）
```python
# ❌ 错误：扩展 15% 边距（会包含更多背景）
margin=0.15  

# ✅ 正确：不扩展（与 dlib 训练一致）
margin=0.0
```

### `force_square=True`（强制正方形）
```python
# dlib 检测框可能不是正方形
w, h = face.width(), face.height()  # 例如: 150x180

# ❌ 如果 force_square=False:
# 直接使用 150x180 → Resize 到 512x512（变形！）

# ✅ 如果 force_square=True:
# 扩展到 180x180（以长边为准）→ Resize 到 512x512（不变形）
```

## 📐 正方形扩展逻辑

```python
if force_square:
    max_side = max(w, h)  # 取长边
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # 中心点
    
    # 以中心点为基准，扩展到正方形
    half_size = max_side / 2
    x1_new = cx - half_size
    x2_new = cx + half_size
    y1_new = cy - half_size
    y2_new = cy + half_size
    
    # 边界检查...
```

## 🎨 Resize 插值方法

```python
# 使用 LANCZOS 高质量插值（与训练时保持一致）
face = face.resize(out_size, Image.Resampling.LANCZOS)
```

## 🔄 RGB vs BGR

### 训练时：
- dlib 检测：灰度图
- 裁剪：BGR（OpenCV `cv2.imread`）
- 训练：**取决于你的模型输入**（如果用 PIL/torchvision，会自动转为 RGB）

### 推理时：
- MTCNN 检测：RGB（PIL Image）
- 裁剪：RGB
- 模型输入：RGB（与 PyTorch 标准一致）

**✅ 结论**：如果你的训练代码使用 `torchvision.transforms` 或 PIL，它会自动将 BGR 转为 RGB，所以推理时直接用 RGB 是正确的。

## 🧪 如何验证一致性

### 方法1：可视化对比
```bash
# 在本地运行测试脚本（需要安装 dlib）
python test_face_detection.py your_image.jpg
```

查看生成的结果：
- `2_face_direct_crop.jpg`：dlib 直接裁剪
- `4_no_margin_square.jpg`：MTCNN + 当前配置

应该非常接近！

### 方法2：统计对比
```python
# 比较裁剪结果的均值和标准差
dlib_crop_mean = np.mean(dlib_face_crop, axis=(0,1))
mtcnn_crop_mean = np.mean(mtcnn_face_crop, axis=(0,1))
diff = np.abs(dlib_crop_mean - mtcnn_crop_mean)
print(f"差异: {diff}")  # 应该很小（<10）
```

## 📝 建议配置

### 推荐配置（当前）
```python
margin=0.0          # 不扩展边距
force_square=True   # 强制正方形
```
**适用场景**：训练时 dlib 直接裁剪，可能得到非正方形人脸

### 备选配置1
```python
margin=0.0
force_square=False
```
**适用场景**：训练时明确使用非正方形人脸（少见）

### 备选配置2
```python
margin=0.1
force_square=True
```
**适用场景**：训练时有轻微扩展边距

## ⚡ 性能优化（可选）

如果 MTCNN 检测速度慢，可以先缩放：

```python
# 在 face_detect.py 中添加
def detect_first_face_box_fast(pil_img, target_short_side=640):
    # 缩小图像检测
    scale = target_short_side / min(pil_img.size)
    if scale < 1:
        small_img = pil_img.resize(
            (int(pil_img.width * scale), int(pil_img.height * scale))
        )
        bbox = detect_first_face_box(small_img)
        if bbox:
            # 还原到原图坐标
            return tuple(int(x / scale) for x in bbox)
    return detect_first_face_box(pil_img)
```

## ✅ 总结

| 预处理步骤 | 训练时（dlib） | 推理时（MTCNN） | 一致性 |
|-----------|---------------|----------------|--------|
| 检测输入 | 灰度图 | RGB | ✅ 不影响检测框 |
| 裁剪边距 | 0（直接裁剪） | 0（margin=0.0） | ✅ 一致 |
| 正方形化 | 可能非正方形 | 强制正方形 | ✅ 改进（避免变形） |
| Resize | 512x512 | 512x512 | ✅ 一致 |
| 颜色空间 | BGR→RGB(自动) | RGB | ✅ 一致 |
| 插值方法 | - | LANCZOS | ✅ 高质量 |

当前配置已经与 dlib 训练预处理保持最大一致性！🎯
