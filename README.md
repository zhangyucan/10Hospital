# 10Hospital - PCOS 面部辅助筛查系统

基于面部图像的多囊卵巢综合征 (PCOS) 风险评估演示系统。使用深度学习模型分析面部特征，并通过 Grad-CAM 热力图可视化关键区域。

## 核心特性
- 🎯 **强制人脸检测** - 确保数据一致性，仅对检测到人脸的图像进行预测
- 🔥 **Grad-CAM 可视化** - 突出显示模型关注的面部特征区域
- 🚀 **PyTorch 生态** - 使用 facenet-pytorch MTCNN，无需编译依赖
- ☁️ **云端部署友好** - 纯 Python 实现，适配 Streamlit Cloud
- 🏥 **医学级 UI** - 专业的风险分层和用户引导

## 快速开始

### 方式 1: Streamlit Cloud 部署（推荐）
**零安装，开箱即用**

直接部署到 Streamlit Cloud，无需本地配置：

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

**特点:**
- ✅ 无需安装 - 在线访问即可使用
- ✅ MTCNN 人脸检测 - 与训练数据预处理保持一致
- ✅ Python 3.13 兼容 - 纯 Python 依赖，无编译需求
- 📦 依赖: `requirements.txt` (streamlit, torch, facenet-pytorch)

### 方式 2: 本地开发
**完整控制和调试能力**

1. **克隆仓库**:
```bash
git clone https://github.com/zhangyucan/10Hospital.git
cd 10Hospital
```

2. **创建 Python 环境** (推荐 3.10+):
```bash
# 使用 conda
conda create -n hospital python=3.10 -y
conda activate hospital

# 或使用 venv
python3.10 -m venv venv
source venv/bin/activate
```

3. **安装依赖**:
```bash
pip install -r requirements.txt
```

4. **运行应用**:
```bash
streamlit run streamlit_app.py
```

5. **模型权重** (Git LFS 自动下载):
   - `weights/epoch006_0.00005_0.29149_0.8864.pth` - PCOS 分类模型 (208 MB)

## 技术架构

### 人脸检测方案
- **当前方案**: facenet-pytorch MTCNN
  - ✅ 纯 Python 实现，无编译依赖
  - ✅ Python 3.13 兼容
  - ✅ 适配 Streamlit Cloud
  - ✅ 强制检测：必须检测到人脸才进行预测

### 预处理一致性
**训练时** (dlib):
- 人脸检测 + 裁剪
- 正方形扩展 (max(w, h))
- margin=0.0 (无额外边距)
- LANCZOS 插值 resize 到 512×512

**推理时** (MTCNN):
- 人脸检测 + 裁剪
- 正方形扩展 (force_square=True)
- margin=0.0 (无额外边距)
- LANCZOS 插值 resize 到 512×512

### 核心模块
- `pcos_infer.py` - 推理引擎 (人脸检测 + 分类 + Grad-CAM)
- `streamlit_app.py` - Web UI (简洁医疗级界面)
- `model.py` - InceptionResNetV2 模型定义

## 重要说明
- ⚠️ **本系统仅供研究和演示使用，不能作为医学诊断依据**
- ⚠️ **必须检测到人脸才能进行预测** - 确保与训练数据一致
- 📸 **拍摄建议**: 正面拍摄、光线充足、避免遮挡
- 🔬 **模型训练**: 使用 dlib 检测和预处理的人脸图像训练
- 🎯 **预测目标**: 二分类 (0=未见明显风险特征, 1=建议进一步检查)
