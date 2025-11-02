# 安装指南 (Installation Guide)

本项目支持两种部署方式：

1. **云端部署** (Streamlit Cloud) - 无需安装，人脸检测不可用
2. **本地开发** - 完整功能，包括人脸检测

人脸检测功能依赖 `dlib` 和 `opencv-python`。`dlib` 的安装需要 CMake。

## 云端部署 (推荐用于演示)

直接部署到 Streamlit Cloud，无需任何安装：

1. Fork 此仓库到你的 GitHub 账号
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 连接你的 GitHub 账号并选择此仓库
4. 点击 Deploy！

**注意**: 云端部署不支持人脸检测（`dlib` 需要编译），但应用仍然可以正常工作，使用完整图像进行预测。

## 本地开发安装 (推荐用于最佳性能)

### 方法 1: 使用 Conda (最简单，强烈推荐)

```bash
# 创建新环境
conda create -n hospital python=3.10 -y
conda activate hospital

# 通过 conda 安装 cmake 和 dlib (避免编译问题)
conda install -c conda-forge cmake dlib opencv -y

# 安装其他依赖 (PyTorch, Streamlit 等)
pip install -r requirements.txt
```

### 方法 2: 使用系统包管理器 + pip (完整功能)

#### Ubuntu/Debian:
```bash
# 1. 安装系统依赖
sudo apt-get update
sudo apt-get install -y cmake build-essential

# 2. 创建虚拟环境 (推荐 Python 3.10 或 3.11)
python3.10 -m venv venv
source venv/bin/activate

# 3. 安装 Python 依赖
pip install --upgrade pip
pip install -r requirements-full.txt  # 包含人脸检测依赖
```

#### macOS:
```bash
# 1. 安装 Homebrew (如果还没有)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装 CMake
brew install cmake

# 3. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 4. 安装依赖
pip install --upgrade pip
pip install -r requirements-full.txt
```

#### Windows:
```powershell
# 1. 下载并安装 CMake
# 访问 https://cmake.org/download/
# 下载 Windows x64 Installer
# 安装时选择 "Add CMake to system PATH"

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements-full.txt
```

## 验证安装

运行测试脚本验证人脸检测功能：

```bash
python test_face_detection.py
```

如果看到以下输出，说明安装成功：
```
============================================================
人脸检测功能测试
============================================================
✓ cv2 和 dlib 导入成功
✓ dlib 人脸检测器初始化成功
✓ shape_predictor 加载成功
============================================================
所有测试通过！人脸检测功能应该能正常工作。
============================================================
```

## 常见问题

### 问题 1: dlib 安装失败 (CMake 错误)

**错误信息**: `CMake is not installed on your system!`

**解决方案**:
1. 确保系统已安装 CMake: `cmake --version`
2. 如果没有，使用系统包管理器安装 (不要用 pip):
   - Ubuntu/Debian: `sudo apt-get install cmake`
   - macOS: `brew install cmake`
   - Windows: 从 cmake.org 下载安装
3. 重新安装 dlib: `pip install dlib`

### 问题 2: Python 3.13 兼容性问题

**错误信息**: dlib 在 Python 3.13 上编译失败

**解决方案**: 使用 Python 3.10 或 3.11:
```bash
conda create -n hospital python=3.10 -y
conda activate hospital
```

### 问题 3: 使用 uv 安装时失败

如果使用 `uv pip install` 遇到问题，建议：
1. 先用系统包管理器安装 CMake
2. 然后使用标准的 pip: `pip install -r requirements.txt`

或者切换到 conda 环境（推荐）。

## 简化安装 (不包含人脸检测)

如果无法安装 dlib 和 opencv，或只想快速测试，可以使用核心依赖：

```bash
# 只安装核心依赖 (与 Streamlit Cloud 相同)
pip install -r requirements.txt
```

系统会自动检测并回退到使用完整图像进行预测（不进行人脸裁剪）。功能仍然可用，准确性可能略有影响。

## 启动应用

```bash
streamlit run streamlit_app.py
```

然后在浏览器中打开显示的 URL (通常是 http://localhost:8501)。
