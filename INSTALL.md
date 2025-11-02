# 安装指南 (Installation Guide)

本项目需要人脸检测功能，依赖 `dlib` 和 `opencv-python`。`dlib` 的安装需要 CMake，请按照以下步骤操作。

## 快速安装 (推荐方式)

### 方法 1: 使用 Conda (最简单，强烈推荐)

```bash
# 创建新环境
conda create -n hospital python=3.10 -y
conda activate hospital

# 通过 conda 安装 cmake 和 dlib (避免编译问题)
conda install -c conda-forge cmake dlib opencv -y

# 安装其他依赖
pip install -r requirements.txt
```

### 方法 2: 使用系统包管理器 + pip

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
pip install cmake  # 先安装 cmake
pip install -r requirements.txt
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
pip install -r requirements.txt
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
pip install -r requirements.txt
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

## 不安装人脸检测依赖

如果无法安装 dlib 和 opencv，系统会自动回退到使用完整图像进行预测（不进行人脸裁剪）。功能仍然可用，只是准确性可能会受影响。

跳过人脸检测依赖的安装：
```bash
# 只安装核心依赖
pip install streamlit numpy Pillow torch torchvision
```

## 启动应用

```bash
streamlit run streamlit_app.py
```

然后在浏览器中打开显示的 URL (通常是 http://localhost:8501)。
