# 10Hospital

This repository contains a lightweight demo for estimating the probability of polycystic ovary syndrome (PCOS) from facial images. A custom Grad-CAM implementation highlights salient regions of the cropped face so the prediction is easier to interpret.

## Features
- Streamlit app with image upload, face crop preview, and Grad-CAM visualization.
- Minimal inference helper (`pcos_infer.py`) that avoids heavy external dependencies.
- Jupyter notebook (`test.ipynb`) showing an end-to-end inference and visualization pipeline.

## Quick Start

### Option 1: Cloud Deployment (Streamlit Cloud)
**推荐用于快速演示**

Deploy directly to Streamlit Cloud - no installation needed!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

The app works without face detection (uses full images for prediction). Face detection is optional for better accuracy.

### Option 2: Local Development with Face Detection
**推荐用于最佳性能和准确性**

**Using Conda (Easiest)**:
```bash
conda create -n hospital python=3.10 -y
conda activate hospital
conda install -c conda-forge cmake dlib opencv -y
pip install -r requirements.txt
```

**Using pip (requires system CMake)**:
```bash
# Install system dependencies first
sudo apt-get install cmake build-essential  # Ubuntu/Debian
# or: brew install cmake  # macOS

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install all dependencies including face detection
pip install -r requirements-full.txt
```

**Using install script (Linux/macOS)**:
```bash
./install.sh
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

2. Ensure the model weights are available in `weights/` directory (Git LFS handles downloads automatically):
   - `epoch006_0.00005_0.29149_0.8864.pth` - PCOS classification model
   - `shape_predictor_68_face_landmarks.dat` - Face landmark detector (optional, improves face alignment)

3. Launch the Streamlit UI:
	```bash
	streamlit run streamlit_app.py
	```

## Deployment Options

### Cloud Deployment (Streamlit Cloud)
- ✅ Zero installation - works out of the box
- ⚠️ No face detection (uses full images)
- 📦 Uses `requirements.txt` (core dependencies only)

### Local Development
- ✅ Full face detection with 68-point landmark alignment
- ✅ Better accuracy with automatic face cropping
- 📦 Uses `requirements-full.txt` (includes opencv-python, dlib)
- 💻 Requires CMake installation

## Notes
- The model is intended for research and demonstration only; it is *not* a medical diagnostic.
- **Face Detection** (optional): 
  - Automatically detects and crops faces for better accuracy
  - Uses `dlib` with 68 facial landmark points for precise alignment
  - Requires local installation with `requirements-full.txt`
  - When unavailable, the full image is used (still works, slightly less accurate)
