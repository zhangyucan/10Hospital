# 10Hospital

This repository contains a lightweight demo for estimating the probability of polycystic ovary syndrome (PCOS) from facial images. A custom Grad-CAM implementation highlights salient regions of the cropped face so the prediction is easier to interpret.

## Features
- Streamlit app with image upload, face crop preview, and Grad-CAM visualization.
- Minimal inference helper (`pcos_infer.py`) that avoids heavy external dependencies.
- Jupyter notebook (`test.ipynb`) showing an end-to-end inference and visualization pipeline.

## Quick Start

### Automatic Installation (Recommended)
```bash
# Using the install script (Linux/macOS)
./install.sh

# Or using conda (easiest, works on all platforms)
conda create -n hospital python=3.10 -y
conda activate hospital
conda install -c conda-forge cmake dlib opencv -y
pip install -r requirements.txt
```

### Manual Installation
See [INSTALL.md](INSTALL.md) for detailed installation instructions for different platforms.

**Important**: `dlib` requires CMake to be installed via your system package manager:
- Ubuntu/Debian: `sudo apt-get install cmake`
- macOS: `brew install cmake`
- Windows: Download from [cmake.org](https://cmake.org/download/)

2. Ensure the model weights are available in `weights/` directory (Git LFS handles downloads automatically):
   - `epoch006_0.00005_0.29149_0.8864.pth` - PCOS classification model
   - `shape_predictor_68_face_landmarks.dat` - Face landmark detector (optional, improves face alignment)

3. Launch the Streamlit UI:
	```bash
	streamlit run streamlit_app.py
	```

## Notes
- The model is intended for research and demonstration only; it is *not* a medical diagnostic.
- **Face Detection**: The system uses `dlib` and `opencv-python` for automatic face detection and cropping:
  - When available, faces are automatically detected and cropped before prediction
  - The `shape_predictor_68_face_landmarks.dat` file enables more precise face alignment using 68 facial landmarks
  - When dependencies are unavailable or no face is detected, the full image is used instead
