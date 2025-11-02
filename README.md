# 10Hospital

This repository contains a lightweight demo for estimating the probability of polycystic ovary syndrome (PCOS) from facial images. A custom Grad-CAM implementation highlights salient regions of the cropped face so the prediction is easier to interpret.

## Features
- Streamlit app with image upload, face crop preview, and Grad-CAM visualization.
- Minimal inference helper (`pcos_infer.py`) that avoids heavy external dependencies.
- Jupyter notebook (`test.ipynb`) showing an end-to-end inference and visualization pipeline.

## Quick Start
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Ensure the model weights are available in `weights/epoch006_0.00005_0.29149_0.8864.pth` (Git LFS handles downloads automatically).
3. Launch the Streamlit UI:
	```bash
	streamlit run streamlit_app.py
	```

## Notes
- The model is intended for research and demonstration only; it is *not* a medical diagnostic.
- Optional preprocessing uses `dlib` and `opencv-python` for face detection. When they are unavailable, the full image is used instead.
