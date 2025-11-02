from __future__ import annotations

from pathlib import Path

import streamlit as st

from pcos_infer import analyze_image_bytes

WEIGHTS_PATH = Path(__file__).parent / "weights" / "epoch006_0.00005_0.29149_0.8864.pth"
# WEIGHTS_PATH = r"/home/yucan/NewDisk/10Hospital/code/regressor/InceptionResNetV2_PCOS2nd/weights_clf/epoch006_0.00005_0.29149_0.8864.pth"


st.set_page_config(page_title="PCOS Probability Analyzer", page_icon="🩺")
st.title("PCOS Probability Analyzer")
st.markdown(
    """
    上传一张人脸照片，模型会给出患 PCOS 的概率，并展示 Grad-CAM 热力图。\
    模型仅用于科研原型，请勿作为医疗诊断依据。
    """
)

if not WEIGHTS_PATH.exists():
    st.error(
        f"未找到权重文件: {WEIGHTS_PATH}.\n"
        "请将训练好的权重放到 weights 目录，或在 Streamlit Secrets 中提供下载地址。"
    )
    st.stop()

uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    bytes_data = uploaded_file.read()
    if not bytes_data:
        st.warning("文件为空，请重新上传。")
    else:
        with st.spinner("模型推理中..."):
            try:
                result = analyze_image_bytes(bytes_data, make_cam=True, target_index=1)
            except Exception as exc:  # pragma: no cover - display to user
                st.error(f"推理失败: {exc}")
            else:
                st.metric("预测类别", result.get("pred"))
                probs = result.get("probs")
                if probs and len(probs) > 1:
                    st.metric("患 PCOS 概率", f"{probs[1] * 100:.2f}%")

                if result.get("crop") is not None:
                    st.image(result["crop"], caption="裁切后的人脸", use_column_width=True)
                if result.get("overlay") is not None:
                    st.image(result["overlay"], caption="裁切区域的 Grad-CAM", use_column_width=True)

                st.json({"logits": result.get("logits")})

else:
    st.info("请先上传图片。")
