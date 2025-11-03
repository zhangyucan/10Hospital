from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw
import streamlit as st

from pcos_infer import analyze_image_bytes

WEIGHTS_PATH = Path(__file__).parent / "weights" / "epoch006_0.00005_0.29149_0.8864.pth"
# WEIGHTS_PATH = r"/home/yucan/NewDisk/10Hospital/code/regressor/InceptionResNetV2_PCOS2nd/weights_clf/epoch006_0.00005_0.29149_0.8864.pth"


st.set_page_config(page_title="PCOS 辅助筛查系统", page_icon="🩺")
st.title("多囊卵巢综合征（PCOS）辅助筛查系统")

# 检查人脸检测功能是否可用
try:
    from face_detect import crop_face_or_full
    face_detection_available = True
    face_detection_msg = "✅ 人脸检测功能可用 (MTCNN - PyTorch)"
except Exception:
    face_detection_available = False
    face_detection_msg = "ℹ️ 人脸检测功能未安装（将使用完整图像）"

st.markdown(
    f"""
    上传一张面部照片，系统将基于深度学习模型进行辅助评估，并提供可视化分析结果。
    
    **{face_detection_msg}**
    
    ⚠️ **重要提示**：本系统仅供科研参考使用，不能替代专业医疗诊断。如有疑虑，请及时就医咨询专业医生。
    """
)

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 设置")
    use_face_detection = st.checkbox(
        "启用人脸检测（MTCNN）", 
        value=face_detection_available,  # 默认开启（如果可用）
        disabled=not face_detection_available,
        help="使用 PyTorch MTCNN 检测并裁剪人脸区域。如果检测失败，将自动使用完整图像。"
    )
    
    st.markdown("---")
    st.markdown("""
    ### 关于人脸检测
    - ✅ 纯 PyTorch 实现
    - ✅ 无需编译依赖
    - ✅ 自动回退机制
    - 💡 建议：如果训练数据使用完整图像，可关闭此选项
    """)

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
                # 配置日志以便在控制台看到人脸检测信息
                import logging
                logging.basicConfig(level=logging.INFO)
                
                result = analyze_image_bytes(
                    bytes_data, 
                    use_face=use_face_detection,
                    make_cam=True, 
                    target_index=1
                )
            except Exception as exc:  # pragma: no cover - display to user
                st.error(f"推理失败: {exc}")
                import traceback
                st.code(traceback.format_exc())
            else:
                # 显示实际使用的人脸检测方法
                detector_used = result.get("detector") or "none"
                st.info(f"图像处理方法: {detector_used}")

                col1, col2 = st.columns(2)
                with col1:
                    pred = result.get("pred")
                    # 将 0/1 转换为专业描述
                    if pred == 0:
                        status = "未见明显风险特征"
                        status_color = "🟢"
                    elif pred == 1:
                        status = "建议进一步检查"
                        status_color = "🟡"
                    else:
                        status = str(pred)
                        status_color = "⚪"
                    st.metric("筛查结果", f"{status_color} {status}")
                with col2:
                    probs = result.get("probs")
                    if probs and len(probs) > 1:
                        risk_level = probs[1] * 100
                        st.metric("风险指标", f"{risk_level:.1f}%")
                
                # 添加结果解读说明
                st.markdown("---")
                st.subheader("📊 结果解读")
                probs = result.get("probs")
                if probs and len(probs) > 1:
                    risk_level = probs[1] * 100
                    if risk_level < 30:
                        st.success("**低风险区间**：模型评估显示特征指标在正常范围内。")
                    elif risk_level < 70:
                        st.warning("**中等风险区间**：建议您关注相关症状，必要时咨询专业医生进行进一步检查。")
                    else:
                        st.error("**较高风险区间**：建议您尽快就医，进行全面的内分泌及超声检查，以获得准确诊断。")
                
                st.info("💡 **提示**：PCOS诊断需要结合临床症状、激素水平、超声检查等多项指标综合判断，本系统仅作为初步筛查参考。")

                # 显示处理后的图像
                col3, col4 = st.columns(2)
                with col3:
                    if result.get("crop") is not None:
                        st.image(result["crop"], caption="分析输入图像", use_column_width=True)
                    
                    # 如果检测到人脸框，显示原图 + 框标注
                    if result.get("bbox") is not None and use_face_detection:
                        from PIL import ImageDraw
                        img_with_box = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                        draw = ImageDraw.Draw(img_with_box)
                        bbox = result["bbox"]
                        draw.rectangle(bbox, outline=(255, 0, 0), width=3)
                        st.image(img_with_box, caption="人脸检测结果（红框）", use_column_width=True)
                
                with col4:
                    if result.get("overlay") is not None:
                        st.image(result["overlay"], caption="模型关注区域热力图", use_column_width=True)

                with st.expander("查看详细信息"):
                    st.json({"logits": result.get("logits"), "probs": result.get("probs")})

else:
    st.info("请先上传图片。")
