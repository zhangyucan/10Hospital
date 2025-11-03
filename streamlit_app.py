from __future__ import annotations
from pathlib import Path
import streamlit as st
from pcos_infer import analyze_image_bytes

WEIGHTS_PATH = Path(__file__).parent / "weights" / "epoch006_0.00005_0.29149_0.8864.pth"

# 语言配置
LANGUAGES = {
    "中文": {
        "title": "多囊卵巢综合征 (PCOS) 辅助筛查系统",
        "intro": "上传面部照片进行 AI 辅助风险评估，并提供可视化分析洞察。",
        "warning": "⚠️ **重要声明**: 本系统仅供研究使用，不能替代专业医学诊断。如有疑虑请咨询医疗专业人员。",
        "model_not_found": "未找到模型权重",
        "model_not_found_msg": "请将训练好的权重文件放在 weights 目录，或在 Streamlit Secrets 中提供下载链接。",
        "photo_guide_title": "📸 **拍摄建议**:",
        "photo_guide_1": "⚠️ 请**关闭美颜滤镜和后期处理效果**",
        "photo_guide_2": "💡 使用自然光线，避免过曝或重度阴影",
        "photo_guide_3": "📷 拍摄条件（光线、角度、相机设置）的差异可能影响结果",
        "photo_guide_4": "🎯 使用原始、未处理的照片以获得更准确的评估",
        "upload_prompt": "选择图像文件",
        "empty_file": "文件为空，请重新上传。",
        "analyzing": "正在分析图像...",
        "analysis_failed": "分析失败",
        "error_prefix": "❌",
        "upload_another": "请上传另一张图片",
        "result_label": "筛查结果",
        "risk_score_label": "风险评分",
        "no_risk": "未见明显风险特征",
        "further_exam": "建议进一步检查",
        "result_interpretation": "📊 结果解读",
        "low_risk": "**低风险范围**: 模型评估显示特征在正常范围内。",
        "moderate_risk": "**中等风险范围**: 建议关注相关症状，必要时咨询医疗专业人员。",
        "high_risk": "**较高风险范围**: 强烈建议寻求医疗咨询，进行全面的内分泌和超声检查。",
        "note": "💡 **注意**: PCOS 诊断需要综合评估临床症状、激素水平、超声检查结果等多项医学指标。本系统仅作为初步筛查参考。",
        "analyzed_face": "分析面部区域",
        "attention_heatmap": "模型关注热力图",
        "upload_to_begin": "请上传图像开始分析。",
    },
    "English": {
        "title": "Polycystic Ovary Syndrome (PCOS) Screening System",
        "intro": "Upload a facial photo for AI-powered risk assessment with visual analysis insights.",
        "warning": "⚠️ **Important Notice**: This system is for research purposes only and cannot replace professional medical diagnosis. Please consult a healthcare provider if you have concerns.",
        "model_not_found": "Model weights not found",
        "model_not_found_msg": "Please place the trained weights in the weights directory or provide a download URL in Streamlit Secrets.",
        "photo_guide_title": "📸 **Photo Capture Guidelines**:",
        "photo_guide_1": "⚠️ Please **disable beauty filters and post-processing effects**",
        "photo_guide_2": "💡 Use natural lighting, avoid overexposure or heavy shadows",
        "photo_guide_3": "📷 Variations in capture conditions (lighting, angle, camera settings) may affect results",
        "photo_guide_4": "🎯 Use original, unprocessed photos for more accurate assessment",
        "upload_prompt": "Choose an image",
        "empty_file": "Empty file, please upload again.",
        "analyzing": "Analyzing image...",
        "analysis_failed": "Analysis failed",
        "error_prefix": "❌",
        "upload_another": "Please upload another image",
        "result_label": "Screening Result",
        "risk_score_label": "Risk Score",
        "no_risk": "No Significant Risk Features",
        "further_exam": "Further Examination Recommended",
        "result_interpretation": "📊 Result Interpretation",
        "low_risk": "**Low Risk Range**: Model assessment indicates features are within normal range.",
        "moderate_risk": "**Moderate Risk Range**: We recommend monitoring related symptoms and consulting a healthcare provider if necessary.",
        "high_risk": "**Higher Risk Range**: We strongly recommend seeking medical consultation for comprehensive endocrine and ultrasound examinations.",
        "note": "💡 **Note**: PCOS diagnosis requires comprehensive evaluation including clinical symptoms, hormone levels, ultrasound findings, and other medical indicators. This system serves only as a preliminary screening reference.",
        "analyzed_face": "Analyzed Face Region",
        "attention_heatmap": "Model Attention Heatmap",
        "upload_to_begin": "Please upload an image to begin.",
    }
}

st.set_page_config(page_title="PCOS Screening System", page_icon="🩺")

# 语言选择器（放在侧边栏）
with st.sidebar:
    language = st.selectbox("Language / 语言", list(LANGUAGES.keys()), index=0)
    t = LANGUAGES[language]

st.title(t["title"])

st.markdown(f"{t['intro']}\n\n{t['warning']}")

if not WEIGHTS_PATH.exists():
    st.error(f"{t['model_not_found']}: {WEIGHTS_PATH}.\n{t['model_not_found_msg']}")
    st.stop()

# 拍摄建议
st.warning(f"""{t['photo_guide_title']}
- {t['photo_guide_1']}
- {t['photo_guide_2']}
- {t['photo_guide_3']}
- {t['photo_guide_4']}
""")

uploaded_file = st.file_uploader(t["upload_prompt"], type=["jpg", "jpeg", "png"])

if uploaded_file:
    bytes_data = uploaded_file.read()
    if not bytes_data:
        st.warning(t["empty_file"])
    else:
        with st.spinner(t["analyzing"]):
            try:
                # Configure logging for face detection info
                import logging
                logging.basicConfig(level=logging.INFO)
                
                result = analyze_image_bytes(
                    bytes_data, 
                    make_cam=True, 
                    target_index=1
                )
            except Exception as exc:  # pragma: no cover - display to user
                st.error(f"{t['analysis_failed']}: {exc}")
                import traceback
                st.code(traceback.format_exc())
            else:
                # Check if face was detected
                if result.get("error"):
                    st.error(f"{t['error_prefix']} {result.get('error')}")
                    st.warning(result.get("message", t["upload_another"]))
                    st.stop()

                col1, col2 = st.columns(2)
                with col1:
                    pred = result.get("pred")
                    # Convert 0/1 to professional descriptions
                    if pred == 0:
                        status = t["no_risk"]
                        status_color = "🟢"
                    elif pred == 1:
                        status = t["further_exam"]
                        status_color = "🟡"
                    else:
                        status = str(pred)
                        status_color = "⚪"
                    st.metric(t["result_label"], f"{status_color} {status}")
                with col2:
                    probs = result.get("probs")
                    if probs and len(probs) > 1:
                        risk_level = probs[1] * 100
                        st.metric(t["risk_score_label"], f"{risk_level:.1f}%")
                
                # Add result interpretation
                st.markdown("---")
                st.subheader(t["result_interpretation"])
                probs = result.get("probs")
                if probs and len(probs) > 1:
                    risk_level = probs[1] * 100
                    if risk_level < 50:
                        st.success(t["low_risk"])
                    elif risk_level < 80:
                        st.warning(t["moderate_risk"])
                    else:
                        st.error(t["high_risk"])
                
                st.info(t["note"])

                # Display processed images
                col3, col4 = st.columns(2)
                with col3:
                    if result.get("crop") is not None:
                        st.image(result["crop"], caption=t["analyzed_face"], use_column_width=True)
                
                with col4:
                    if result.get("overlay") is not None:
                        st.image(result["overlay"], caption=t["attention_heatmap"], use_column_width=True)

else:
    st.info(t["upload_to_begin"])
