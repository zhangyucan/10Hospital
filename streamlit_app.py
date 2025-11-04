from __future__ import annotations
from pathlib import Path
import streamlit as st
from pcos_infer import analyze_image_bytes

WEIGHTS_PATH = Path(__file__).parent / "weights" / "epoch006_0.00005_0.29149_0.8864.pth"

# 语言配置
LANGUAGES = {
    "中文": {
        # 启动页面
        "welcome_title": "AI 辅助 PCOS 面部筛查",
        "welcome_subtitle": "本系统利用深度学习从面部图像中识别与多囊卵巢综合征（PCOS）相关的表型特征，进行非侵入式初筛与风险评估。",
        "data_title": "📊 数据与内部验证",
        "data_content": "本研究采用来自**上海市与湖南省三家三甲医院**的多中心数据（共 **325 例**，2023 年 6 月–2024 年 8 月），在统一、规范的采集流程下完成训练与验证。于内部留出测试集，PCOS 二分类准确率超过 **80%**。不同人群与成像条件下的实际表现可能存在差异，结果仅供参考。",
        "disclaimer_title": "⚠️ 重要声明",
        "disclaimer_content": """**PCOS 诊断需综合临床症状、激素水平、排卵功能与卵巢超声等多项医学指标。** 本系统目前开放测试，仅用于科学研究，以便搜集更多的科研资料和临床证据，**不构成医疗诊断或治疗依据**；任何健康相关决策请咨询正规医疗机构专业医生。""",
        "privacy_title": "🔒 隐私与数据使用",
        "privacy_content": """- 上传图像仅用于本次评估，默认不做长期存储。
- 继续即表示你已阅读并同意本工具的使用与隐私说明。""",
        
        # 主界面
        "title": "多囊卵巢综合征 (PCOS) 辅助筛查系统",
        "intro": "上传一张面部照片，系统将基于深度学习模型进行辅助评估，并提供可视化分析结果。",
        "warning": "⚠️ **重要提示**: 本系统仅供科研参考使用，不能替代专业医疗诊断。如有疑虑，请及时就医咨询专业医生。",
        "model_not_found": "未找到模型权重",
        "model_not_found_msg": "请将训练好的权重文件放在 weights 目录，或在 Streamlit Secrets 中提供下载链接。",
        "photo_guide_title": "📸 **图像拍摄建议**:",
        "photo_guide_1": "⚠️ 请关闭美颜、滤镜等后处理功能",
        "photo_guide_2": "💡 使用自然光线拍摄，避免过度曝光或阴影",
        "photo_guide_3": "📷 拍摄方式的差异（光线、角度、相机设置等）可能影响分析结果",
        "photo_guide_4": "🎯 建议使用原始、未经处理的照片以获得更准确的评估",
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
        # 启动页面
        "welcome_title": "AI-Assisted PCOS Facial Screening",
        "welcome_subtitle": "This system uses deep learning to identify phenotypic features associated with Polycystic Ovary Syndrome (PCOS) from facial images for non-invasive preliminary screening and risk assessment.",
        "data_title": "📊 Data & Internal Validation",
        "data_content": "This study utilizes multi-center data from **three tertiary hospitals in Shanghai and Hunan Province** (**325 cases total**, June 2023 – August 2024), trained and validated under standardized collection protocols. Internal holdout test set achieved PCOS binary classification accuracy exceeding **80%**. Actual performance may vary across different populations and imaging conditions; results are for reference only.",
        "disclaimer_title": "⚠️ Important Disclaimer",
        "disclaimer_content": """**PCOS diagnosis requires comprehensive assessment of clinical symptoms, hormone levels, ovulation function, and ovarian ultrasound, among other medical indicators.** This system is currently in open testing for scientific research purposes to collect more research data and clinical evidence. **It does not constitute medical diagnosis or treatment advice**; please consult professional physicians at accredited medical institutions for any health-related decisions.""",
        "privacy_title": "🔒 Privacy & Data Usage",
        "privacy_content": """- Uploaded images are used solely for this assessment and are not stored long-term by default.
- Proceeding indicates you have read and agree to this tool's usage and privacy statement.""",
        
        # 主界面
        "title": "Polycystic Ovary Syndrome (PCOS) Screening System",
        "intro": "Upload a facial photo. The system will perform AI-assisted assessment based on deep learning models and provide visual analysis results.",
        "warning": "⚠️ **Important Notice**: This system is for research reference only and cannot replace professional medical diagnosis. If you have concerns, please seek medical consultation with a healthcare professional in a timely manner.",
        "model_not_found": "Model weights not found",
        "model_not_found_msg": "Please place the trained weights in the weights directory or provide a download URL in Streamlit Secrets.",
        "photo_guide_title": "📸 **Image Capture Guidelines**:",
        "photo_guide_1": "⚠️ Please disable beauty filters and post-processing features",
        "photo_guide_2": "💡 Use natural lighting, avoid overexposure or shadows",
        "photo_guide_3": "📷 Variations in capture methods (lighting, angle, camera settings, etc.) may affect analysis results",
        "photo_guide_4": "🎯 It is recommended to use original, unprocessed photos for more accurate assessment",
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

st.set_page_config(page_title="PCOS Screening System", page_icon="🩺", layout="wide")

# 初始化 session state
if "language" not in st.session_state:
    st.session_state.language = "中文"  # 默认中文

# ========== 主界面 ==========
language = st.session_state.language
t = LANGUAGES[language]

# 侧边栏
with st.sidebar:
    st.markdown("### " + ("设置" if language == "中文" else "Settings"))
    
    # 语言切换
    new_language = st.selectbox("Language / 语言", list(LANGUAGES.keys()), 
                                 index=list(LANGUAGES.keys()).index(language))
    if new_language != language:
        st.session_state.language = new_language
        st.rerun()
    
    st.markdown("---")
    
    # 关于
    with st.expander("ℹ️ " + ("关于本系统" if language == "中文" else "About This System"), expanded=True):
        st.markdown(f"**{t['welcome_title']}**")
        st.caption(t['welcome_subtitle'])
        
        st.markdown("---")
        
        # 数据与验证
        st.markdown(f"**{t['data_title']}**")
        st.caption(t["data_content"])
        
        st.markdown("---")
        
        # 重要声明
        st.markdown(f"**{t['disclaimer_title']}**")
        st.caption(t["disclaimer_content"])
        
        st.markdown("---")
        
        # 隐私说明
        st.markdown(f"**{t['privacy_title']}**")
        st.caption(t["privacy_content"])

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

if uploaded_file is not None:
    # 读取文件内容
    bytes_data = uploaded_file.getvalue()
    
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
                        st.image(result["crop"], caption=t["analyzed_face"], width="stretch")
                
                with col4:
                    if result.get("overlay") is not None:
                        st.image(result["overlay"], caption=t["attention_heatmap"], width="stretch")

else:
    st.info(t["upload_to_begin"])
