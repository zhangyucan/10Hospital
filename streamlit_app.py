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
- 你可选择匿名授权数据用于模型改进（可在设置中随时撤回）。
- 继续即表示你已阅读并同意本工具的使用与隐私说明。""",
        "agree_button": "✅ 同意并开始",
        "demo_button": "👀 仅体验演示",
        "exit_button": "❌ 退出",
        
        # 主界面
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
        # 启动页面
        "welcome_title": "AI-Assisted PCOS Facial Screening",
        "welcome_subtitle": "This system uses deep learning to identify phenotypic features associated with Polycystic Ovary Syndrome (PCOS) from facial images for non-invasive preliminary screening and risk assessment.",
        "data_title": "📊 Data & Internal Validation",
        "data_content": "This study utilizes multi-center data from **three tertiary hospitals in Shanghai and Hunan Province** (**325 cases total**, June 2023 – August 2024), trained and validated under standardized collection protocols. Internal holdout test set achieved PCOS binary classification accuracy exceeding **80%**. Actual performance may vary across different populations and imaging conditions; results are for reference only.",
        "disclaimer_title": "⚠️ Important Disclaimer",
        "disclaimer_content": """**PCOS diagnosis requires comprehensive assessment of clinical symptoms, hormone levels, ovulation function, and ovarian ultrasound, among other medical indicators.** This system is currently in open testing for scientific research purposes to collect more research data and clinical evidence. **It does not constitute medical diagnosis or treatment advice**; please consult professional physicians at accredited medical institutions for any health-related decisions.""",
        "privacy_title": "🔒 Privacy & Data Usage",
        "privacy_content": """- Uploaded images are used solely for this assessment and are not stored long-term by default.
- You may choose to anonymously authorize data for model improvement (can be revoked in settings at any time).
- Proceeding indicates you have read and agree to this tool's usage and privacy statement.""",
        "agree_button": "✅ Agree & Start",
        "demo_button": "👀 Demo Only",
        "exit_button": "❌ Exit",
        
        # 主界面
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

st.set_page_config(page_title="PCOS Screening System", page_icon="🩺", layout="wide")

# 初始化 session state
if "language" not in st.session_state:
    st.session_state.language = None
if "agreed" not in st.session_state:
    st.session_state.agreed = False
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

# ========== 启动页面 ==========
if not st.session_state.agreed and not st.session_state.demo_mode:
    # 语言选择
    if st.session_state.language is None:
        st.markdown("<h1 style='text-align: center;'>🩺</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>请选择语言 / Please Select Language</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            col_cn, col_en = st.columns(2)
            with col_cn:
                if st.button("🇨🇳 中文", use_container_width=True, type="primary", key="lang_cn"):
                    st.session_state.language = "中文"
                    st.rerun()
            with col_en:
                if st.button("🇬🇧 English", use_container_width=True, type="primary", key="lang_en"):
                    st.session_state.language = "English"
                    st.rerun()
        st.stop()
    
    # 显示启动页面内容
    t = LANGUAGES[st.session_state.language]
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(f"<h1 style='text-align: center;'>{t['welcome_title']}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 1.1em;'>{t['welcome_subtitle']}</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 数据与验证
        with st.expander(t["data_title"], expanded=True):
            st.markdown(t["data_content"])
        
        # 重要声明
        with st.expander(t["disclaimer_title"], expanded=True):
            st.warning(t["disclaimer_content"])
        
        # 隐私说明
        with st.expander(t["privacy_title"], expanded=True):
            st.info(t["privacy_content"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 操作按钮
        col_agree, col_demo, col_exit = st.columns(3)
        with col_agree:
            if st.button(t["agree_button"], use_container_width=True, type="primary", key="agree_btn"):
                st.session_state.agreed = True
                st.rerun()
        with col_demo:
            if st.button(t["demo_button"], use_container_width=True, key="demo_btn"):
                st.session_state.demo_mode = True
                st.rerun()
        with col_exit:
            if st.button(t["exit_button"], use_container_width=True, key="exit_btn"):
                st.stop()
    
    st.stop()

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
    
    # 返回启动页
    if st.button("← " + ("返回启动页" if language == "中文" else "Back to Welcome")):
        st.session_state.agreed = False
        st.session_state.demo_mode = False
        st.rerun()
    
    # 演示模式提示
    if st.session_state.demo_mode:
        st.info("🔍 " + ("演示模式" if language == "中文" else "Demo Mode"))

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
