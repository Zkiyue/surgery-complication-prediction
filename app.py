import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="外科手术并发症风险预测", layout="wide")
st.title("🩺 外科手术后严重并发症风险预测系统")
st.markdown("**600例真实临床数据 · CatBoost + SHAP 可解释性**")

@st.cache_resource
def load_model():
    with open("catboost_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ==================== 侧边栏（性别已改成中文） ====================
st.sidebar.header("📋 患者信息")
age = st.sidebar.slider("年龄 (岁)", 18, 85, 55)
sex_label = st.sidebar.selectbox("性别", ["女", "男"])
sex = 0 if sex_label == "女" else 1
bmi = st.sidebar.slider("BMI", 16.0, 42.0, 27.0)
asa_score = st.sidebar.slider("ASA 分级", 1, 4, 2)
surgery_duration_min = st.sidebar.slider("手术时长 (分钟)", 20, 300, 80)
diabetes = st.sidebar.checkbox("糖尿病")
hypertension = st.sidebar.checkbox("高血压")
smoking = st.sidebar.checkbox("吸烟史")
preop_albumin = st.sidebar.slider("术前白蛋白 (g/L)", 2.0, 5.0, 3.8)
preop_crp = st.sidebar.slider("术前 CRP (mg/L)", 3, 215, 40)
emergency = st.sidebar.checkbox("急诊手术")
open_surgery = st.sidebar.checkbox("开放手术")
iss_like_score = st.sidebar.slider("ISS-like 评分", 1, 33, 11)

if st.sidebar.button("🚀 预测并发症风险"):
    input_df = pd.DataFrame({
        'age': [age], 'sex': [sex], 'bmi': [bmi], 'asa_score': [asa_score],
        'surgery_duration_min': [surgery_duration_min], 'diabetes': [int(diabetes)],
        'hypertension': [int(hypertension)], 'smoking': [int(smoking)],
        'preop_albumin': [preop_albumin], 'preop_crp': [preop_crp],
        'emergency': [int(emergency)], 'open_surgery': [int(open_surgery)],
        'iss_like_score': [iss_like_score]
    })
    
    # 高级特征
    input_df['Inflammation_Nutrition_Ratio'] = input_df['preop_crp'] / (input_df['preop_albumin'] + 1e-6)
    input_df['Hypoalbuminemia'] = (input_df['preop_albumin'] < 3.5).astype(int)
    input_df['HyperCRP'] = (input_df['preop_crp'] > 50).astype(int)
    input_df['Surgery_Risk_Score'] = input_df['asa_score'] * (input_df['surgery_duration_min'] / 60)
    input_df['Open_Emergency_Risk'] = input_df['emergency'] * input_df['open_surgery']
    input_df['High_Risk_Surgery'] = ((input_df['emergency'] == 1) | (input_df['open_surgery'] == 1)).astype(int)
    input_df['Metabolic_Risk_Score'] = (input_df['bmi'] - 24) / 4 + input_df['diabetes'] * 2
    input_df['Age_Comorbidity_Index'] = input_df['age'] * (input_df['diabetes'] + input_df['hypertension'])
    input_df['CRP_Iss_Interaction'] = input_df['preop_crp'] * input_df['iss_like_score'] / 100
    input_df['Composite_Surgical_Risk_Index'] = (0.35 * input_df['Inflammation_Nutrition_Ratio'] +
                                                 0.30 * input_df['Surgery_Risk_Score'] +
                                                 0.20 * input_df['Age_Comorbidity_Index'] +
                                                 0.15 * input_df['Metabolic_Risk_Score'])

    prob = model.predict_proba(input_df)[0][1]
    st.success(f"**并发症发生概率：{prob:.1%}**")

    # ==================== 所有实验图 ====================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    st.subheader("🔍 图1: SHAP 特征重要性柱状图")
    fig1 = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap.Explanation(values=shap_values, data=input_df.values, feature_names=input_df.columns.tolist()), show=False)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    for feat, title in [
        ("asa_score", "图2: ASA 分级贡献图"),
        ("emergency", "图3: 急诊手术贡献图"),
        ("surgery_duration_min", "图4: 手术时长贡献图")
    ]:
        st.subheader(title)
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns.tolist()
        ), max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption("✅ 所有实验图已正常显示！红色=增加风险，蓝色=降低风险")
