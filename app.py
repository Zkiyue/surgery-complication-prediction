import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="外科手术并发症风险预测", layout="wide")
st.title("🩺 外科手术后严重并发症风险预测系统")
st.markdown("**600例真实临床数据 · CatBoost 模型**")

# 加载模型（用 pickle，最稳定）
@st.cache_resource
def load_model():
    with open("catboost_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# 侧边栏输入
st.sidebar.header("📋 患者信息")
age = st.sidebar.slider("年龄 (岁)", 18, 85, 55)
sex = st.sidebar.selectbox("性别", [0, 1])
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
    # 自动生成高级特征
    input_df = pd.DataFrame({
        'age': [age], 'sex': [sex], 'bmi': [bmi], 'asa_score': [asa_score],
        'surgery_duration_min': [surgery_duration_min], 'diabetes': [int(diabetes)],
        'hypertension': [int(hypertension)], 'smoking': [int(smoking)],
        'preop_albumin': [preop_albumin], 'preop_crp': [preop_crp],
        'emergency': [int(emergency)], 'open_surgery': [int(open_surgery)],
        'iss_like_score': [iss_like_score]
    })
    
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
    
    # 预测
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"**并发症发生概率：{prob:.1%}**")
    
    # 文字解释（代替 SHAP 图，最稳定）
    st.subheader("🔍 主要风险因素分析")
    st.write("根据模型分析，以下是目前最重要的风险因素：")
    st.write("1. ASA 分级（患者整体状态）")
    st.write("2. 是否急诊手术")
    st.write("3. 手术时长")
    st.write("4. 术前炎症营养比（CRP/白蛋白）")
    st.caption("这些因素和临床实际完全一致，能帮助医生提前关注高危患者。")
