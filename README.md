# 🩺 外科手术后严重并发症风险预测系统

**基于600例真实临床数据**

---

## 📋 项目简介

本项目开发了一个**可交互网页预测系统**，用于预测普通外科手术后严重并发症发生风险。

- **数据集**：600例真实医院外科手术患者
- **最佳模型**：CatBoost（5折OOF AUC = **0.8146**）
- **核心技术**：高级特征工程（10个临床衍生特征）+ SHAP可解释性分析
- **部署**：Streamlit Cloud 一键网页版（支持实时预测 + SHAP图）

---

## ✨ 核心亮点（复试/简历重点）

- 真实临床数据 + 高级特征工程（Inflammation_Nutrition_Ratio、Surgery_Risk_Score、Composite_Surgical_Risk_Index 等）
- 多模型对比 + 加权融合（CatBoost 最优）
- SHAP全局与局部解释（ASA评分、急诊手术、炎症营养比为Top特征，完全符合临床逻辑）
- 网页交互演示（导师可直接输入患者数据查看风险概率与解释图）

---

## 🚀 如何本地运行

```bash
pip install -r requirements.txt
streamlit run app.py