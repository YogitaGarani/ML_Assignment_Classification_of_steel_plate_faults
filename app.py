# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report

# ==============================
# INTERFACE THEMING & OVERRIDES
# ==============================
st.set_page_config(page_title="Industrial Defect Dashboard", layout="wide")
st.markdown("""
<style>
    .brand-title { font-size: 2.4rem; font-weight: 800; color: var(--text-color); letter-spacing: -0.05rem; }
    .brand-subtitle { font-size: 1.1rem; color: var(--text-color); opacity: 0.8; margin-bottom: 2rem; }
    
    .panel-section-header { font-size: 1.3rem; font-weight: 700; color: var(--text-color); margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 2px solid var(--secondary-background-color); padding-bottom: 0.3rem; }
    
    .kpi-card { 
        background-color: var(--secondary-background-color); 
        border-radius: 8px; 
        border: 1px solid var(--secondary-background-color); 
        padding: 1rem; 
        border-top: 4px solid #3B82F6;
    }
    
    .kpi-card h2 {
        color: var(--text-color) !important;
        margin: 0 !important;
        padding-top: 0.3rem !important;
        font-size: 1.8rem !important;
    }
    
    .sidebar-branding { font-weight: 700; font-size: 1.1rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ===================
# APPLICATION HEADER
# ===================
st.markdown("<div class='brand-title'>⚡ Metallurgical Quality Assurance Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='brand-subtitle'>Automated pattern recognition engine that evaluates 27 surface, luminosity, and geometric telemetry points to diagnose structural anomalies in steel plates.</div>", unsafe_allow_html=True)

target_labels = ["Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

# ========
# SIDEBAR
# ========
st.sidebar.markdown("<div class='sidebar-branding'>🏭 Command Center</div>", unsafe_allow_html=True)

st.sidebar.markdown("### 📥 Ingest Data")
uploaded_file = st.sidebar.file_uploader("Upload your data here", type=["csv"], label_visibility="collapsed")

st.sidebar.markdown("### 🧠 Select Classifier")
model_options = ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Support_Vector_Machine"]
selected_model_name = st.sidebar.selectbox("Pipeline Architecture", model_options, label_visibility="collapsed")

# ===============
# MAIN INTERFACE
# ===============
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Logged telemetry streams successfully.")

    if 'Target' not in data.columns:
        st.error("Missing Vector Dimension: The loaded dataset requires a 'Target' integer field matching index bounds 0-6 for performance monitoring.")
    else:
        st.markdown("<div class='panel-section-header'>🔬 Preview uploaded data</div>", unsafe_allow_html=True)
        st.dataframe(data.head(5), use_container_width=True)

        y_true = data['Target'].values
        X_eval = data.drop(columns=['Target'])

        scaler_path = "model/scaler.joblib"
        model_path = f"model/{selected_model_name}.joblib"

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            st.error(f"Deployment Mismatch: The requested artifacts (`scaler.joblib` or `{selected_model_name}.joblib`) were not located within the local production workspace paths.")
        else:
            # Inference
            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path)

            X_eval_scaled = scaler.transform(X_eval)
            y_pred = model.predict(X_eval_scaled)
            y_prob = model.predict_proba(X_eval_scaled)

            # Performance
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            mcc = matthews_corrcoef(y_true, y_pred)

            st.markdown(f"<div class='panel-section-header'>📊 Multi-Class Performance Report: {selected_model_name.replace('_', ' ')}</div>", unsafe_allow_html=True)

            c1, col_divider, c2, col_divider2, c3, col_divider3, c4, col_divider4, c5, col_divider5, c6 = st.columns([1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1])
            with c1: st.markdown(f"<div class='kpi-card'><small style='color:#6B7280; font-weight:600;'>ACCURACY</small><h2>{acc:.4f}</h2></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='kpi-card'><small style='color:#6B7280; font-weight:600;'>ROC-AUC SCORE</small><h2>{auc:.4f}</h2></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='kpi-card'><small style='color:#6B7280; font-weight:600;'>PRECISION (MACRO)</small><h2>{prec:.4f}</h2></div>", unsafe_allow_html=True)
            with c4: st.markdown(f"<div class='kpi-card'><small style='color:#6B7280; font-weight:600;'>RECALL (MACRO)</small><h2>{rec:.4f}</h2></div>", unsafe_allow_html=True)
            with c5: st.markdown(f"<div class='kpi-card'><small style='color:#6B7280; font-weight:600;'>F1-SCORE (MACRO)</small><h2>{f1:.4f}</h2></div>", unsafe_allow_html=True)
            with c6: st.markdown(f"<div class='kpi-card'><small style='color:#6B7280; font-weight:600;'>MCC COEFFICIENT</small><h2>{mcc:.4f}</h2></div>", unsafe_allow_html=True)

            st.markdown("<div class='panel-section-header'>📋 Deep-Dive Error Diagnostics & Breakdown</div>", unsafe_allow_html=True)
            layout_left, space, layout_right = st.columns([1.1, 0.1, 1.1])

            with layout_left:
                st.markdown("<p style='font-weight:600; color: var(--text-color); margin-bottom: 25px;'>Confusion Heatmap</p>", unsafe_allow_html=True)
                
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=target_labels, yticklabels=target_labels, ax=ax, cbar=False)
                
                ax.set_ylabel("Truth Values", fontweight='bold', color='#374151')
                ax.set_xlabel("Model Predictions", fontweight='bold', color='#374151')
                
                plt.tight_layout()
                
                fig.patch.set_facecolor('#F8FAFC')
                st.pyplot(fig)



            with layout_right:
                st.markdown("<p style='font-weight:600; color:#4B5563; margin-bottom:10px;'>Classification Report</p>", unsafe_allow_html=True)
                report_dict = classification_report(y_true, y_pred, target_names=target_labels, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 Awaiting structural configuration file payload. Please feed or upload a valid csv file with steel pipeline surface data to analyse anomalies")
