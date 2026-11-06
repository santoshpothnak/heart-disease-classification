import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Heart Disease Classification Benchmark",
    page_icon="🩺",
    layout="wide"
)

# Custom header styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-lbl {
        font-size: 14px;
        color: #6c757d;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Heart Disease Classification Benchmark")
st.markdown(
    "Evaluate pre-trained machine learning classifiers against standardized clinical test records without retraining."
)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Configuration & Data")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_mapping = {
    "Logistic Regression": (os.path.join(BASE_DIR, "model/logistic_regression.joblib"), True),
    "Decision Tree": (os.path.join(BASE_DIR, "model/decision_tree.joblib"), False),
    "KNN": (os.path.join(BASE_DIR, "model/knn.joblib"), True),
    "Naive Bayes": (os.path.join(BASE_DIR, "model/naive_bayes.joblib"), False),
    "Support Vector Machine": (os.path.join(BASE_DIR, "model/svm.joblib"), True)
}

selected_model_name = st.sidebar.selectbox(
    "Select ML Classification Model",
    list(model_mapping.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (.csv)",
    type=["csv"],
    help="Upload a CSV with the 13 clinical input features and the 'target' column."
)

default_test_path = os.path.join(BASE_DIR, "test_data.csv")

# Load Test Data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom test dataset loaded successfully.")
elif os.path.exists(default_test_path):
    data = pd.read_csv(default_test_path)
    st.sidebar.info("Using default repository 'test_data.csv'.")
else:
    st.error("Test data not found. Please upload a valid test CSV file.")
    st.stop()

# Validate Schema
if "target" not in data.columns:
    st.error("Uploaded test CSV must contain the ground-truth 'target' column.")
    st.stop()

X_test = data.drop(columns=["target"])
y_test = data["target"]

if X_test.shape[1] < 12:
    st.error(f"Expected at least 12 input features. Found: {X_test.shape[1]}")
    st.stop()

# ----------------- MODEL EXECUTION -----------------
model_path, needs_scaler = model_mapping[selected_model_name]
scaler_path = os.path.join(BASE_DIR, "model/scaler.joblib")

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please run train_and_export.py first.")
    st.stop()

clf = joblib.load(model_path)

if needs_scaler:
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_eval = scaler.transform(X_test)
    else:
        st.error("Scaler file 'model/scaler.joblib' missing.")
        st.stop()
else:
    X_eval = X_test

# Inference
y_pred = clf.predict(X_eval)
if hasattr(clf, "predict_proba"):
    y_prob = clf.predict_proba(X_eval)[:, 1]
elif hasattr(clf, "decision_function"):
    y_prob = clf.decision_function(X_eval)
else:
    y_prob = y_pred

# Calculate Metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# ----------------- METRIC DISPLAY -----------------
st.subheader(f"Evaluation Metrics: {selected_model_name}")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Accuracy", f"{acc:.4f}")
m2.metric("AUC Score", f"{auc:.4f}")
m3.metric("Precision", f"{prec:.4f}")
m4.metric("Recall", f"{rec:.4f}")
m5.metric("F1 Score", f"{f1:.4f}")
m6.metric("MCC Score", f"{mcc:.4f}")

st.markdown("---")

# ----------------- DIAGNOSTIC VISUALS -----------------
col_cm, col_report = st.columns([1, 1])

with col_cm:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Disease (0)", "Disease (1)"],
        yticklabels=["No Disease (0)", "Disease (1)"],
        ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    st.pyplot(fig)

with col_report:
    st.subheader("Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(
        report_df.style.format({
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1-score": "{:.3f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

# ----------------- ALL MODELS BENCHMARK TABLE -----------------
with st.expander("📊 Benchmark Comparison Across All 5 Models on Current Test Data", expanded=True):
    all_rows = []
    for m_name, (m_path, m_scale) in model_mapping.items():
        if os.path.exists(m_path):
            m_clf = joblib.load(m_path)
            X_m = scaler.transform(X_test) if (m_scale and os.path.exists(scaler_path)) else X_test
            yp = m_clf.predict(X_m)
            ypr = m_clf.predict_proba(X_m)[:, 1] if hasattr(m_clf, "predict_proba") else yp
            all_rows.append({
                "ML Model Name": m_name,
                "Accuracy": round(accuracy_score(y_test, yp), 4),
                "AUC": round(roc_auc_score(y_test, ypr), 4),
                "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
                "Recall": round(recall_score(y_test, yp, zero_division=0), 4),
                "F1": round(f1_score(y_test, yp, zero_division=0), 4),
                "MCC": round(matthews_corrcoef(y_test, yp), 4)
            })
    if all_rows:
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

# ----------------- DATASET PREVIEW -----------------
with st.expander("Inspect Test Dataset & Predictions"):
    preview_df = X_test.copy()
    preview_df["Actual_Target"] = y_test.values
    preview_df["Predicted_Class"] = y_pred
    if hasattr(clf, "predict_proba"):
        preview_df["Probability_Class_1"] = np.round(y_prob, 4)
    st.dataframe(preview_df.head(20), use_container_width=True)
