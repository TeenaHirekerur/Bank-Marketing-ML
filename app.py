
import streamlit as st
import pandas as pd
import joblib
import os
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
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Classification")
st.write("Compare five machine learning models to predict term deposit subscriptions.")

MODEL_PATH = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

@st.cache_resource
def load_models():
    loaded = {}

    for name, filename in model_files.items():
        path = os.path.join(MODEL_PATH, filename)
        loaded[name] = joblib.load(path)

    return loaded

try:
    trained_models = load_models()
    st.success("All five trained models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    list(model_files.keys())
)

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview")
    st.dataframe(data.head())

    if "y" not in data.columns:
        st.error("Target column 'y' is missing.")
        st.stop()

    model = trained_models[selected_model]

    required_features = list(model.feature_names_in_)

    missing = [
        col for col in required_features
        if col not in data.columns
    ]

    if missing:
        st.error(f"Missing input features: {missing}")
        st.stop()

    X = data[required_features]
    y_true = data["y"].replace({"no": 0, "yes": 1})

    try:
        y_true = pd.to_numeric(y_true).astype(int)

        if not set(y_true.unique()).issubset({0, 1}):
            raise ValueError("Target must contain only 0 and 1.")

    except Exception as e:
        st.error(f"Invalid target values: {e}")
        st.stop()

    if not set(y_true.unique()) == {0, 1}:
        st.error("Test data must contain both target classes.")
        st.stop()

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.divider()

    st.subheader(f"Evaluation Results: {selected_model}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.success("Model evaluation completed successfully!")

else:
    st.info("Please upload test_data.csv to evaluate the selected model.")
