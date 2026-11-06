
import streamlit as st
import pandas as pd
import numpy as np
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

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Online Shopper Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f7f8fc;
}
h1 {
    color: #243b64;
}
.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛍️ Online Shopper Purchasing Intention")
st.markdown(
    "### Machine Learning Classification Dashboard"
)

st.write(
    "Predict whether an online shopping session "
    "will result in a purchase using five ML algorithms."
)

st.divider()

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

MODEL_PATH = "models"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}

@st.cache_resource
def load_models():
    loaded_models = {}

    for name, filename in model_files.items():
        path = os.path.join(MODEL_PATH, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing model file: {path}"
            )

        loaded_models[name] = joblib.load(path)

    return loaded_models

try:
    trained_models = load_models()
    st.success("All five trained models loaded successfully!")

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    # --------------------------------------------------
    # VALIDATE DATASET
    # --------------------------------------------------

    required_features = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend"
    ]

    if "Revenue" not in data.columns:
        st.error("CSV must contain the Revenue target column.")
        st.stop()

    missing_features = [
        col for col in required_features
        if col not in data.columns
    ]

    if missing_features:
        st.error(f"Missing features: {missing_features}")
        st.stop()

    if data.empty:
        st.error("Uploaded CSV is empty.")
        st.stop()

    if data[required_features].isnull().any().any():
        st.error("Input features contain missing values.")
        st.stop()

    if data["Revenue"].isnull().any():
        st.error("Revenue contains missing values.")
        st.stop()

    if not data["Revenue"].isin([True, False, 0, 1]).all():
        st.error("Revenue must contain TRUE/FALSE or 1/0.")
        st.stop()

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    X = data[required_features].copy()

    y = data["Revenue"].astype(int)

    st.write("Number of records:", len(data))

    st.divider()

    # --------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------

    st.header("2. Select Classification Model")

    selected_model = st.selectbox(
        "Choose an ML Algorithm",
        list(trained_models.keys())
    )

    if st.button("Evaluate Model", type="primary"):

        model = trained_models[selected_model]

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        # --------------------------------------------------
        # CALCULATE METRICS
        # --------------------------------------------------

        accuracy = accuracy_score(y, predictions)

        auc = roc_auc_score(y, probabilities)

        precision = precision_score(
            y, predictions, zero_division=0
        )

        recall = recall_score(
            y, predictions, zero_division=0
        )

        f1 = f1_score(
            y, predictions, zero_division=0
        )

        mcc = matthews_corrcoef(y, predictions)

        # --------------------------------------------------
        # DISPLAY METRICS
        # --------------------------------------------------

        st.header("3. Model Evaluation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y, predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Purchase", "Purchase"],
            yticklabels=["No Purchase", "Purchase"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(selected_model)

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        st.header("5. Classification Report")

        report = classification_report(
            y,
            predictions,
            target_names=["No Purchase", "Purchase"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True
        )

    # --------------------------------------------------
    # COMPARE ALL MODELS
    # --------------------------------------------------

    st.divider()

    st.header("6. Compare All Five Models")

    if st.button("Compare All Models"):

        comparison = []

        for name, model in trained_models.items():

            pred = model.predict(X)

            prob = model.predict_proba(X)[:, 1]

            comparison.append({
                "Model": name,
                "Accuracy": accuracy_score(y, pred),
                "AUC": roc_auc_score(y, prob),
                "Precision": precision_score(
                    y, pred, zero_division=0
                ),
                "Recall": recall_score(
                    y, pred, zero_division=0
                ),
                "F1 Score": f1_score(
                    y, pred, zero_division=0
                ),
                "MCC": matthews_corrcoef(y, pred)
            })

        comparison_df = pd.DataFrame(comparison)

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True
        )

else:
    st.info(
        "Please upload test_data.csv to evaluate "
        "your trained classification models."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "BITS Pilani | M.Tech Artificial Intelligence "
    "& Machine Learning | Machine Learning Assignment 2"
)