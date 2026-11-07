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
    classification_report,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Online Shopper Purchasing Intention",
    page_icon="assets/online-store.png",
    layout="wide",
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #243b64;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #243b64;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8edf5;
        padding: 13px 15px;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(25, 45, 80, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
    .subtle-note {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* Dropdown selected value */
    .stSelectbox div[data-baseweb="select"] div {
        color: #000000 !important;
    }

    /* Dropdown options */
    div[data-baseweb="popover"] ul li {
        color: #000000 !important;
    }

    /* All buttons */
    .stButton > button {
        background-color: #3BE0D0 !important;
        color: #000000 !important;
        border: 1px solid #3BE0D0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Button hover */
    .stButton > button:hover {
        background-color: #25CDBD !important;
        color: #000000 !important;
        border-color: #25CDBD !important;
    }

    .stButton > button:focus {
        color: #000000 !important;
        border-color: #25CDBD !important;
        box-shadow: 0 0 0 2px rgba(59, 224, 208, 0.3) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# CONSTANTS AND TRAINED MODEL LOADING
# --------------------------------------------------

MODEL_PATH = "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}

REQUIRED_FEATURES = [
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
    "Weekend",
]


@st.cache_resource
def load_models():
    models = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODEL_PATH, filename)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")
        models[model_name] = joblib.load(model_path)
    return models


# --------------------------------------------------
# METRIC CALCULATION
# --------------------------------------------------

def evaluate_model(model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": (
            roc_auc_score(target, probabilities)
            if target.nunique() == 2
            else np.nan
        ),
        "Precision": precision_score(
            target, predictions, zero_division=0
        ),
        "Recall": recall_score(
            target, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            target, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(target, predictions),
    }
    return predictions, metrics


def show_metric_cards(metrics):
    first_row = st.columns(3)
    second_row = st.columns(3)

    metric_order = [
        ("Accuracy", first_row[0]),
        ("AUC", first_row[1]),
        ("Precision", first_row[2]),
        ("Recall", second_row[0]),
        ("F1 Score", second_row[1]),
        ("MCC", second_row[2]),
    ]

    for metric_name, column in metric_order:
        value = metrics[metric_name]
        display_value = "N/A" if pd.isna(value) else f"{value:.4f}"
        column.metric(metric_name, display_value)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

logo_col, title_col = st.columns([0.12, 0.88], vertical_alignment="center")

with logo_col:
    st.image("assets/online-store.png", width=65)

with title_col:
    st.title("Online Shopper Purchasing Intention")
st.markdown("**Machine Learning Classification Dashboard**")
st.markdown(
    '<p class="subtle-note">Evaluate five trained classification models '
    'to predict whether an online shopping session results in a purchase.</p>',
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

try:
    trained_models = load_models()
    icon_col, text_col = st.columns([0.06, 0.94])

    with icon_col:
        st.image("assets/done.png", width=24)

    with text_col:
        st.write("All five trained models loaded successfully!")

except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()


# --------------------------------------------------
# DATASET UPLOAD AND VALIDATION
# --------------------------------------------------

st.subheader("Test dataset")
st.caption("Upload the held-out test CSV file. The application uses saved models and does not retrain them.")

uploaded_file = st.file_uploader(
    "Choose test_data.csv",
    type=["csv"],
    help="The CSV must contain the 17 input features and the Revenue target column.",
)

if uploaded_file is None:
    st.info("Upload test_data.csv to begin evaluating the trained models.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the uploaded CSV: {error}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

missing_columns = [
    column
    for column in REQUIRED_FEATURES + ["Revenue"]
    if column not in data.columns
]
if missing_columns:
    st.error("The CSV is missing these required columns: " + ", ".join(missing_columns))
    st.stop()

if data[REQUIRED_FEATURES].isnull().any().any():
    st.error("The input features contain missing values. Please check the CSV.")
    st.stop()

if data["Revenue"].isnull().any():
    st.error("The Revenue target contains missing values.")
    st.stop()

if not data["Revenue"].isin([True, False, 0, 1]).all():
    st.error("Revenue must contain only TRUE/FALSE or 1/0 values.")
    st.stop()

X = data[REQUIRED_FEATURES].copy()
y = data["Revenue"].astype(int)

# Keep preview compact so it does not dominate the page.
summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Test records", f"{len(data):,}")
summary_col2.metric("Purchase sessions", f"{int(y.sum()):,}")
summary_col3.metric("No-purchase sessions", f"{int((y == 0).sum()):,}")

with st.expander("Preview test data"):
    st.dataframe(data.head(5), use_container_width=True, height=190)

st.divider()

# --------------------------------------------------
# EVALUATION MODE
# --------------------------------------------------

st.subheader("Model evaluation")
evaluation_mode = st.radio(
    "Choose what you want to do",
    ["Evaluate one model", "Compare all five models"],
    horizontal=True,
)

if evaluation_mode == "Evaluate one model":
    selected_model = st.selectbox(
        "Select a classification algorithm",
        list(trained_models.keys()),
    )

    evaluate_clicked = st.button(
        "Evaluate selected model",
        type="primary",
        use_container_width=False,
    )

    if evaluate_clicked:
        with st.spinner(f"Evaluating {selected_model}..."):
            model = trained_models[selected_model]
            predictions, metrics = evaluate_model(model, X, y)

        st.markdown(f"#### {selected_model} — performance summary")
        show_metric_cards(metrics)

        st.divider()
        st.markdown("#### Prediction analysis")

        cm = confusion_matrix(y, predictions, labels=[0, 1])
        report = classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["No Purchase", "Purchase"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose().round(3)

        chart_col, report_col = st.columns([0.9, 1.1], gap="large")

        with chart_col:
            st.markdown("**Confusion matrix**")
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                square=True,
                linewidths=0.6,
                linecolor="white",
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=["No Purchase", "Purchase"],
                yticklabels=["No Purchase", "Purchase"],
                ax=ax,
            )
            ax.set_xlabel("Predicted", fontsize=9, labelpad=8)
            ax.set_ylabel("Actual", fontsize=9, labelpad=8)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_title("Actual vs predicted", fontsize=10, pad=10)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        with report_col:
            st.markdown("**Classification report**")
            st.dataframe(
                report_df,
                use_container_width=True,
                height=245,
            )

        st.caption(
            "The confusion matrix shows correct and incorrect predictions. "
            "The classification report provides precision, recall and F1-score for each class."
        )

else:
    st.markdown(
        "Evaluate all five saved models on the same uploaded test dataset."
    )
    compare_clicked = st.button(
        "Compare all five models",
        type="primary",
        use_container_width=False,
    )

    if compare_clicked:
        comparison_rows = []

        with st.spinner("Evaluating all five trained models..."):
            for model_name, model in trained_models.items():
                _, model_metrics = evaluate_model(model, X, y)
                comparison_rows.append(
                    {"Model": model_name, **model_metrics}
                )

        comparison_df = pd.DataFrame(comparison_rows)

        # Rank by F1 score, which balances precision and recall.
        winner_row = comparison_df.loc[comparison_df["F1 Score"].idxmax()]
        st.success(
            f"Highest F1-score: {winner_row['Model']} "
            f"({winner_row['F1 Score']:.4f})"
        )

        st.markdown("#### Model comparison")
        st.dataframe(
            comparison_df.round(4).set_index("Model"),
            use_container_width=True,
        )

        st.markdown("#### F1-score comparison")
        chart_df = comparison_df.set_index("Model")[["F1 Score"]].sort_values(
            "F1 Score", ascending=True
        )
        fig, ax = plt.subplots(figsize=(7, 3.2))
        bars = ax.barh(
            chart_df.index,
            chart_df["F1 Score"],
            color="#5276A7",
            height=0.58,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 Score", fontsize=9)
        ax.set_ylabel("")
        ax.set_title("F1 score by classification model", fontsize=11, pad=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, chart_df["F1 Score"]):
            ax.text(
                min(value + 0.015, 0.96),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.caption(
            "The F1 score is used to highlight the balance between precision and recall. "
            "Review all six metrics when selecting a model for the task."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "BITS Pilani Wilp | M.Tech Artificial Intelligence & Machine Learning | "
    "Machine Learning Assignment 2"
)
