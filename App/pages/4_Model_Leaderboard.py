
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Model Leaderboard",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Machine Learning Model Leaderboard")
st.caption(
    "Model comparison for Diabetes, Heart Disease "
    "and High Blood Pressure"
)

st.info(
    "Evaluation metrics must come from held-out test results. "
    "Saved models alone do not establish ROC-AUC, "
    "precision, recall or F1."
)

APP = Path(__file__).resolve().parent.parent
MODEL_DIR = APP / "models"

CONFIG = [
    ("Diabetes", "best_model_diabetes.pkl"),
    ("Heart Disease", "best_model_heart.pkl"),
    ("High Blood Pressure", "best_model_bp.pkl")
]

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.subheader("Saved Best-Performing Models")

records = []

for disease, filename in CONFIG:
    path = MODEL_DIR / filename

    if path.exists():
        try:
            model = load_model(str(path))
            records.append({
                "Disease": disease,
                "Saved model type": type(model).__name__,
                "Artifact": filename,
                "Status": "Loaded"
            })
        except Exception as exc:
            records.append({
                "Disease": disease,
                "Saved model type": "Unknown",
                "Artifact": filename,
                "Status": "Load failed: " + str(exc)[:100]
            })
    else:
        records.append({
            "Disease": disease,
            "Saved model type": "Missing",
            "Artifact": filename,
            "Status": "Not found"
        })

st.dataframe(
    pd.DataFrame(records),
    use_container_width=True,
    hide_index=True
)

st.subheader("Verified Model Evaluation Results")

st.write(
    "To compare Logistic Regression, Random Forest and "
    "XGBoost for all three diseases, export the verified "
    "evaluation metrics from your training notebook "
    "to App/data/model_metrics.csv."
)

metrics_path = APP / "data" / "model_metrics.csv"

required = [
    "Disease", "Model", "ROC_AUC",
    "Precision", "Recall", "F1"
]

if metrics_path.exists():
    try:
        metrics = pd.read_csv(
            metrics_path,
            encoding="utf-8-sig"
        )

        missing = [
            c for c in required
            if c not in metrics.columns
        ]

        if missing:
            st.error(
                "Missing columns: " + ", ".join(missing)
            )
        else:
            for col in required[2:]:
                metrics[col] = pd.to_numeric(
                    metrics[col],
                    errors="coerce"
                )

            selected = st.selectbox(
                "Choose Disease",
                sorted(metrics["Disease"].dropna().unique())
            )

            subset = metrics.loc[
                metrics["Disease"] == selected
            ].copy()

            st.dataframe(
                subset[required].sort_values(
                    "ROC_AUC",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            chart_data = (
                subset.dropna(subset=["ROC_AUC"])
                .set_index("Model")["ROC_AUC"]
            )

            if not chart_data.empty:
                st.bar_chart(chart_data)

            st.download_button(
                "Download Evaluation Metrics",
                metrics.to_csv(index=False).encode("utf-8"),
                "verified_model_metrics.csv",
                "text/csv"
            )

    except Exception as exc:
        st.error("Could not read metrics CSV: " + str(exc))

else:
    st.warning(
        "model_metrics.csv is not present yet. "
        "Verified evaluation results must be imported."
    )

st.subheader("Evaluation Methodology")

st.markdown("""
- Use an untouched test set for each disease.
- Fit scaling and SMOTE on training data only.
- Evaluate ROC-AUC, precision, recall and F1-score.
- Examine confusion matrices and class imbalance.
- Check probability calibration before interpreting
  prediction percentages as real-world risk.
- The CDC BRFSS 2015 dataset is from the United States.
  Clinical validity for Sri Lankan users is not established.
""")
