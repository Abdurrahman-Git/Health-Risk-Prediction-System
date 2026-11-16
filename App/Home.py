import streamlit as st

st.set_page_config(
    page_title="Health Risk Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Multi-Disease Health Risk Prediction System")
st.markdown("### Explainable AI-Powered Personal Health Intelligence")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset Records", "253,680", "CDC BRFSS 2015")
with col2:
    st.metric("Diseases Predicted", "3", "Diabetes · Heart · BP")
with col3:
    st.metric("ML Models Trained", "9", "LR · RF · XGBoost")

st.divider()
st.subheader("About This System")
st.write("""
This system uses machine learning trained on 253,680 real health survey
records from the US Centers for Disease Control and Prevention (CDC) to
simultaneously predict your risk of three chronic diseases:

- 🩺 **Diabetes Mellitus**
- ❤️ **Heart Disease / Heart Attack**
- 🫀 **High Blood Pressure (Hypertension)**

Every prediction is accompanied by a **SHAP explanation** — a personalised
chart showing exactly which lifestyle factors are driving your risk up or down.
""")

st.info("👈 Click **Risk Predictor** in the sidebar to check your health risk.")

st.divider()
st.caption("Final Year Project · BSc (Hons) Data Science · February 2026 · ESOFT Metro Campus")