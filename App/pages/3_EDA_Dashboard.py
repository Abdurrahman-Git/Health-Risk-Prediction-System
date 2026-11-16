# -*- coding: utf-8 -*-
"""Step 19 - CDC BRFSS exploratory dashboard (descriptive, not diagnostic)."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
st.title("📊 Population Health EDA Dashboard")
st.caption("CDC BRFSS 2015 - United States survey data; not Sri Lankan prevalence or a diagnostic tool.")

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
FILES = [DATA_DIR / "cdc_dashboard.csv.gz", DATA_DIR / "cdc_dashboard.csv", DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"]

@st.cache_data(show_spinner="Loading survey data...")
def load_csv(path_string):
    path = Path(path_string)
    # Read the compressed CSV as compressed data, and allow Windows-encoded plain CSVs.
    errors = []
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            data = pd.read_csv(path, compression="infer", encoding=encoding, low_memory=False)
            data.columns = data.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
            return data
        except (UnicodeError, UnicodeDecodeError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Unable to decode dataset. " + " | ".join(errors))

path = next((p for p in FILES if p.is_file()), None)
if path is None:
    st.error("Dataset missing. Place cdc_dashboard.csv.gz inside App/data/ (do not extract it).")
    st.code(str(DATA_DIR))
    st.stop()

try:
    df = load_csv(str(path))
except Exception as exc:
    st.error("Could not load dataset. Check that it is a valid CSV or .csv.gz file.")
    st.exception(exc)
    st.stop()

TARGETS = {
    "Diabetes": "Diabetes_binary",
    "Heart disease or heart attack": "HeartDiseaseorAttack",
    "High blood pressure": "HighBP",
}
AGE_LABELS = {1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",6:"45-49",7:"50-54",8:"55-59",9:"60-64",10:"65-69",11:"70-74",12:"75-79",13:"80+"}
REQUIRED = list(TARGETS.values()) + ["Age", "Sex", "BMI", "GenHlth", "PhysActivity"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("Required CDC columns missing: " + ", ".join(missing))
    st.write("Columns available:", list(df.columns))
    st.stop()

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

st.sidebar.header("Filters")
selected_disease = st.sidebar.selectbox("Health condition", list(TARGETS))
target = TARGETS[selected_disease]
sex = st.sidebar.selectbox("Sex (as recorded in survey)", ["All", "Female", "Male"])
ages = sorted(int(x) for x in df["Age"].dropna().unique() if int(x) in AGE_LABELS)
selected_ages = st.sidebar.multiselect("Age groups", ages, default=ages, format_func=lambda x: AGE_LABELS.get(x, str(x)))
filtered = df[df["Age"].isin(selected_ages)].copy()
if sex != "All":
    filtered = filtered[filtered["Sex"] == (0 if sex == "Female" else 1)]
if filtered.empty:
    st.warning("No records match these filters. Choose more age groups or change sex.")
    st.stop()

st.subheader("Survey overview")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Survey records", f"{len(filtered):,}")
c2.metric("Selected condition prevalence", f"{filtered[target].mean()*100:.1f}%")
c3.metric("Mean BMI", f"{filtered['BMI'].mean():.1f}")
c4.metric("Reported physical activity", f"{filtered['PhysActivity'].mean()*100:.1f}%")

st.subheader("1. Prevalence of the three conditions")
prevalence = pd.DataFrame({"Condition":list(TARGETS), "Prevalence (%)":[filtered[c].mean()*100 for c in TARGETS.values()]})
fig = px.bar(prevalence, x="Condition", y="Prevalence (%)", text="Prevalence (%)", title="Share reporting each condition")
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Condition prevalence by age")
age_data = filtered.groupby("Age", as_index=False)[target].mean()
age_data["Prevalence (%)"] = age_data[target]*100
age_data["Age group"] = age_data["Age"].astype(int).map(AGE_LABELS)
fig = px.line(age_data, x="Age group", y="Prevalence (%)", markers=True, title=f"{selected_disease} by age group")
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. BMI distribution by condition status")
bmi = filtered[["BMI", target]].dropna().copy()
bmi["Condition status"] = bmi[target].map({0:"Not reported", 1:"Reported"})
fig = px.histogram(bmi, x="BMI", color="Condition status", nbins=50, barmode="overlay", histnorm="percent", title="Within-group percentage distribution of BMI")
st.plotly_chart(fig, use_container_width=True)

st.subheader("4. Lifestyle and health factors")
factors = {"Smoking":"Smoker", "Physical activity":"PhysActivity", "Fruit consumption":"Fruits", "Vegetable consumption":"Veggies", "High cholesterol":"HighChol", "Heavy alcohol consumption":"HvyAlcoholConsump"}
records = []
for label, col in factors.items():
    if col not in filtered.columns: continue
    for code, status in [(0,"Not reported"),(1,"Reported")]:
        group = filtered[filtered[target] == code]
        if not group.empty:
            records.append({"Factor":label,"Condition status":status,"Percentage (%)":group[col].mean()*100})
if records:
    fig = px.bar(pd.DataFrame(records), x="Factor", y="Percentage (%)", color="Condition status", barmode="group", title="Factor prevalence within condition groups")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No lifestyle-factor columns available.")

st.subheader("5. General health rating")
ratings = {1:"Excellent",2:"Very good",3:"Good",4:"Fair",5:"Poor"}
health = filtered.groupby("GenHlth", as_index=False)[target].mean()
health["Prevalence (%)"] = health[target]*100
health["General health"] = health["GenHlth"].map(ratings)
fig = px.bar(health, x="General health", y="Prevalence (%)", title=f"{selected_disease} by self-rated general health")
st.plotly_chart(fig, use_container_width=True)

st.subheader("6. Correlation heatmap")
corr_cols = [c for c in ["Diabetes_binary", "HeartDiseaseorAttack", "HighBP", "HighChol", "BMI", "Smoker", "Stroke", "PhysActivity", "GenHlth", "MentHlth", "PhysHlth", "Age"] if c in filtered.columns]
corr = filtered[corr_cols].corr()
fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Pearson correlations (associations, not causation)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("7. Download summary")
st.download_button("Download prevalence summary (CSV)", prevalence.to_csv(index=False).encode("utf-8-sig"), file_name="disease_prevalence_summary.csv", mime="text/csv")