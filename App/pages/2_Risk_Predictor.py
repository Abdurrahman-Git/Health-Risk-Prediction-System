import streamlit as st
import joblib, shap, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

st.set_page_config(page_title="Risk Predictor", page_icon="🔮", layout="wide")

# ── Load assets ───────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    return {
        'models': {
            'diabetes': joblib.load(os.path.join(base,'best_model_diabetes.pkl')),
            'heart':    joblib.load(os.path.join(base,'best_model_heart.pkl')),
            'bp':       joblib.load(os.path.join(base,'best_model_bp.pkl')),
        },
        'scalers': {
            'diabetes': joblib.load(os.path.join(base,'scaler_diabetes.pkl')),
            'heart':    joblib.load(os.path.join(base,'scaler_heart.pkl')),
            'bp':       joblib.load(os.path.join(base,'scaler_bp.pkl')),
        },
        'explainers': {
            'diabetes': joblib.load(os.path.join(base,'explainer_diabetes.pkl')),
            'heart':    joblib.load(os.path.join(base,'explainer_heart.pkl')),
            'bp':       joblib.load(os.path.join(base,'explainer_bp.pkl')),
        },
    }

assets = load_assets()
models    = assets['models']
scalers   = assets['scalers']
explainers= assets['explainers']

FEAT_DIAB  = ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
               'PhysActivity','Fruits','Veggies','HvyAlcoholConsump',
               'AnyHealthcare','NoDocbcCost','GenHlth','MentHlth',
               'PhysHlth','DiffWalk','Sex','Age','Education','Income',
               'HeartDiseaseorAttack']
FEAT_HEART = [f for f in FEAT_DIAB if f != 'HeartDiseaseorAttack']
FEAT_BP    = [f for f in FEAT_DIAB if f != 'HighBP']

# ── Risk bar helper ───────────────────────────────────────────
def risk_bar(label, pct, color):
    bar_html = f"""
    

      
{label}
{pct:.1f}%

      

        

      

    
"""
    st.markdown(bar_html, unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────
st.title("🔮 Personal Health Risk Predictor")
st.markdown("""
Enter your health details below and click **Predict My Health Risk**.
The system will assess your risk for **three chronic diseases simultaneously**
and explain which factors are driving your risk.
""")
st.divider()

# ── Input form ────────────────────────────────────────────────
st.subheader("📋 Your Health Information")

with st.form("health_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Personal Details**")
        age = st.selectbox("Age Group", options=list(range(1,14)),
              format_func=lambda x:['18–24','25–29','30–34','35–39',
              '40–44','45–49','50–54','55–59','60–64','65–69',
              '70–74','75–79','80+'][x-1])

        # FIX: number_input instead of slider — avoids the frozen slider bug
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=165.0, step=1.0)
        weight_kg = st.number_input("Weight (kg)", min_value=25.0, max_value=250.0, value=65.0, step=0.5)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        st.caption(f"Calculated BMI: **{bmi:.1f}** kg/m²")

        sex = st.selectbox("Sex", options=[0,1],
              format_func=lambda x:'Female' if x==0 else 'Male')

        education = st.selectbox("Highest Education Level",
              options=list(range(1,7)),
              format_func=lambda x:['Never attended school','Elementary school',
              'Some high school','High school graduate',
              'Some college','College graduate'][x-1])

    with c2:
        st.markdown("**Health Conditions**")
        gen_hlth = st.selectbox("How would you rate your general health?",
                   options=[1,2,3,4,5],
                   format_func=lambda x:['Excellent','Very Good',
                   'Good','Fair','Poor'][x-1])

        high_bp   = st.radio("Do you have High Blood Pressure?",
                   options=[0,1], horizontal=True,
                   format_func=lambda x:'No' if x==0 else 'Yes')

        high_chol = st.radio("Do you have High Cholesterol?",
                   options=[0,1], horizontal=True,
                   format_func=lambda x:'No' if x==0 else 'Yes')

        chol_check = st.radio("Had cholesterol check in past 5 years?",
                   options=[1,0], horizontal=True,
                   format_func=lambda x:'Yes' if x==1 else 'No')

        stroke    = st.radio("Have you ever had a Stroke?",
                   options=[0,1], horizontal=True,
                   format_func=lambda x:'No' if x==0 else 'Yes')

        heart_history = st.radio("Ever told you had a heart attack or coronary heart disease?",
                   options=[0,1], horizontal=True, format_func=lambda x: "No" if x==0 else "Yes",
                   help="Used by the diabetes model; not the same as the heart-disease outcome being predicted.")
        mental_days = st.number_input("Days mental health was not good (last 30 days)",
                   min_value=0, max_value=30, value=0)
        physical_days = st.number_input("Days physical health was not good (last 30 days)",
                   min_value=0, max_value=30, value=0)

        diff_walk = st.radio("Difficulty walking or climbing stairs?",
                   options=[0,1], horizontal=True,
                   format_func=lambda x:'No' if x==0 else 'Yes')

    with c3:
        st.markdown("**Lifestyle Habits**")
        phys_act  = st.radio("Physically active in the past 30 days?",
                   options=[1,0], horizontal=True,
                   format_func=lambda x:'Yes' if x==1 else 'No')

        smoker    = st.radio("Have you smoked 100+ cigarettes in your life?",
                   options=[0,1], horizontal=True,
                   format_func=lambda x:'No' if x==0 else 'Yes')

        fruits    = st.radio("Do you eat fruit at least once per day?",
                   options=[1,0], horizontal=True,
                   format_func=lambda x:'Yes' if x==1 else 'No')

        veggies   = st.radio("Do you eat vegetables at least once per day?",
                   options=[1,0], horizontal=True,
                   format_func=lambda x:'Yes' if x==1 else 'No')

        alc       = st.radio("Heavy alcohol consumption?",
                   options=[0,1], horizontal=True,
                   format_func=lambda x:'No' if x==0 else 'Yes',
                   help="Men: 14+ drinks/week | Women: 7+ drinks/week")

        no_doc_cost = st.radio("In the past 12 months, did cost prevent you from seeing a doctor?",
                   options=[0,1], horizontal=True, format_func=lambda x: "No" if x==0 else "Yes")

        healthcare= st.radio("Do you have any healthcare coverage?",
                   options=[1,0], horizontal=True,
                   format_func=lambda x:'Yes' if x==1 else 'No')

    st.divider()
    st.subheader("🧪 Optional Clinical Measurements")
    st.caption("Enter measurements only if you know them from a test or a reliable device. Leave unknown values at zero or 'Not sure'. These measurements are displayed separately and DO NOT change the trained ML predictions.")
    lab1, lab2, lab3 = st.columns(3)
    with lab1:
        fasting_glucose = st.number_input("Fasting blood glucose (mg/dL) — 0 if unknown", min_value=0, max_value=600, value=0, step=1)
        hba1c = st.number_input("HbA1c (%) — 0 if unknown", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        family_diabetes = st.selectbox("Family history of diabetes", ["Not sure", "Yes", "No"])
    with lab2:
        systolic = st.number_input("Systolic blood pressure (mmHg) — 0 if unknown", min_value=0, max_value=300, value=0, step=1)
        diastolic = st.number_input("Diastolic blood pressure (mmHg) — 0 if unknown", min_value=0, max_value=200, value=0, step=1)
        family_heart = st.selectbox("Family history of heart disease", ["Not sure", "Yes", "No"])
    with lab3:
        total_cholesterol = st.number_input("Total cholesterol (mg/dL) — 0 if unknown", min_value=0, max_value=600, value=0, step=1)
        resting_hr = st.number_input("Resting heart rate (beats/min) — 0 if unknown", min_value=0, max_value=250, value=0, step=1)
        st.caption("Use readings from a healthcare professional or a reliable device; do not guess.")

    st.divider()
    submitted = st.form_submit_button(
        "🔍  Predict My Health Risk",
        type="primary", use_container_width=True)

# ── Prediction & Results ──────────────────────────────────────
if submitted:
    # Original US CDC income feature is retained as a fixed reference code.
    # It is NOT an equivalent Sri Lankan income category; this limits model validity.
    income = 4
    # Build full feature vector in FEAT_DIAB order
    inp_vals = [high_bp, high_chol, chol_check, bmi, smoker, stroke,
                phys_act, fruits, veggies, alc, healthcare, no_doc_cost,
                gen_hlth, mental_days, physical_days, diff_walk, sex, age, education, income, heart_history]

    inp_d = pd.DataFrame([inp_vals], columns=FEAT_DIAB)
    inp_h = pd.DataFrame([[v for f,v in zip(FEAT_DIAB,inp_vals)
                           if f in FEAT_HEART]], columns=FEAT_HEART)
    inp_b = pd.DataFrame([[v for f,v in zip(FEAT_DIAB,inp_vals)
                           if f in FEAT_BP]],   columns=FEAT_BP)

    inp_d_sc = scalers['diabetes'].transform(inp_d)
    inp_h_sc = scalers['heart'].transform(inp_h)
    inp_b_sc = scalers['bp'].transform(inp_b)

    # Get probabilities (0.0 to 1.0) — convert to percentage
    p_d = float(models['diabetes'].predict_proba(inp_d_sc)[0][1]) * 100
    p_h = float(models['heart'].predict_proba(inp_h_sc)[0][1])    * 100
    p_b = float(models['bp'].predict_proba(inp_b_sc)[0][1])       * 100

    # Overall health score
    overall = (p_d * 0.40 + p_h * 0.40 + p_b * 0.20)

    # Risk tier
    def get_tier(score):
        if score < 25:   return "LOW RISK",      "🟢", "#1E8449"
        elif score < 55: return "MODERATE RISK",  "🟡", "#D68910"
        else:            return "HIGH RISK",       "🔴", "#C0392B"

    tier_d, icon_d, col_d = get_tier(p_d)
    tier_h, icon_h, col_h = get_tier(p_h)
    tier_b, icon_b, col_b = get_tier(p_b)
    tier_o, icon_o, col_o = get_tier(overall)

    st.divider()
    st.subheader("📊 Your Health Risk Results")
    st.info("**Research prototype — not a medical diagnosis.** The models were trained on US CDC survey data and have not been validated for Sri Lankan patients. The original US income feature is held at a fixed reference code (4) for every user. This is a limitation, not a Sri Lankan income conversion. Optional clinical measurements are not used by these trained models. Scores and low/moderate/high bands are experimental, not clinically calibrated.")

    # Overall score — big display
    st.markdown(f"""
    

      


        OVERALL HEALTH RISK SCORE


      


        {overall:.1f} / 100


      


        {icon_o} {tier_o}


    
""", unsafe_allow_html=True)

    # Individual disease bars
    st.markdown("**Individual Disease Risk Scores**")
    risk_bar(f"🩺 Diabetes Risk  —  {icon_d} {tier_d}",      p_d, col_d)
    risk_bar(f"❤️  Heart Disease Risk  —  {icon_h} {tier_h}", p_h, col_h)
    risk_bar(f"🫀 High Blood Pressure Risk  —  {icon_b} {tier_b}", p_b, col_b)

    # Colour legend
    st.caption("Experimental score bands: Low <25 | Moderate 25–<55 | High ≥55. These are not clinical thresholds or validated probabilities.")
    st.divider()

    # SHAP explanation
    st.subheader("🧠 Why? — SHAP Explanation (Diabetes Model)")
    st.write("This research explanation estimates which model inputs are "
             "**increasing** (red → right) or **decreasing** (blue → left) "
             "your diabetes risk.")

    try:
        exp_d = explainers['diabetes']
        sv = exp_d.shap_values(pd.DataFrame(inp_d_sc, columns=FEAT_DIAB))
        sv_single = sv[1][0] if isinstance(sv, list) else sv[0]
        base_val  = (exp_d.expected_value[1]
                     if hasattr(exp_d.expected_value, '__len__')
                     else exp_d.expected_value)
        explanation = shap.Explanation(
            values        = sv_single,
            base_values   = base_val,
            data          = inp_d_sc[0],
            feature_names = FEAT_DIAB)
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP chart could not render: {e}")

    st.divider()

    # Clinical measurements are descriptive only: the existing models were not trained on these fields.
    st.subheader("🧪 Your Optional Clinical Measurements")
    measurements = []
    for label, value, unit in [
        ("Fasting glucose", fasting_glucose, "mg/dL"),
        ("HbA1c", hba1c, "%"),
        ("Systolic blood pressure", systolic, "mmHg"),
        ("Diastolic blood pressure", diastolic, "mmHg"),
        ("Total cholesterol", total_cholesterol, "mg/dL"),
        ("Resting heart rate", resting_hr, "beats/min"),
    ]:
        if value > 0:
            measurements.append({"Measurement": label, "Reported value": f"{value:g} {unit}"})
    if measurements:
        st.dataframe(pd.DataFrame(measurements), hide_index=True, use_container_width=True)
    else:
        st.caption("No clinical measurements entered.")
    st.write(f"**Family history:** Diabetes — {family_diabetes}; Heart disease — {family_heart}.")
    if (systolic > 0) != (diastolic > 0):
        st.warning("For blood pressure, enter both systolic and diastolic readings for a complete measurement.")
    st.info("These measurements and family-history answers are recorded for your review only. They do not change the diabetes, heart disease, hypertension, or overall ML scores. Ask a healthcare professional to interpret test results; a single reading cannot establish a diagnosis.")
    if systolic >= 180 or diastolic >= 120:
        st.error("A blood pressure reading at or above 180 systolic or 120 diastolic can be serious. Repeat the measurement after a brief rest and seek urgent medical advice. If you have chest pain, severe headache, shortness of breath, weakness, or vision changes, seek emergency care now.")
    st.divider()

    # Personalised recommendations
    st.subheader("💡 Personalised Recommendations")
    recs = []
    if p_d >= 55: recs.append("🩺 **Diabetes:** The model score is elevated. Consider discussing blood glucose or HbA1c testing with a clinician.")
    elif p_d >= 25: recs.append("🩺 **Diabetes:** Consider a routine health review and discuss physical activity and nutrition with a clinician.")
    else: recs.append("🩺 **Diabetes:** A lower model score does not rule out diabetes; follow routine screening advice.")

    if p_h >= 55: recs.append("❤️ **Heart Disease:** Consider a cardiovascular risk review with a qualified clinician.")
    elif p_h >= 25: recs.append("❤️ **Heart Disease:** Discuss cardiovascular risk factors such as smoking and activity with a clinician.")
    else: recs.append("❤️ **Heart Disease:** A lower model score cannot rule out heart disease.")

    if p_b >= 55: recs.append("🫀 **Blood Pressure:** Check your blood pressure using a validated cuff and consult a clinician if elevated.")
    elif p_b >= 25: recs.append("🫀 **Blood Pressure:** Consider routine blood pressure measurement and healthy lifestyle habits.")
    else: recs.append("🫀 **Blood Pressure:** Only a blood pressure measurement can establish whether BP is elevated.")

    if overall >= 25:
        recs.append("⚕️ **Overall:** Discuss any health concerns with a healthcare professional; this experimental score does not determine urgency.")

    for r in recs:
        st.markdown(f"- {r}")

    st.info("ℹ️ **Important:** This tool is for awareness and screening purposes only. "
            "It does not replace professional medical advice, diagnosis, or treatment. "
            "Always consult a qualified healthcare professional.")