
# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(
    page_title="Health Recommendations",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Health Recommendations")

st.caption(
    "General health education; not a medical diagnosis"
)

st.warning(
    "This page does not change your machine learning "
    "predictions. Optional clinical measurements are "
    "not inputs to the saved CDC-trained models."
)

st.subheader("Optional Health Information")

with st.form("health_education"):

    col1, col2 = st.columns(2)

    with col1:
        glucose = st.number_input(
            "Fasting glucose (mg/dL) - 0 if unknown",
            min_value=0.0,
            max_value=600.0,
            value=0.0
        )

        a1c = st.number_input(
            "HbA1c (%) - 0 if unknown",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1
        )

        sbp = st.number_input(
            "Systolic BP (mmHg) - 0 if unknown",
            min_value=0,
            max_value=300,
            value=0
        )

        dbp = st.number_input(
            "Diastolic BP (mmHg) - 0 if unknown",
            min_value=0,
            max_value=200,
            value=0
        )

    with col2:
        cholesterol = st.number_input(
            "Total cholesterol (mg/dL) - 0 if unknown",
            min_value=0,
            max_value=600,
            value=0
        )

        pulse = st.number_input(
            "Resting pulse (beats/min) - 0 if unknown",
            min_value=0,
            max_value=250,
            value=0
        )

        family_diabetes = st.checkbox(
            "Family history of diabetes"
        )

        family_heart = st.checkbox(
            "Family history of heart disease"
        )

    submitted = st.form_submit_button(
        "Show Educational Guidance"
    )

if submitted:

    st.subheader("Your Measurements")

    measurements = {
        "Fasting glucose (mg/dL)": glucose,
        "HbA1c (%)": a1c,
        "Systolic BP (mmHg)": sbp,
        "Diastolic BP (mmHg)": dbp,
        "Total cholesterol (mg/dL)": cholesterol,
        "Resting pulse (beats/min)": pulse
    }

    for name, value in measurements.items():
        if value:
            st.write(f"**{name}:** {value:g}")

    if sbp and dbp:

        if sbp >= 180 or dbp >= 120:
            st.error(
                "Very high blood pressure reading. "
                "Repeat the measurement after a minute "
                "and seek urgent medical evaluation. "
                "If you have chest pain, breathing "
                "difficulty, weakness, vision changes "
                "or other concerning symptoms, "
                "seek emergency care immediately."
            )

        elif sbp >= 130 or dbp >= 80:
            st.info(
                "Your blood pressure may be elevated. "
                "Repeat measurements under appropriate "
                "conditions and discuss persistent "
                "readings with a clinician."
            )

    elif sbp or dbp:
        st.info(
            "Enter both systolic and diastolic values "
            "for a meaningful blood pressure reading."
        )

    if glucose or a1c:
        st.info(
            "Glucose and HbA1c results require "
            "interpretation based on test type, "
            "symptoms and medical history. "
            "Discuss abnormal results with a clinician."
        )

    if cholesterol:
        st.info(
            "Total cholesterol alone cannot establish "
            "cardiovascular risk. A full lipid panel "
            "and other risk factors may be needed."
        )

    if pulse:
        st.info(
            "Resting heart rate varies with fitness, "
            "medications and illness. Seek medical "
            "advice for persistent unusual readings "
            "or associated symptoms."
        )

    if family_diabetes or family_heart:
        st.info(
            "Family history may influence screening "
            "decisions. Discuss it with your clinician."
        )

    st.success(
        "These measurements are displayed for "
        "educational purposes only. They were not "
        "passed into the machine learning models."
    )

st.subheader("Everyday Health Recommendations")

st.markdown("""
**Healthy Eating**
- Choose balanced meals with vegetables and whole grains.
- Limit excessive salt and added sugar.
- Maintain appropriate portion sizes.

**Physical Activity**
- Build regular physical activity gradually.
- Choose exercises suitable for your health and mobility.

**Heart Health**
- Avoid smoking and secondhand smoke.
- Monitor blood pressure when advised.
- Attend appropriate medical checkups.

**Diabetes Prevention**
- Maintain a healthy lifestyle.
- Discuss blood glucose screening if you have risk factors.
- Follow professional guidance for abnormal results.

**General Wellbeing**
- Prioritize sufficient sleep.
- Manage stress.
- Seek medical attention for concerning symptoms.
""")

st.caption(
    "Research prototype trained using US CDC survey data. "
    "Not clinically validated for Sri Lankan patients."
)
