import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Resume Shortlisting Predictor", page_icon="📄")

st.title("📄 Resume Shortlisting Prediction")
st.caption("Educational ML case study — decision support, not an automated hiring decision.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python train.py")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())

st.info(
    f"Selected model: **{metadata['best_model']}**  \n"
    f"Selection rule: {metadata['selection_rule']}"
)

col1, col2 = st.columns(2)

with col1:
    experience = st.number_input("Years of experience", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    education = st.selectbox("Education level", ["Bachelor's", "Master's", "PhD"])
    skills = st.number_input("Relevant skill count", min_value=0, max_value=30, value=6, step=1)

with col2:
    certifications = st.number_input("Certification count", min_value=0, max_value=20, value=2, step=1)
    industry = st.selectbox(
        "Previous industry",
        ["IT", "Finance", "Healthcare", "Retail", "Manufacturing"]
    )

if st.button("Predict Shortlisting", type="primary"):
    candidate = pd.DataFrame([{
        "experience_years": experience,
        "education_level": education,
        "relevant_skill_count": skills,
        "certification_count": certifications,
        "previous_industry": industry
    }])

    prediction = int(model.predict(candidate)[0])
    probability = float(model.predict_proba(candidate)[0, 1])

    if prediction == 1:
        st.success("Recommendation: SHORTLIST")
    else:
        st.warning("Recommendation: REVIEW FURTHER")

    st.metric("Model confidence for shortlist", f"{probability * 100:.1f}%")
    st.write("### Prediction details")
    st.json({
        "selected_model": metadata["best_model"],
        "predicted_class": prediction,
        "shortlist_probability": round(probability, 4),
        "note": "Confidence is model probability, not certainty."
    })

st.divider()
st.subheader("Important limitation")
st.write(
    "This project uses synthetic data. It is intended to demonstrate ML workflow, "
    "not to make real hiring decisions. Human review, validation on representative "
    "real-world data, documentation, monitoring and legal/ethical review would be "
    "required before any real deployment."
)
