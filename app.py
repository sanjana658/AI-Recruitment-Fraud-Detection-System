import streamlit as st
import pickle
import pandas as pd
import os
from datetime import datetime

# =====================================
# Load Model & Vectorizer
# =====================================

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# =====================================
# Risk Score Function
# =====================================

def calculate_risk_score(text):

    risk_score = 0

    risk_rules = {
        "registration": 20,
        "fee": 20,
        "payment": 15,
        "investment": 20,
        "earn": 10,
        "guaranteed": 15,
        "no experience": 10,
        "urgent": 10,
        "limited seats": 10,
        "quick money": 20,
        "easy money": 20,
        "immediate": 10,
        "work from home": 10
    }

    text = text.lower()

    for word, score in risk_rules.items():
        if word in text:
            risk_score += score

    return min(risk_score, 100)

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="AI Recruitment Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

# =====================================
# Header
# =====================================

st.title("🛡️ AI Recruitment Fraud Detection System")

st.caption(
    "NLP + Machine Learning + Risk Intelligence"
)

st.markdown("""
Analyze job postings and identify potentially fraudulent opportunities
using Machine Learning and Risk Analysis.
""")

# =====================================
# Inputs
# =====================================

company_name = st.text_input(
    "Company Name",
    placeholder="Enter company name"
)

job_text = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Paste the complete job posting here..."
)

# =====================================
# Analyze Button
# =====================================

if st.button("Analyze Job Posting"):

    if job_text.strip() == "":
        st.warning("Please enter a job description.")
        st.stop()

    # ==========================
    # Prediction
    # ==========================

    job_vector = vectorizer.transform([job_text])

    probabilities = model.predict_proba(job_vector)[0]

    real_probability = probabilities[0] * 100
    fake_probability = probabilities[1] * 100

    # ==========================
    # Keyword Detection
    # ==========================

    suspicious_keywords = [
        "registration",
        "fee",
        "payment",
        "investment",
        "earn",
        "work from home",
        "immediate",
        "guaranteed",
        "income",
        "no experience",
        "urgent",
        "limited seats",
        "quick money",
        "easy money"
    ]

    found_keywords = []

    text_lower = job_text.lower()

    for keyword in suspicious_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)

    # ==========================
    # Risk Calculations
    # ==========================

    keyword_risk = calculate_risk_score(job_text)

    fraud_score = int(
        (fake_probability * 0.8) +
        (keyword_risk * 0.2)
    )

    fraud_score = min(fraud_score, 100)

    safety_score = max(
        0,
        min(
            100,
            int(real_probability - (keyword_risk * 0.2))
        )
    )

    st.divider()

    # ==========================
    # Analysis Result
    # ==========================

    st.subheader("📋 Analysis Result")

    if fake_probability >= 70:

        st.error(
            f"⚠️ High Risk Job ({fake_probability:.2f}% confidence)"
        )

        prediction_label = "High Risk"

    elif fake_probability >= 40:

        st.warning(
            f"🟡 Suspicious Job - Manual Review Recommended ({fake_probability:.2f}% confidence)"
        )

        prediction_label = "Suspicious"

    else:

        st.success(
            f"✅ Likely Legitimate ({real_probability:.2f}% confidence)"
        )

        prediction_label = "Legitimate"

    # ==========================
    # Safety Score
    # ==========================

    st.subheader("🛡️ Safety Score")

    st.progress(safety_score)

    st.write(f"Safety Score: {safety_score}/100")

    # ==========================
    # Fraud Score
    # ==========================

    st.subheader("🚨 Fraud Risk Score")

    st.progress(fraud_score)

    st.write(f"Fraud Risk Score: {fraud_score}/100")

    # ==========================
    # Suspicious Keywords
    # ==========================

    st.subheader("🚨 Suspicious Keywords")

    if found_keywords:

        st.write("Detected Risk Keywords:")

        for keyword in found_keywords:
            st.write(f"• {keyword}")

    else:

        st.success("No suspicious keywords detected.")

    # ==========================
    # Explainability
    # ==========================

    st.subheader("🧠 Why was this flagged?")

    if found_keywords:

        for keyword in found_keywords:
            st.write(
                f"• Contains suspicious term: {keyword}"
            )

    else:

        st.write(
            "No major suspicious keywords found."
        )

    # ==========================
    # Save History
    # ==========================

    record = pd.DataFrame(
        [
            {
                "Date": datetime.now(),
                "Company": company_name,
                "Prediction": prediction_label,
                "Fraud Score": fraud_score
            }
        ]
    )

    history_file = "history.csv"

    if os.path.exists(history_file):

        record.to_csv(
            history_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        record.to_csv(
            history_file,
            index=False
        )

    # ==========================
    # Risk Analysis
    # ==========================

    st.subheader("📊 Risk Analysis")

    risk_data = pd.DataFrame(
        {
            "Metric": [
                "Fake Probability",
                "Real Probability",
                "Keyword Risk",
                "Fraud Score"
            ],
            "Score": [
                round(fake_probability, 2),
                round(real_probability, 2),
                keyword_risk,
                fraud_score
            ]
        }
    )

    st.bar_chart(
        risk_data.set_index("Metric")
    )

    # ==========================
    # Model Performance
    # ==========================

    st.subheader("📈 Model Performance")

    performance_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Recall",
                "Precision",
                "F1 Score"
            ],
            "Score": [
                96.78,
                90.00,
                62.00,
                73.00
            ]
        }
    )

    st.bar_chart(
        performance_data.set_index("Metric")
    )

    # ==========================
    # Summary
    # ==========================

    st.subheader("📝 Summary")

    if fake_probability >= 70:

        st.error("""
High Risk Posting Detected

Recommendations:
• Verify company website
• Avoid registration fees
• Confirm recruiter identity
• Check LinkedIn presence
• Research company reviews
""")

    elif fake_probability >= 40:

        st.warning("""
Suspicious Posting

Manual verification recommended.

• Verify company details
• Contact official HR
• Research online reviews
""")

    else:

        st.success("""
Likely Legitimate Posting

The job appears safe based on ML analysis and risk assessment.

Continue normal verification before applying.
""")

# =====================================
# History Dashboard
# =====================================

st.divider()

st.subheader("📈 Previous Analyses")

if os.path.exists("history.csv"):

    history = pd.read_csv("history.csv")

    st.dataframe(
        history.tail(10),
        use_container_width=True
    )

else:

    st.info(
        "No analysis history available yet."
    )

# =====================================
# Footer
# =====================================

st.divider()

st.caption(
    "Built using Python, Scikit-Learn, TF-IDF, Logistic Regression / Random Forest and Streamlit."
)