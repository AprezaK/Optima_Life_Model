import streamlit as st
import pandas as pd
import pickle
import gzip

st.set_page_config(page_title="Optima Life Retention Predictor", layout="centered")

PRODUCT_KEYS = {
    "Healthy Meals":     "healthy_meals",
    "Daily Fitness":     "daily_fitness",
    "Wellness Tracker":  "wellness_tracker",
    "Mindful Living":    "mindful_living",
    "Premium Health":    "premium_health",
}

@st.cache_resource
def load_artifacts():
    artefacts = {}
    for display_name, key in PRODUCT_KEYS.items():
        with gzip.open(f"churn_model_{key}.pkl.gz", "rb") as f:
            model = pickle.load(f)
        with open(f"churn_encoder_{key}.pkl", "rb") as f:
            encoder = pickle.load(f)
        artefacts[display_name] = {"model": model, "encoder": encoder}
    return artefacts

product_artefacts = load_artifacts()

st.title("Optima Life Retention Predictor")
st.write("Choose a product, then describe a customer's activity and demographics "
         "to predict renewal likelihood and recommended action.")

product = st.selectbox("Product", list(product_artefacts.keys()))

st.subheader("Engagement (past 12 months)")
total_sessions = st.slider("Total sessions", 0, 300, 40)
avg_session_length = st.slider("Average session length (minutes)", 5, 90, 25)
active_quarters = st.slider("Active quarters (out of 4)", 0, 4, 3)

st.subheader("Demographics")
age = st.slider("Age", 20, 60, 36)
tech_comfort = st.slider("Tech comfort score (1 = low, 5 = high)", 1, 5, 3)
tenure_months = st.slider("Tenure (months since first subscription)", 0, 60, 24)

INCOME_OPTIONS    = ["Low", "Medium", "High", "Very High"]
EDUCATION_OPTIONS = ["High School", "Graduate", "Post-Graduate", "Other"]
DEVICE_OPTIONS    = ["Desktop-only", "Mobile-only", "Multi-device"]

st.subheader("Customer profile")
income_level = st.radio("Income Level", INCOME_OPTIONS, horizontal=True)
education    = st.radio("Education", EDUCATION_OPTIONS, horizontal=True)
device_type  = st.radio("Device Type", DEVICE_OPTIONS, horizontal=True)

if st.button("Assess Churn Risk", type="primary"):
    art = product_artefacts[product]
    model, encoder = art["model"], art["encoder"]

    gross_total_session_length      = total_sessions * avg_session_length
    avg_sessions_per_active_quarter = (
        total_sessions / active_quarters if active_quarters > 0 else 0
    )
    tech_engagement_interaction     = tech_comfort * avg_sessions_per_active_quarter

    raw_cat = pd.DataFrame([{
        "INCOME_LEVEL": income_level,
        "EDUCATION":    education,
        "DEVICE_TYPE":  device_type,
    }])
    encoded = encoder.transform(raw_cat)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    numeric_df = pd.DataFrame([{
        "TOTAL_NUM_SESSIONS":              total_sessions,
        "GROSS_TOTAL_SESSION_LENGTH":      gross_total_session_length,
        "ACTIVE_QUARTERS":                 active_quarters,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": avg_sessions_per_active_quarter,
        "AVG_SESSION_LENGTH":              avg_session_length,
        "AGE":                             age,
        "TECH_COMFORT_SCORE":              tech_comfort,
        "TECH_ENGAGEMENT_INTERACTION":     tech_engagement_interaction,
        "TENURE_MONTHS":                   tenure_months,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)
    input_df = input_df[model.feature_names_in_]

    prob_renew = model.predict_proba(input_df)[0][1]
    prob_churn = 1 - prob_renew

    if prob_churn >= 0.30:
        risk_label = "HIGH"
        risk_action = "high-touch retention recommended"
    elif prob_churn >= 0.10:
        risk_label = "MEDIUM"
        risk_action = "automated engagement recommended"
    else:
        risk_label = "LOW"
        risk_action = "monitor only"

    st.markdown("### Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn probability", f"{prob_churn:.1%}")
    with col2:
        st.metric("Renewal probability", f"{prob_renew:.1%}")

    st.info(f"**Risk tier: {risk_label}** - {risk_action}")
