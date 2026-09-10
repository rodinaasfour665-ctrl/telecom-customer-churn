import streamlit as st
import numpy as np
import pickle

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Churn Radar",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "churn_prediction_model.pkl"

# Exact column order the model was trained on (from the training notebook)
COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]

# StandardScaler parameters, refit on the original training split
# (mean/std per column, in the same order as COLUMNS)
MEAN = [
    0.5028399006034788, 0.16329428470003549, 0.48438054668086616,
    0.29801206957756476, 32.48509052183174, 0.9007809726659567,
    0.5912318068867589, 64.92996095136671, 2301.3190273340433,
    0.09921902733404331, 0.42421015264465745, 0.44071707490237844,
    0.2154774582889599, 0.2154774582889599, 0.2880724174653887,
    0.2154774582889599, 0.3510827121050763, 0.2154774582889599,
    0.3457578984735534, 0.2154774582889599, 0.2926872559460419,
    0.2154774582889599, 0.38942137025204115, 0.2154774582889599,
    0.39101881434149804, 0.20820021299254526, 0.24121405750798722,
    0.21529996450124245, 0.33564075257365994, 0.22825701100461485,
]

SCALE = [
    0.49999193489951654, 0.3696339558053876, 0.4997559731288975,
    0.4573848226205823, 24.56656301538815, 0.2989558695676164,
    0.4916063032673372, 30.135430576006627, 2277.6070530763936,
    0.2989558695676164, 0.4942225197599295, 0.49647309573819787,
    0.41115316277305886, 0.41115316277305886, 0.45286499065509084,
    0.41115316277305886, 0.47730874847002375, 0.41115316277305886,
    0.47561473286338113, 0.41115316277305886, 0.4549960726784552,
    0.41115316277305886, 0.4876190794493831, 0.41115316277305886,
    0.48797858679707157, 0.4060207929434206, 0.4278198639246671,
    0.411030278430928, 0.4722139745766183, 0.41970912300288876,
]

# ============================================================
# STYLE — dark "control room" theme, teal signal / coral alert
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .headline { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #10222a 0%, #071016 45%, #050a0d 100%);
        color: #e7f3f1;
    }

    section[data-testid="stSidebar"] { background: #0a1a20; }

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(0,194,168,0.16), rgba(255,90,95,0.08));
        border: 1px solid rgba(0,194,168,0.25);
        margin-bottom: 1.6rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: #f4fffd; }
    .hero p { margin: 0.35rem 0 0 0; color: #9fc9c2; font-size: 0.98rem; }

    .section-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(0,194,168,0.15);
        border-radius: 16px;
        padding: 1.3rem 1.4rem 0.6rem 1.4rem;
        margin-bottom: 1.1rem;
    }
    .section-title {
        color: #00c2a8;
        font-weight: 600;
        font-size: 1.02rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .stButton>button {
        background: linear-gradient(120deg, #00c2a8, #00e0c0);
        color: #06171a;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.4rem;
        font-size: 1.02rem;
        letter-spacing: 0.02em;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(0,194,168,0.35); }

    .verdict-card {
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-top: 0.4rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .risk-high { background: linear-gradient(120deg, rgba(255,90,95,0.22), rgba(255,90,95,0.05)); border-color: rgba(255,90,95,0.4); }
    .risk-mid  { background: linear-gradient(120deg, rgba(255,190,90,0.20), rgba(255,190,90,0.05)); border-color: rgba(255,190,90,0.4); }
    .risk-low  { background: linear-gradient(120deg, rgba(0,194,168,0.20), rgba(0,194,168,0.05)); border-color: rgba(0,194,168,0.4); }

    .verdict-label { font-size: 0.9rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.8; }
    .verdict-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; margin: 0.2rem 0; }

    .factor-chip {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .chip-up   { background: rgba(255,90,95,0.15); color: #ffb3b5; }
    .chip-down { background: rgba(0,194,168,0.15); color: #7fe9d8; }

    hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>📡 Churn Radar</h1>
    <p>Live churn-risk scoring for telecom subscribers, powered by a logistic regression model.</p>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1.15, 1], gap="large")

# ============================================================
# INPUT FORM
# ============================================================
with left_col:
    with st.form("customer_form"):

        st.markdown('<div class="section-card"><div class="section-title">Customer profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", ["No", "Yes"])
        partner = c3.selectbox("Has partner", ["No", "Yes"])
        c4, c5 = st.columns(2)
        dependents = c4.selectbox("Has dependents", ["No", "Yes"])
        tenure = c5.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Services</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        phone_service = c1.selectbox("Phone service", ["No", "Yes"])
        multiple_lines = c2.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = c3.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        c4, c5, c6 = st.columns(3)
        online_security = c4.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = c5.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = c6.selectbox("Device protection", ["No", "Yes", "No internet service"])
        c7, c8 = st.columns(2)
        tech_support = c7.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = c8.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Account &amp; billing</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = c2.selectbox("Paperless billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        c3, c4 = st.columns(2)
        monthly_charges = c3.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = c4.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Predict churn risk", use_container_width=True)

# ============================================================
# FEATURE ENCODING  (mirrors the training notebook exactly)
# ============================================================
def build_feature_vector():
    raw = {col: 0 for col in COLUMNS}

    raw["gender"] = 1 if gender == "Male" else 0
    raw["SeniorCitizen"] = 1 if senior == "Yes" else 0
    raw["Partner"] = 1 if partner == "Yes" else 0
    raw["Dependents"] = 1 if dependents == "Yes" else 0
    raw["tenure"] = tenure
    raw["PhoneService"] = 1 if phone_service == "Yes" else 0
    raw["PaperlessBilling"] = 1 if paperless_billing == "Yes" else 0
    raw["MonthlyCharges"] = monthly_charges
    raw["TotalCharges"] = total_charges

    if multiple_lines == "No phone service":
        raw["MultipleLines_No phone service"] = 1
    elif multiple_lines == "Yes":
        raw["MultipleLines_Yes"] = 1

    if internet_service == "Fiber optic":
        raw["InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        raw["InternetService_No"] = 1

    for value, no_internet_key, yes_key in [
        (online_security, "OnlineSecurity_No internet service", "OnlineSecurity_Yes"),
        (online_backup, "OnlineBackup_No internet service", "OnlineBackup_Yes"),
        (device_protection, "DeviceProtection_No internet service", "DeviceProtection_Yes"),
        (tech_support, "TechSupport_No internet service", "TechSupport_Yes"),
        (streaming_tv, "StreamingTV_No internet service", "StreamingTV_Yes"),
        (streaming_movies, "StreamingMovies_No internet service", "StreamingMovies_Yes"),
    ]:
        if value == "No internet service":
            raw[no_internet_key] = 1
        elif value == "Yes":
            raw[yes_key] = 1

    if contract == "One year":
        raw["Contract_One year"] = 1
    elif contract == "Two year":
        raw["Contract_Two year"] = 1

    if payment_method == "Credit card (automatic)":
        raw["PaymentMethod_Credit card (automatic)"] = 1
    elif payment_method == "Electronic check":
        raw["PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        raw["PaymentMethod_Mailed check"] = 1

    ordered = np.array([raw[col] for col in COLUMNS], dtype=float)
    scaled = (ordered - np.array(MEAN)) / np.array(SCALE)
    return scaled.reshape(1, -1), raw

# ============================================================
# PREDICTION + RESULT DISPLAY
# ============================================================
with right_col:
    st.markdown('<div class="section-card"><div class="section-title">Risk verdict</div>', unsafe_allow_html=True)

    if submitted:
        model = load_model()
        features_scaled, raw = build_feature_vector()
        probability = model.predict_proba(features_scaled)[0][1]
        pct = probability * 100

        if pct >= 60:
            risk_label, risk_class, emoji = "High risk", "risk-high", "🔴"
        elif pct >= 30:
            risk_label, risk_class, emoji = "Medium risk", "risk-mid", "🟠"
        else:
            risk_label, risk_class, emoji = "Low risk", "risk-low", "🟢"

        st.markdown(f"""
        <div class="verdict-card {risk_class}">
            <div class="verdict-label">Churn probability</div>
            <div class="verdict-value">{emoji} {pct:.1f}%</div>
            <div class="verdict-label">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(int(pct), 100))

        # Top contributing factors (coefficient x scaled value)
        contributions = model.coef_[0] * features_scaled[0]
        top_idx = np.argsort(np.abs(contributions))[::-1][:5]

        st.markdown("<div style='margin-top:1rem; font-weight:600; color:#9fc9c2;'>Top factors for this customer</div>", unsafe_allow_html=True)
        chips_html = ""
        for i in top_idx:
            feature_name = COLUMNS[i].replace("_", ": ")
            direction = "increases" if contributions[i] > 0 else "lowers"
            css_class = "chip-up" if contributions[i] > 0 else "chip-down"
            arrow = "↑" if contributions[i] > 0 else "↓"
            chips_html += f'<span class="factor-chip {css_class}">{arrow} {feature_name}</span>'
        st.markdown(chips_html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <p style="color:#7c9a95;">Fill in the customer details on the left and hit
        <b>Predict churn risk</b> to see the score here.</p>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
