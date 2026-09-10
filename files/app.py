"""
Share Prediction — Customer Churn App
--------------------------------------
Streamlit UI wired to a pre-trained (already fitted) scikit-learn
LogisticRegression model for the IBM Telco Customer Churn dataset.

IMPORTANT: this app does NOT retrain anything. It only:
  1) loads the trained model + the StandardScaler used at training time,
  2) turns the form inputs into the exact 30 engineered features the
     model was fit on (same names, same order),
  3) scales them with the SAME scaler statistics used during training,
  4) calls model.predict / model.predict_proba,
  5) shows the real, pre-computed evaluation metrics (not recalculated).
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_prediction_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

# Exact feature order the model was trained on (fallback if feature_columns.pkl
# is ever missing — kept in sync with models/feature_columns.pkl).
FEATURE_COLUMNS_FALLBACK = [
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

# Real metrics fallback — only used if models/metrics.pkl can't be loaded.
# These are the ACTUAL values printed by the training notebook
# (Telco_Customer_Churn_Team4.ipynb, cell 145) for this exact model.
METRICS_FALLBACK = {
    "model_name": "Logistic Regression",
    "Accuracy": 0.7381121362668559,
    "Precision": 0.5042301184433164,
    "Recall": 0.7967914438502673,
    "F1 Score": 0.6176165803108808,
    "ROC-AUC": 0.8403239556692241,
    "confusion_matrix": {"TN": 742, "FP": 293, "FN": 76, "TP": 298},
}

st.set_page_config(
    page_title=" Customer Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Purple theme (inspired by the provided design reference)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root{
        --purple-700:#5B21B6;
        --purple-600:#6D28D9;
        --purple-500:#7C3AED;
        --purple-400:#9061F9;
        --purple-100:#EDE7FB;
        --lavender-bg:#F6F4FD;
        --text-dark:#241C3D;
        --muted:#6B6180;
    }
    html, body, [class*="css"]{
        font-family:'Segoe UI', Roboto, sans-serif;
    }
    .stApp{
        background:var(--lavender-bg);
    }
    .app-header{
        background:linear-gradient(135deg, var(--purple-600) 0%, var(--purple-400) 100%);
        padding:28px 32px;
        border-radius:22px;
        color:white;
        margin-bottom:22px;
        box-shadow:0 10px 30px rgba(124,58,237,0.25);
    }
    .app-header h1{
        margin:0;
        font-size:28px;
        font-weight:700;
    }
    .app-header p{
        margin:6px 0 0 0;
        opacity:0.9;
        font-size:14px;
    }
    .purple-card{
        background:white;
        border-radius:18px;
        padding:22px 24px;
        box-shadow:0 4px 18px rgba(124,58,237,0.08);
        border:1px solid var(--purple-100);
        margin-bottom:18px;
    }
    .section-title{
        color:var(--purple-700);
        font-weight:700;
        font-size:16px;
        margin-bottom:10px;
    }
    div.stButton > button, div.stFormSubmitButton > button{
        background:linear-gradient(135deg, var(--purple-600), var(--purple-400));
        color:white;
        border:none;
        border-radius:12px;
        padding:10px 26px;
        font-weight:600;
        box-shadow:0 6px 16px rgba(124,58,237,0.3);
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover{
        background:linear-gradient(135deg, var(--purple-700), var(--purple-500));
        color:white;
    }
    .metric-pill{
        background:var(--purple-100);
        border-radius:14px;
        padding:14px 16px;
        text-align:center;
    }
    .metric-pill .value{
        font-size:22px;
        font-weight:800;
        color:var(--purple-700);
    }
    .metric-pill .label{
        font-size:12px;
        color:var(--muted);
        text-transform:uppercase;
        letter-spacing:0.04em;
    }
    .result-churn{
        background:#FDECEC;
        border:1px solid #F5B5B5;
        border-radius:16px;
        padding:20px;
        color:#8A1F1F;
    }
    .result-safe{
        background:#EAF7EE;
        border:1px solid #B7E4C7;
        border-radius:16px;
        padding:20px;
        color:#1E6B3C;
    }
    section[data-testid="stSidebar"]{
        background:white;
        border-right:1px solid var(--purple-100);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Load artifacts (cached — the .pkl files are read once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    errors = []
    model = None
    scaler = None
    feature_columns = None
    metrics = None

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        errors.append(
            f"Model file not found at `{MODEL_PATH}`. "
            f"Place your trained `.pkl` there (see instructions below)."
        )
    except Exception as e:  # noqa: BLE001
        errors.append(f"Could not load the model file: {e}")

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        errors.append(
            f"Scaler file not found at `{SCALER_PATH}`. Predictions need it "
            f"because the model was trained on standardized features."
        )
    except Exception as e:  # noqa: BLE001
        errors.append(f"Could not load the scaler file: {e}")

    try:
        with open(FEATURES_PATH, "rb") as f:
            feature_columns = pickle.load(f)
    except Exception:
        feature_columns = FEATURE_COLUMNS_FALLBACK

    try:
        with open(METRICS_PATH, "rb") as f:
            metrics = pickle.load(f)
    except Exception:
        metrics = METRICS_FALLBACK

    return model, scaler, feature_columns, metrics, errors


model, scaler, FEATURE_COLUMNS, METRICS, LOAD_ERRORS = load_artifacts()
MODEL_READY = model is not None and scaler is not None


# --------------------------------------------------------------------------
# Feature engineering — mirrors the training notebook exactly:
#   - gender: Female->0, Male->1
#   - Partner/Dependents/PhoneService/PaperlessBilling: No->0, Yes->1
#   - SeniorCitizen: No->0, Yes->1
#   - tenure / MonthlyCharges / TotalCharges: numeric, unchanged
#   - MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
#     DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
#     Contract, PaymentMethod: one-hot encoded with drop_first=True
#     (the first category alphabetically is the implicit baseline —
#     all-zero row for that column's dummies)
# --------------------------------------------------------------------------
ONE_HOT_SPECS = {
    "MultipleLines": ("No", ["No phone service", "Yes"]),
    "InternetService": ("DSL", ["Fiber optic", "No"]),
    "OnlineSecurity": ("No", ["No internet service", "Yes"]),
    "OnlineBackup": ("No", ["No internet service", "Yes"]),
    "DeviceProtection": ("No", ["No internet service", "Yes"]),
    "TechSupport": ("No", ["No internet service", "Yes"]),
    "StreamingTV": ("No", ["No internet service", "Yes"]),
    "StreamingMovies": ("No", ["No internet service", "Yes"]),
    "Contract": ("Month-to-month", ["One year", "Two year"]),
    "PaymentMethod": (
        "Bank transfer (automatic)",
        ["Credit card (automatic)", "Electronic check", "Mailed check"],
    ),
}


def build_feature_row(raw: dict, feature_columns: list) -> pd.DataFrame:
    """Turn raw form inputs into the exact one-row DataFrame the model expects."""
    row = {col: 0 for col in feature_columns}

    row["gender"] = 1 if raw["gender"] == "Male" else 0
    row["SeniorCitizen"] = 1 if raw["SeniorCitizen"] == "Yes" else 0
    row["Partner"] = 1 if raw["Partner"] == "Yes" else 0
    row["Dependents"] = 1 if raw["Dependents"] == "Yes" else 0
    row["PhoneService"] = 1 if raw["PhoneService"] == "Yes" else 0
    row["PaperlessBilling"] = 1 if raw["PaperlessBilling"] == "Yes" else 0
    row["tenure"] = float(raw["tenure"])
    row["MonthlyCharges"] = float(raw["MonthlyCharges"])
    row["TotalCharges"] = float(raw["TotalCharges"])

    for field, (baseline, categories) in ONE_HOT_SPECS.items():
        value = raw[field]
        if value != baseline:
            col_name = f"{field}_{value}"
            if col_name in row:
                row[col_name] = 1

    return pd.DataFrame([row], columns=feature_columns)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🔮 Share Prediction — Customer Churn</h1>
        <p>Logistic Regression model trained on the Telco Customer Churn dataset ·
        predictions run through the exact preprocessing pipeline used at training time</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if LOAD_ERRORS:
    for err in LOAD_ERRORS:
        st.error(err)
    st.info(
        "The app still renders below, but predictions are disabled until the "
        "missing `.pkl` file(s) are placed in the `models/` folder."
    )

with st.sidebar:
    st.markdown("### ℹ️ About this model")
    st.write(f"**Algorithm:** {METRICS.get('model_name', 'Logistic Regression')}")
    st.write("**Class balancing:** SMOTE oversampling on the training split")
    st.write("**Scaling:** StandardScaler (fit on training data only)")
    st.write(f"**Input features:** {len(FEATURE_COLUMNS)}")
    st.write("**Target:** `Churn` — 0 = stays, 1 = churns")
    st.markdown("---")
    st.caption(
        "Model status: " + ("🟢 loaded" if MODEL_READY else "🔴 not loaded — check `models/` folder")
    )

tab_predict, tab_metrics = st.tabs(["🔮 Predict", "📈 Model Metrics"])

# --------------------------------------------------------------------------
# TAB 1 — Predict
# --------------------------------------------------------------------------
with tab_predict:
    st.markdown('<div class="purple-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Customer Information</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)

        with col2:
            st.markdown("**Services**")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        with col3:
            st.markdown("**More Services & Billing**")
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        col4, col5 = st.columns(2)
        with col4:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with col5:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=840.0, step=1.0)

        submitted = st.form_submit_button("Predict Churn", disabled=not MODEL_READY)

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        # ---- graceful input validation ----
        validation_errors = []
        if tenure < 0:
            validation_errors.append("Tenure cannot be negative.")
        if monthly_charges < 0:
            validation_errors.append("Monthly Charges cannot be negative.")
        if total_charges < 0:
            validation_errors.append("Total Charges cannot be negative.")
        if phone_service == "No" and multiple_lines not in ("No phone service",):
            st.warning(
                "Multiple Lines was set even though Phone Service is 'No' — "
                "typically this should be 'No phone service'. Proceeding with your selection."
            )
        if internet_service == "No" and any(
            v not in ("No internet service",)
            for v in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
        ):
            st.warning(
                "Internet Service is 'No' but some internet-dependent options aren't set to "
                "'No internet service' — proceeding with your selections as entered."
            )

        if validation_errors:
            for e in validation_errors:
                st.error(e)
        elif not MODEL_READY:
            st.error("Model is not loaded — cannot predict. Check the `models/` folder.")
        else:
            raw_inputs = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            try:
                feature_row = build_feature_row(raw_inputs, FEATURE_COLUMNS)
                scaled_row = scaler.transform(feature_row)

                prediction = model.predict(scaled_row)[0]
                if hasattr(model, "predict_proba"):
                    churn_probability = model.predict_proba(scaled_row)[0][1]
                else:
                    churn_probability = None

                st.markdown("### Result")
                if prediction == 1:
                    prob_txt = f" ({churn_probability:.1%} estimated probability)" if churn_probability is not None else ""
                    st.markdown(
                        f'<div class="result-churn"><h3>⚠️ Likely to Churn</h3>'
                        f'<p>This customer is predicted to <b>leave</b>{prob_txt}.</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    prob_txt = f" ({churn_probability:.1%} estimated churn probability)" if churn_probability is not None else ""
                    st.markdown(
                        f'<div class="result-safe"><h3>✅ Likely to Stay</h3>'
                        f'<p>This customer is predicted to <b>stay</b>{prob_txt}.</p></div>',
                        unsafe_allow_html=True,
                    )

                if churn_probability is not None:
                    st.progress(min(max(churn_probability, 0.0), 1.0))

                with st.expander("See engineered features sent to the model"):
                    st.dataframe(feature_row.T.rename(columns={0: "value"}))

            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")

# --------------------------------------------------------------------------
# TAB 2 — Metrics
# --------------------------------------------------------------------------
with tab_metrics:
    st.markdown('<div class="purple-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evaluation Metrics (held-out test set)</div>', unsafe_allow_html=True)
    st.caption(
        "These are the model's real, pre-computed metrics — not recalculated live. "
        "Accuracy/Precision/Recall/F1 match the training notebook's printed output exactly."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    metric_display = [
        ("Accuracy", METRICS.get("Accuracy")),
        ("Precision", METRICS.get("Precision")),
        ("Recall", METRICS.get("Recall")),
        ("F1 Score", METRICS.get("F1 Score")),
        ("ROC-AUC", METRICS.get("ROC-AUC")),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4, m5], metric_display):
        with col:
            display_val = f"{value:.2%}" if isinstance(value, (int, float)) else "N/A"
            st.markdown(
                f'<div class="metric-pill"><div class="value">{display_val}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    cm = METRICS.get("confusion_matrix")
    if cm:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_df = pd.DataFrame(
            [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
            index=["Actual: No Churn", "Actual: Churn"],
            columns=["Predicted: No Churn", "Predicted: Churn"],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
