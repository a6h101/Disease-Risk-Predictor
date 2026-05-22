import streamlit as st
import numpy as np
import joblib

# ── Page config ──
st.set_page_config(
    page_title="Disease Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
/* Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit header and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background-color: #0f1117;
    color: #ffffff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2d2d3d;
}

[data-testid="stSidebar"] .stRadio label {
    color: #c9d1d9 !important;
    font-size: 15px;
    padding: 6px 0;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    border: 1px solid #2a4a6b;
}

.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
}

.hero p {
    font-size: 1rem;
    color: #8b9fc4;
    margin: 0;
}

/* Section header */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 24px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #21262d;
}

/* Input card */
.input-card {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Input labels */
.stNumberInput label, .stSelectbox label {
    color: #8b949e !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Input fields */
.stNumberInput input, .stSelectbox select {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 15px !important;
}

.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #1f6feb !important;
    box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.15) !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #0d4ea6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 8px !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #1f6feb) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(31, 111, 235, 0.4) !important;
}

/* Result cards */
.result-high {
    background: linear-gradient(135deg, #2d1b1b, #1a0e0e);
    border: 1px solid #f85149;
    border-radius: 14px;
    padding: 24px 28px;
    margin: 20px 0;
}

.result-low {
    background: linear-gradient(135deg, #1b2d1b, #0e1a0e);
    border: 1px solid #3fb950;
    border-radius: 14px;
    padding: 24px 28px;
    margin: 20px 0;
}

.result-high h2 {
    color: #f85149;
    font-size: 1.5rem;
    margin: 0 0 6px 0;
}

.result-low h2 {
    color: #3fb950;
    font-size: 1.5rem;
    margin: 0 0 6px 0;
}

.result-high p, .result-low p {
    color: #8b949e;
    font-size: 0.9rem;
    margin: 0;
}

/* Probability bar */
.prob-bar-container {
    background-color: #21262d;
    border-radius: 8px;
    height: 10px;
    margin: 8px 0 4px 0;
    overflow: hidden;
}

.prob-bar-fill-high {
    background: linear-gradient(90deg, #f85149, #ff6b6b);
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

.prob-bar-fill-low {
    background: linear-gradient(90deg, #3fb950, #56d364);
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #8b949e;
    margin-top: 4px;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 16px;
    margin-top: 16px;
}

.metric-card {
    flex: 1;
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e6edf3;
}

.metric-card .metric-label {
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Disclaimer */
.disclaimer {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #d29922;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 16px;
    color: #8b949e;
    font-size: 13px;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 24px 0;
}

/* Metrics from streamlit */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    color: #e6edf3 !important;
}

[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ──
@st.cache_resource
def load_models():
    diabetes_model  = joblib.load('models/diabetes_model.pkl')
    diabetes_scaler = joblib.load('models/diabetes_scaler.pkl')
    heart_model     = joblib.load('models/heart_model.pkl')
    heart_scaler    = joblib.load('models/heart_scaler.pkl')
    return diabetes_model, diabetes_scaler, heart_model, heart_scaler

diabetes_model, diabetes_scaler, heart_model, heart_scaler = load_models()

# ── Hero Banner ──
st.markdown("""
<div class="hero">
    <h1>🏥 Disease Risk Predictor</h1>
    <p>ML-powered predictions for Diabetes and Heart Disease risk — trained on real medical datasets</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown("## 🔬 Select Prediction")
disease = st.sidebar.radio("", ["🩸 Diabetes", "❤️ Heart Disease"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background:#0d1117; border:1px solid #21262d; border-radius:10px; padding:14px; font-size:13px; color:#8b949e;'>
    <b style='color:#e6edf3'>📊 Model Info</b><br><br>
    🩸 <b style='color:#c9d1d9'>Diabetes</b><br>
    Algorithm: Gradient Boosting<br>
    Accuracy: 76.62%<br>
    Dataset: 768 samples<br><br>
    ❤️ <b style='color:#c9d1d9'>Heart Disease</b><br>
    Algorithm: Gradient Boosting<br>
    Accuracy: 92.44%<br>
    Dataset: 1190 samples
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("")
st.sidebar.markdown("""
<div style='font-size:12px; color:#484f58; text-align:center; padding-top:8px;'>
    Built with Python · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════
# DIABETES
# ════════════════════════════════════════
if disease == "🩸 Diabetes":
    st.markdown('<div class="section-header">🩸 Diabetes Risk Assessment</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**Patient Info**")
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Age         = st.number_input("Age", min_value=1, max_value=120, value=25)
        BMI         = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**Lab Results**")
        Glucose       = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        Insulin       = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80)
        DiabetesPedigreeFunction = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
            value=0.5, step=0.01,
            help="Scores likelihood based on family history")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        glucose_bmi           = Glucose * BMI
        age_bmi               = Age * BMI
        glucose_age           = Glucose * Age
        insulin_glucose_ratio = Insulin / (Glucose + 1)

        input_data   = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                   Insulin, BMI, DiabetesPedigreeFunction, Age,
                                   glucose_bmi, age_bmi, glucose_age, insulin_glucose_ratio]])
        input_scaled = diabetes_scaler.transform(input_data)
        prediction   = diabetes_model.predict(input_scaled)
        probability  = diabetes_model.predict_proba(input_scaled)[0]

        risk_pct   = probability[1] * 100
        safe_pct   = probability[0] * 100

        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-high">
                <h2>⚠️ High Risk of Diabetes</h2>
                <p>The model predicts elevated diabetes risk based on the provided values.</p>
            </div>
            """, unsafe_allow_html=True)
            bar_class = "prob-bar-fill-high"
        else:
            st.markdown(f"""
            <div class="result-low">
                <h2>✅ Low Risk of Diabetes</h2>
                <p>The model predicts low diabetes risk based on the provided values.</p>
            </div>
            """, unsafe_allow_html=True)
            bar_class = "prob-bar-fill-low"

        st.markdown(f"""
        <div style='margin-top:16px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                <span style='color:#8b949e; font-size:13px;'>Risk probability</span>
                <span style='color:#e6edf3; font-weight:600;'>{risk_pct:.1f}%</span>
            </div>
            <div class="prob-bar-container">
                <div class="{bar_class}" style="width:{risk_pct}%"></div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{safe_pct:.1f}%</div>
                <div class="metric-label">No Diabetes</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{risk_pct:.1f}%</div>
                <div class="metric-label">Diabetes Risk</div>
            </div>
        </div>
        <div class="disclaimer">
            ⚠️ This is <b>not a medical diagnosis</b>. This tool is for educational purposes only.
            Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════
# HEART DISEASE
# ════════════════════════════════════════
elif disease == "❤️ Heart Disease":
    st.markdown('<div class="section-header">❤️ Heart Disease Risk Assessment</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**Patient Info**")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
            format_func=lambda x: {
                1: "1 — Typical angina",
                2: "2 — Atypical angina",
                3: "3 — Non-anginal pain",
                4: "4 — Asymptomatic"}[x])
        resting_bp  = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
        fasting_bs  = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", [0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("**ECG & Stress Test**")
        resting_ecg = st.selectbox("Resting ECG Results", [0, 1, 2],
            format_func=lambda x: {
                0: "0 — Normal",
                1: "1 — ST-T wave abnormality",
                2: "2 — Left ventricular hypertrophy"}[x])
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina?", [0, 1],
                                        format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak  = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", [1, 2, 3],
            format_func=lambda x: {
                1: "1 — Upsloping",
                2: "2 — Flat",
                3: "3 — Downsloping"}[x])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
        age_maxhr     = age * max_hr
        age_chol      = age * cholesterol
        maxhr_slope   = max_hr * st_slope
        oldpeak_slope = oldpeak * st_slope

        input_data   = np.array([[age, sex_val, cp, resting_bp, cholesterol,
                                   fasting_bs, resting_ecg, max_hr, exercise_angina,
                                   oldpeak, st_slope, age_maxhr, age_chol,
                                   maxhr_slope, oldpeak_slope]])
        input_scaled = heart_scaler.transform(input_data)
        prediction   = heart_model.predict(input_scaled)
        probability  = heart_model.predict_proba(input_scaled)[0]

        risk_pct = probability[1] * 100
        safe_pct = probability[0] * 100

        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-high">
                <h2>⚠️ High Risk of Heart Disease</h2>
                <p>The model predicts elevated heart disease risk based on the provided values.</p>
            </div>
            """, unsafe_allow_html=True)
            bar_class = "prob-bar-fill-high"
        else:
            st.markdown(f"""
            <div class="result-low">
                <h2>✅ Low Risk of Heart Disease</h2>
                <p>The model predicts low heart disease risk based on the provided values.</p>
            </div>
            """, unsafe_allow_html=True)
            bar_class = "prob-bar-fill-low"

        st.markdown(f"""
        <div style='margin-top:16px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                <span style='color:#8b949e; font-size:13px;'>Risk probability</span>
                <span style='color:#e6edf3; font-weight:600;'>{risk_pct:.1f}%</span>
            </div>
            <div class="prob-bar-container">
                <div class="{bar_class}" style="width:{risk_pct}%"></div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{safe_pct:.1f}%</div>
                <div class="metric-label">No Disease</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{risk_pct:.1f}%</div>
                <div class="metric-label">Heart Disease Risk</div>
            </div>
        </div>
        <div class="disclaimer">
            ⚠️ This is <b>not a medical diagnosis</b>. This tool is for educational purposes only.
            Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)