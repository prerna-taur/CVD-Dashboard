import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("best_cvd_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CVD Prediction Dashboard",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 22px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 30px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">❤️ Cardiovascular Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Dashboard • Random Forest Model</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("👤 Patient Information")
st.sidebar.markdown("Enter the patient details below.")

age = st.sidebar.number_input(
    "Age", 1, 120, 50
)

gender = st.sidebar.selectbox(
    "Gender", [0, 1]
)

chestpain = st.sidebar.number_input(
    "Chest Pain", 0, 3, 1
)

restingBP = st.sidebar.number_input(
    "Resting Blood Pressure", 0, 250, 120
)

serumcholestrol = st.sidebar.number_input(
    "Serum Cholesterol", 0, 600, 200
)

fastingbloodsugar = st.sidebar.selectbox(
    "Fasting Blood Sugar", [0, 1]
)

restingrelectro = st.sidebar.number_input(
    "Resting ECG", 0, 2, 0
)

maxheartrate = st.sidebar.number_input(
    "Maximum Heart Rate", 50, 250, 150
)

exerciseangia = st.sidebar.selectbox(
    "Exercise Angina", [0, 1]
)

oldpeak = st.sidebar.number_input(
    "Oldpeak", 0.0, 10.0, 1.0
)

slope = st.sidebar.number_input(
    "Slope", 0, 2, 1
)

noofmajorvessels = st.sidebar.number_input(
    "Number of Major Vessels", 0, 3, 0
)

# -----------------------------
# Input Data
# -----------------------------
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "chestpain": [chestpain],
    "restingBP": [restingBP],
    "serumcholestrol": [serumcholestrol],
    "fastingbloodsugar": [fastingbloodsugar],
    "restingrelectro": [restingrelectro],
    "maxheartrate": [maxheartrate],
    "exerciseangia": [exerciseangia],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "noofmajorvessels": [noofmajorvessels]
})

# -----------------------------
# Patient Summary
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📋 Patient Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Age", age)
col2.metric("Blood Pressure", restingBP)
col3.metric("Cholesterol", serumcholestrol)
col4.metric("Max Heart Rate", maxheartrate)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🔍 CVD Prediction")

if st.button("❤️ Predict CVD", use_container_width=True):

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0]

    cvd_probability = probability[1] * 100

    st.markdown('<div class="result">', unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠️ CVD Predicted")
    else:
        st.success("✅ No CVD Predicted")

    st.metric(
        "CVD Probability",
        f"{cvd_probability:.2f}%"
    )

    st.progress(float(probability[1]))

    # Probability chart
    chart_data = pd.DataFrame({
        "Result": ["No CVD", "CVD"],
        "Probability": [
            probability[0] * 100,
            probability[1] * 100
        ]
    })

    st.bar_chart(
        chart_data.set_index("Result")
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Model Performance
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🤖 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", "Random Forest")
col2.metric("Accuracy", "98.50%")
col3.metric("F1 Score", "98.71%")
col4.metric("ROC-AUC", "99.90%")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">'
    '⚕️ This dashboard is for educational and demonstration purposes only. '
    'It is not a medical diagnosis.'
    '</div>',
    unsafe_allow_html=True
)