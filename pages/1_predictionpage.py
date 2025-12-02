import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logreg_model.pkl")

st.title("Telco Customer Interactive Churn Predictor")

# ---- USER INPUTS ----
st.header("Enter Customer Details")

tenure = st.number_input("Tenure (months)", 0, 72)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# ---- FEATURE VECTOR ----
input_dict = {
    "Contract_One year": 1 if Contract == "One year" else 0,
    "Contract_Two year": 1 if Contract == "Two year" else 0,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges
}

df_input = pd.DataFrame([input_dict])

# ---- PREDICTION ----
if st.button("Predict Churn"):
    prob = model.predict_proba(df_input)[:, 1][0]

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prob * 100:.2f}%")

    if prob >= 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
