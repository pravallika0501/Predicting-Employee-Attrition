import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model, scaler, and column names
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("🧠 Employee Attrition Prediction App")
st.markdown("This app predicts whether an employee is likely to leave the company based on input features.")

# Sidebar inputs
st.sidebar.header("🔧 Employee Features Input")

# Example input features – customize based on your dataset
age = st.sidebar.slider("Age", 18, 60, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
job_role = st.sidebar.selectbox("Job Role", (
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative', 'Manager',
    'Sales Representative', 'Research Director', 'Human Resources'
))
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
business_travel = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
department = st.sidebar.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])

# Build the input DataFrame
input_data = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "Gender": 1 if gender == "Male" else 0,
    "OverTime_Yes": 1 if overtime == "Yes" else 0
}

# One-hot encode JobRole
for role in [
    'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative'
]:
    input_data[role] = 1 if role.split("_")[1] == job_role else 0

# One-hot encode BusinessTravel
for travel in ['BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely']:
    input_data[travel] = 1 if travel.split("_")[1] == business_travel else 0

# One-hot encode Department
for dept in ['Department_Human Resources', 'Department_Research & Development', 'Department_Sales']:
    input_data[dept] = 1 if dept.split("_")[1] == department else 0

# Add any missing columns from training with default 0
for col in columns:
    if col not in input_data:
        input_data[col] = 0

# Create DataFrame and reorder
input_df = pd.DataFrame([input_data])
input_df = input_df[columns]

# Scale input
scaled_input = scaler.transform(input_df)

# Prediction
if st.button("🚀 Predict Attrition"):
    prediction = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ The employee is likely to leave. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ The employee is likely to stay. (Probability: {1 - prob:.2f})")

# Optional: Upload CSV for basic visualizations
st.markdown("---")
st.markdown("### 📊 Explore the Dataset")
uploaded_file = st.file_uploader("Upload your HR CSV data (optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("#### Attrition Count")
    st.bar_chart(df['Attrition'].value_counts())

    st.write("#### Monthly Income by Job Role")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="JobRole", y="MonthlyIncome", hue="Attrition", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("#### Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap='coolwarm', annot=False, ax=ax2)
    st.pyplot(fig2)
