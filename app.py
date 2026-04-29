import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load("employee_performance_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Employee Performance Predictor")

st.write("Fill in employee details to predict performance rating")

# INPUT FIELDS

age = st.slider("Age", 18, 60, 30)
hourly_rate = st.number_input("Hourly Rate", value=50)

job_satisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])
environment_satisfaction = st.selectbox("Environment Satisfaction", [1,2,3,4])
relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1,2,3,4])
work_life_balance = st.selectbox("Work Life Balance", [1,2,3,4])
job_involvement = st.selectbox("Job Involvement", [1,2,3,4])

training_times = st.slider("Training Times Last Year", 0, 10, 2)
years_company = st.slider("Years at Company", 0, 40, 5)
years_role = st.slider("Years in Current Role", 0, 20, 3)
years_manager = st.slider("Years with Current Manager", 0, 20, 3)

distance = st.slider("Distance From Home", 1, 50, 10)
total_exp = st.slider("Total Work Experience", 0, 40, 5)

gender = st.selectbox("Gender", ["Male", "Female"])
overtime = st.selectbox("Overtime", ["Yes", "No"])

business_travel = st.selectbox("Business Travel", [
    "Non-Travel", "Travel_Rarely", "Travel_Frequently"
])

marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
department = st.selectbox("Department", [
    "Sales", "Research & Development", "Human Resources",
    "Finance", "Development", "Data Science"
])

job_role = st.selectbox("Job Role", [
    "Manager", "Sales Executive", "Developer", "Scientist",
    "Analyst", "HR", "Technician"
])

education_bg = st.selectbox("Education Background", [
    "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"
])

# PREPROCESS INPUT

def preprocess_input():
    data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Numerical
    data["Age"] = age
    data["EmpHourlyRate"] = hourly_rate
    data["EmpJobSatisfaction"] = job_satisfaction
    data["EmpEnvironmentSatisfaction"] = environment_satisfaction
    data["EmpRelationshipSatisfaction"] = relationship_satisfaction
    data["EmpWorkLifeBalance"] = work_life_balance
    data["EmpJobInvolvement"] = job_involvement
    data["TrainingTimesLastYear"] = training_times
    data["ExperienceYearsAtThisCompany"] = years_company
    data["ExperienceYearsInCurrentRole"] = years_role
    data["YearsWithCurrManager"] = years_manager
    data["DistanceFromHome"] = distance
    data["TotalWorkExperienceInYears"] = total_exp

    # Binary
    data["Gender"] = 1 if gender == "Male" else 0
    data["OverTime"] = 1 if overtime == "Yes" else 0

    # Ordinal
    travel_map = {
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel_Frequently": 2
    }
    data["BusinessTravelFrequency"] = travel_map[business_travel]

    # One-hot encoding
    def set_one_hot(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in data.columns:
            data[col_name] = 1

    set_one_hot("MaritalStatus", marital_status)
    set_one_hot("EmpDepartment", department)
    set_one_hot("EmpJobRole", job_role)
    set_one_hot("EducationBackground", education_bg)

    return data

# PREDICTION

if st.button("Predict Performance"):

    input_data = preprocess_input()

    # Scale if needed
    if type(model).__name__ in ["LogisticRegression", "KNeighborsClassifier", "SVC"]:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Performance Rating: {prediction}")