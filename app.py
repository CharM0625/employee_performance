import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL

model = joblib.load("full_pipeline.pkl")

st.set_page_config(page_title="Employee Performance Predictor", layout="wide")

# APP TITLE

st.title("📊 Employee Performance Prediction App")
st.write("Predict employee performance using machine learning")

# SIDEBAR INPUTS

st.sidebar.header("Enter Employee Details")

def user_input():
    Age = st.sidebar.slider("Age", 18, 60, 30)

    EmpDepartment = st.sidebar.selectbox(
        "Department",
        ["Sales", "Development", "Finance", "HR"]
    )

    EmpJobSatisfaction = st.sidebar.slider(
        "Job Satisfaction (1-4)", 1, 4, 2
    )

    EmpEnvironmentSatisfaction = st.sidebar.slider(
        "Environment Satisfaction (1-4)", 1, 4, 2
    )

    EmpWorkLifeBalance = st.sidebar.slider(
        "Work-Life Balance (1-4)", 1, 4, 2
    )

    TrainingTimesLastYear = st.sidebar.slider(
        "Training Times Last Year", 0, 10, 2
    )

    ExperienceYearsAtThisCompany = st.sidebar.slider(
        "Years at Company", 0, 40, 5
    )

    EmpHourlyRate = st.sidebar.slider(
        "Hourly Rate", 10, 100, 50
    )

    data = {
        "Age": Age,
        "EmpDepartment": EmpDepartment,
        "EmpJobSatisfaction": EmpJobSatisfaction,
        "EmpEnvironmentSatisfaction": EmpEnvironmentSatisfaction,
        "EmpWorkLifeBalance": EmpWorkLifeBalance,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "ExperienceYearsAtThisCompany": ExperienceYearsAtThisCompany,
        "EmpHourlyRate": EmpHourlyRate
    }

    return pd.DataFrame([data])

# GET INPUT DATA

input_df = user_input()

# DISPLAY INPUT

st.subheader("📋 Input Data")
st.write(input_df)

# PREDICTION

if st.button("Predict Performance"):
    try:
        prediction = model.predict(input_df)[0]

        st.subheader("🎯 Prediction Result")
        st.success(f"Predicted Performance Rating: {prediction}")

        # Interpretation
        if prediction == 4:
            st.success("🌟 High Performer")
        elif prediction == 3:
            st.info("👍 Average Performer")
        else:
            st.warning("⚠️ Needs Improvement")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# MODEL INSIGHTS

st.subheader("📌 Model Insights")

try:
    final_model = model.named_steps['model']
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

    if hasattr(final_model, "feature_importances_"):
        import numpy as np

        importances = final_model.feature_importances_

        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        st.bar_chart(feat_df.set_index("Feature"))

    else:
        st.write("Feature importance not available for this model")

except Exception as e:
    st.write(f"Error loading model insights: {e}")