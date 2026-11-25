
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👩‍💼",
    layout="wide"
)

# Application heading
st.title("Employee Attrition Prediction")
st.write(
    "Upload employee information to predict whether employees "
    "are likely to leave the organization."
)

# Load the trained machine learning model
model_path = Path("models/best_attrition_model.pkl")

if not model_path.exists():
    st.error("Trained model not found. Please run train.py first.")
    st.stop()

model = joblib.load(model_path)

# Upload CSV file
st.subheader("Upload Employee Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read uploaded employee data
        employee_data = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully!")

        st.subheader("Uploaded Employee Data")
        st.dataframe(employee_data)

        # Remove target and unnecessary columns
        prediction_data = employee_data.drop(
            columns=[
                "Attrition",
                "EmployeeNumber",
                "EmployeeCount",
                "Over18",
                "StandardHours"
            ],
            errors="ignore"
        )

        # Make predictions
        predictions = model.predict(prediction_data)
        probabilities = model.predict_proba(prediction_data)[:, 1]

        # Prepare prediction results
        results = employee_data.copy()

        results["Predicted_Attrition"] = [
            "Yes" if value == 1 else "No"
            for value in predictions
        ]

        results["Attrition_Probability"] = probabilities

        # Display results
        
# Dashboard Summary

        # Dashboard Summary
        st.subheader("Employee Attrition Dashboard")

        total_employees = len(predictions)
        employees_leaving = int(sum(predictions == 1))
        employees_staying = int(sum(predictions == 0))

        leaving_percentage = (
            employees_leaving / total_employees * 100
            if total_employees > 0 else 0
        )

        # Summary Cards
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Employees", total_employees)
        col2.metric("Predicted to Leave", employees_leaving)
        col3.metric("Predicted to Stay", employees_staying)

        # Chart 1
        st.write("### 1. Attrition Prediction Comparison")

        attrition_chart = pd.DataFrame({
            "Number of Employees": [
                employees_leaving,
                employees_staying
            ]
        }, index=["Predicted to Leave", "Predicted to Stay"])

        st.bar_chart(attrition_chart)

        # Chart 2
       
        # Chart 2: Attrition Probability Distribution
        st.write("### 2. Attrition Probability Distribution")

        probability_data = pd.DataFrame({
            "Attrition Probability (%)": probabilities * 100
        })

        probability_data["Probability Range"] = pd.cut(
            probability_data["Attrition Probability (%)"],
            bins=[0, 20, 40, 60, 80, 100.000001],
            labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            include_lowest=True
        )

        probability_chart = (
            probability_data["Probability Range"]
            .value_counts(sort=False)
            .to_frame(name="Number of Employees")
        )

        st.bar_chart(probability_chart)


        # Interpretation
        st.write("### 3. Dashboard Interpretation")

        st.write(
            f"Out of {total_employees} employees, "
            f"{employees_leaving} are predicted to leave "
            f"and {employees_staying} are predicted to stay."
        )

        st.write(
            f"The predicted attrition rate is "
            f"{leaving_percentage:.2f}%."
        )

        # Download results
        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="attrition_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.info("Please upload a CSV file to get predictions.")
