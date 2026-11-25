
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/best_attrition_model.pkl")

# Load new employee records
new_data = pd.read_csv("new_employees.csv")

# Remove target and ID columns if present
new_data = new_data.drop(
    columns=[
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ],
    errors="ignore"
)

# Predict attrition
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]

# Create result
results = pd.DataFrame({
    "Predicted_Attrition": [
        "Yes" if value == 1 else "No"
        for value in predictions
    ],
    "Attrition_Probability": probabilities
})

# Save predictions
results.to_csv("outputs/predictions.csv", index=False)

print(results)
print("Predictions saved successfully!")
