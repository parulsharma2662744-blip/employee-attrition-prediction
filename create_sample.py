
import pandas as pd

# Read original dataset
df = pd.read_csv("dataset.csv")

# Remove unnecessary columns
new_data = df.drop(
    columns=[
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ],
    errors="ignore"
)

# Select first five employee records
sample_data = new_data.head(5)

# Create new CSV file
sample_data.to_csv("new_employees.csv", index=False)

print("New employee file created successfully!")
print(sample_data)
