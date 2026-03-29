import pandas as pd

# Load Excel file
df = pd.read_excel("all_projects_metadata.xlsx")

# Save as CSV
df.to_csv("all_projects_metadata.csv", index=False)

print("Converted successfully!")
