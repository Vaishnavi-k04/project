import pandas as pd

data = pd.read_csv("water_potability.csv")

# Display the first few rows
print(data.head())

# Check data types and summary statistics
print(data.info())
print(data.describe())

# Handle missing values
data = data.dropna()


# Save the preprocessed dataset
data.to_csv("Dataloading_preprossed.csv", index=False)
