import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
)
from sklearn.impute import SimpleImputer


water_quality_data = pd.read_csv("water_potability.csv")

selected_columns = [
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

data_for_anomaly_detection = water_quality_data[selected_columns]

# Initialize the dataframe to replace missing values with the mean
# for anomaly_detection
imputer = SimpleImputer(strategy="mean")

data_for_anomaly_detection_imputed = imputer.fit_transform(data_for_anomaly_detection)

model = IsolationForest(contamination=0.05, random_state=42)

model.fit(data_for_anomaly_detection_imputed)

anomaly_predictions = model.predict(data_for_anomaly_detection_imputed)

water_quality_data["Anomaly"] = anomaly_predictions

anomalies = water_quality_data[water_quality_data["Anomaly"] == -1]

print("Detected Anomalies:")
print(anomalies)

water_quality_data.to_csv("anomaly_detection.csv", index=False)
