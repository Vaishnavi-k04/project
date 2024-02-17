import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed dataset
data = pd.read_csv("Dataloading_preprossed.csv")

# Data Visualization

# Create histograms for selected parameters
plt.figure(figsize=(10, 8))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sns.histplot(data[col], bins=20, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Create scatter plots for selected parameter pairs
plt.figure(figsize=(10, 8))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sns.scatterplot(
        data=data, x=col, y="Potability", hue="Potability", palette="viridis"
    )
    plt.title(f"Scatter Plot: {col} vs. Potability")
    plt.xlabel(col)
    plt.ylabel("Potability")
plt.tight_layout()
plt.show()

# Create Line Chart selected parameters
plt.figure(figsize=(10, 8))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    plt.plot(data.index, data[col], label=col, color="black")
    plt.title(f"Line Chart: {col}")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.legend()
plt.tight_layout()
plt.show()

# Calculate the correlation matrix and create a heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Building a Predictive Model

# Data Splitting
X = data.drop("Potability", axis=1)
y = data["Potability"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a Logistic Regression model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Model Evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)
