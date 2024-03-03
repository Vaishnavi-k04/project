import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed dataset
data = pd.read_csv("Dataloading_preprossed.csv")

# Scatter plots
plt.figure(figsize=(4, 4))
sns.pairplot(
    data[["Solids", "ph", "Sulfate", "Hardness", "Potability"]],
    hue="Potability",
    palette="viridis",
)
plt.suptitle("PairPlot_Potability")
plt.show()

# Create histograms for selected parameters
plt.figure(figsize=(7, 7))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sns.histplot(data[col], bins=20, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Calculate the correlation matrix and create a heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Building a Predictive Model

# Data Splitting with stratified sampling
X = data.drop("Potability", axis=1)
y = data["Potability"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)

# Create a Logistic Regression model with hyperparameter tuning
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

# Model Evaluation
y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Logistic Regression Model Evaluation:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
