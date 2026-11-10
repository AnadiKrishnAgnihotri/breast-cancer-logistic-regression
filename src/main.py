import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, confusion_matrix, classification_report
)

df = pd.read_csv("data/data.csv")
print("Initial Dataset Shape", df.shape)

df = df.drop(columns=[col for col in ["id", "Unnamed: 32"] if col in df.columns])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("\nMissing values per column:\n", df.isnull().sum())
df = df.fillna(df.median(numeric_only=True))
print("\nCleaned Dataset Shape:", df.shape)
print("Target Class Counts:\n", df["diagnosis"].value_counts())

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.25, random_state=42, stratify=y.values
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nX_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape :", X_test_scaled.shape)



def run_eda(df_input):
    
    features = df_input.drop(columns=["diagnosis"], errors="ignore")
    features = features.select_dtypes(include="number")

    print("\nFeature means:\n", features.mean())
    print("\nFeature standard deviations:\n", features.std())
    print("\nFeature variances:\n", features.var())
    print("\nFeature covariance:\n", features.cov())
    print("\nFeature correlation:\n", features.corr())

    
    counts = df_input["diagnosis"].value_counts().sort_index()
    labels = ["Benign" if value == 0 else "Malignant" for value in counts.index]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts.values)
    plt.title("Benign and Malignant Cases")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of cases")
    plt.tight_layout()
    plt.show()


run_eda(df)



model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_probs = model.predict_proba(X_test_scaled)[:, 1]
sorted_probs = np.sort(y_probs)

plt.figure(figsize=(8, 4))
plt.plot(sorted_probs, marker='.', linestyle='none', color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.title('Predicted Probabilities (Test Set)')
plt.xlabel('Sorted Samples')
plt.ylabel('Probability of Class 1 (Malignant)')
plt.legend()
plt.tight_layout()
plt.show()

weights = pd.Series(model.coef_[0], index=X.columns).sort_values()

plt.figure(figsize=(10, 8))
weights.plot(kind='barh', color=np.where(weights > 0, 'green', 'red'))
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



def evaluate_model(y_train_arg, y_train_pred_arg, y_test_arg, y_test_pred_arg):
    print("\n--- Training Set Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_train_arg, y_train_pred_arg):.4f}")
    print(f"Precision: {precision_score(y_train_arg, y_train_pred_arg):.4f}")
    print(f"Recall   : {recall_score(y_train_arg, y_train_pred_arg):.4f}")
    print(f"F1-Score : {f1_score(y_train_arg, y_train_pred_arg):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train_arg, y_train_pred_arg)}")

    print("\n--- Testing Set Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test_arg, y_test_pred_arg):.4f}")
    print(f"Precision: {precision_score(y_test_arg, y_test_pred_arg):.4f}")
    print(f"Recall   : {recall_score(y_test_arg, y_test_pred_arg):.4f}")
    print(f"F1-Score : {f1_score(y_test_arg, y_test_pred_arg):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_arg, y_test_pred_arg)}")
    print("\nClassification Report:\n", classification_report(y_test_arg, y_test_pred_arg))


y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

evaluate_model(y_train, y_train_pred, y_test, y_test_pred)
