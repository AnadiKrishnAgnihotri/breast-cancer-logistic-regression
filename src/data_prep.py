"""Prepare data, train a logistic-regression classifier, and report test metrics."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "data.csv"
    df = pd.read_csv(data_path)
    print("Initial Dataset Shape:", df.shape)

    # Remove identifier/empty columns if present, then encode the diagnosis target.
    df = df.drop(columns=[col for col in ("id", "Unnamed: 32") if col in df.columns])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    print("\nMissing values per column:\n", df.isnull().sum())
    df = df.fillna(df.median(numeric_only=True))

    print("\nCleaned Dataset Shape:", df.shape)
    print("Target Class Counts:\n", df["diagnosis"].value_counts())

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Fit the scaler on training data only to avoid leaking test-set information.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nX_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape :", X_test_scaled.shape)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=["Benign", "Malignant"]
    ))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
