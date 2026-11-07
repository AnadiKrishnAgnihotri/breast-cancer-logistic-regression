import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
df = pd.read_csv("data/data.csv")
print("Initial Dataset Shape",df.shape)
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