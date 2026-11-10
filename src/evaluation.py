import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, confusion_matrix, classification_report
)

def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
    print("\n--- Training Set Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f"Recall   : {recall_score(y_train, y_train_pred):.4f}")
    print(f"F1-Score : {f1_score(y_train, y_train_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train, y_train_pred)}")

    print("\n--- Testing Set Evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_test_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
