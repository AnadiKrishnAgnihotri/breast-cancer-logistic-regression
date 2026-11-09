import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_probs = model.predict_proba(X_test_scaled)[:, 1]

sorted_probs = np.sort(y_probs)


plt.plot(sorted_probs, marker='.', linestyle='none', color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.title('Predicted Probabilities (Test Set)')
plt.xlabel('Sorted Samples')
plt.ylabel('Probability of Class 1 (Malignant)')
plt.legend()
plt.show()

weights = pd.Series(model.coef_[0], index=X.columns).sort_values()


weights.plot(kind='barh', color=np.where(weights > 0, 'green', 'red'))
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')


plt.show()
