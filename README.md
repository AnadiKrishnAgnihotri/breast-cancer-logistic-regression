# Breast Cancer Diagnostic Classifier

Classifies breast biopsy tumors as benign or malignant using Logistic Regression from scratch (`pandas`, `numpy`, `matplotlib`, `scikit-learn`).

`AI101 Mini-Project` · `Python 3.10+`

## Team and roles

| Member | Entry no. | Role | GitHub Handle |
| :--- | :--- | :--- | :--- |
| ANADI KRISHN AGNIHOTRI | 2026AIB1009 | Documentation lead | `@AnadiKrishnAgnihotri' |
| ABHIMANYU SHARMA | 2026EEB1314 | Execution lead | `@AbhimanyuSharma420` |
| SHREYA SANDIP UMAVANE | 2026MCB1570 | Video lead | `@WaterNymph37` |
| NELAPATI YADIDYA ROY | 2026ESB1483| PPT lead | `@yadidya-roy09` |

All four members contributed code (see commit history).

## Project structure

```text
breast-cancer-logistic-regression/
├── data/
│   └── data.csv                  # 569 rows, 30 cytological features (UCI Wisconsin Diagnostic)
├── docs/
│   └── presentation.pdf          # 5-slide project presentation
├── outputs/                      # Generated plots and statistical summary CSV are saved here
│   ├── class_distribution.png
│   ├── loss_curve.png
│   ├── sigmoid_curve.png
│   └── summary_statistics.csv
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point (runs full pipeline)
│   ├── data_prep.py              # Data loading, cleaning, stratified split, StandardScaler
│   ├── eda.py                    # Mean, Std, Variance, Covariance, Correlation, class plot
│   ├── logistic_model.py         # Sigmoid, Binary Cross-Entropy loss, Batch Gradient Descent
│   └── evaluate_model.py         # Train/Test Accuracy, Precision, Recall, F1, RMSE, Confusion Matrix
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Project documentation
