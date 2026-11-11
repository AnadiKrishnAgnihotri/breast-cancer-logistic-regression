# Breast Cancer Diagnostic Classifier

This project classifies breast biopsy tumours as **benign or malignant** using Logistic Regression implemented from scratch with Python. It uses the Wisconsin Diagnostic Breast Cancer dataset and libraries such as `pandas`, `numpy`, `matplotlib` and `scikit-learn`.

We first explore the dataset, analyse statistical relationships between cytological features, prepare the data and implement Logistic Regression using the Sigmoid function . Finally, we evaluate the model using classification metrics and visualizations.

**AI101 Mini-Project · Group 23 · Python 3.10+**

## Team and roles

| Member | Entry no. | Role | GitHub Handle |
| :--- | :--- | :--- | :--- |
| ANADI KRISHN AGNIHOTRI | 2026AIB1009 | Documentation lead | [@AnadiKrishAgnihotri](https://github.com/AnadiKrishAgnihotri) |
| ABHIMANYU SHARMA | 2026EEB1314 | Execution lead | [@AbhimanyuSharma420](https://github.com/AbhimanyuSharma420) |
| SHREYA SANDIP UMAVANE | 2026MCB1570 | Video lead | [@WaterNymph37](https://github.com/WaterNymph37) |
| NELAPATI YADIDYA ROY | 2026ESB1483 | PPT lead | [@yadidya-roy09](https://github.com/yadidya-roy09) |

All four team members contributed to the coding and implementation of the project. Individual contributions can also be viewed through the GitHub commit history.

## About the project

The Wisconsin Diagnostic Breast Cancer dataset contains information extracted from digitized images of fine needle aspirates (FNA) of breast masses. It describes characteristics of cell nuclei present in the images, including radius, texture, perimeter, area, smoothness, compactness, concavity and symmetry.

The aim of this project is to analyse these features and develop a machine learning model that can classify a tumour into one of two categories:

- `1` — Malignant (cancerous)
- `0` — Benign (non-cancerous)

Since there are only two possible outcomes, this is a **binary classification** problem. We use Logistic Regression to estimate the probability of a tumour belonging to the malignant class.



## Dataset

We use the **Wisconsin Diagnostic Breast Cancer (WDBC) dataset**, available through the UCI Machine Learning Repository. The dataset is included in this repository as `data/data.csv`.

- **569 rows** representing breast tumour samples.
- **30 numerical features** describing cell nuclei characteristics.
- **Target column:** `Diagnosis`.
- **Classes:** Malignant (M) and Benign (B).

The 30 numerical features are derived from ten basic characteristics of cell nuclei:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

For each characteristic, three measurements are provided: mean, standard error and worst (largest) value.

The diagnosis column is converted into numerical labels for binary classification.

## How the project works

The project follows these steps:

1. **Explore the data:** Load the dataset and inspect its shape, columns, data types, missing values and class distribution.
2. **Clean the data:** Remove unnecessary identifier columns, check for missing values and prepare the target labels.
3. **Perform statistical analysis:** Calculate mean, standard deviation, variance, covariance and correlation to understand the distribution and relationships between features.
4. **Visualize the data:** Generate a class distribution plot to understand the number of malignant and benign samples.
5. **Split the dataset:** Divide the dataset into training and testing sets using an 75:25 split, with stratification to preserve the original class distribution.
6. **Scale the features:** Apply `StandardScaler` to normalize the feature values. The scaler is fitted only on the training data to prevent data leakage.
7. **Implement Logistic Regression:** Calculate the weighted sum of input features and pass it through the sigmoid function to obtain predicted probabilities.

8. **Train the model:** 
9. **Evaluate the model:** Calculate training and testing accuracy, precision, recall, F1-score and RMSE. Generate a confusion matrix to analyse classification performance.
10. **Generate visualizations:** Plot the sigmoid curve to understand the model's behaviour and learning process.

## Project files

The repository contains the following project structure:

```text
breast-cancer-logistic-regression/
├── data/
│   └── data.csv
├── docs/
│   └── presentation.pdf
├── outputs/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── sigmoid_curve.png
│   └── summary_statistics.csv
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data_prep.py
│   ├── eda.py
│   ├── logistic_model.py
│   └── evaluate_model.py
├── requirements.txt
└── README.md
└── main.ipynb
```

### Description of important files

- `data/data.csv` — Contains the Wisconsin Diagnostic Breast Cancer dataset with 569 samples and 30 numerical features.
- `src/main.py` — Main entry point that executes the complete machine learning pipeline.
- `src/data_prep.py` — Handles dataset loading, cleaning, stratified train-test splitting and feature scaling.
- `src/eda.py` — Performs exploratory data analysis and calculates mean, standard deviation, variance, covariance and correlation. It also generates the class distribution plot.
- `src/logistic_model.py` — Implements the sigmoid function  algorithm for Logistic Regression.
- `src/evaluate_model.py` — Evaluates the model using accuracy, precision, recall, F1-score and confusion matrix.
- `outputs/class_distribution.png` — Visualization of the distribution of malignant and benign samples.
- `outputs/sigmoid_curve.png` — Graphical representation of the sigmoid activation function.
- `docs/presentation.pdf` — Five-slide presentation explaining the project, methodology and results.
- `requirements.txt` — Contains the Python dependencies required to run the project.
- `README.md` — Project documentation and instructions.
- `main.ipynb` - For easier demonstration, this file would actually execute all the code so that everything can be seen easily.

## Getting started

### Requirements

- Python 3.10 or newer
- NumPy
- pandas
- Matplotlib
- scikit-learn

The required packages and their versions are listed in `requirements.txt`.

## Setup
Tested on Python 3.14(Windows,Macos)
**1. Clone the repository**
```bash
git clone https://github.com/AnadiKrishnAgnihotri/breast-cancer-logistic-regression.git
cd breast-cancer-logistic-regression
```

**2. Create and activate a virtual environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install the dependencies**
```bash
python -m pip install -r requirements.txt
```
## How to run

1. Start Jupyter from the project folder:
   ```bash
   jupyter notebook main.ipynb
   ```
2. In the notebook, choose **Kernel → Restart & Run All** (or press `Shift+Enter` through each cell in order).
3. The notebook loads the dataset from `data/`, trains the model and prints the evaluation metrics and plots.

## What to expect

When the main program runs successfully, it performs the complete machine learning workflow.

The expected outputs include:

- Dataset overview and feature information.
- Summary statistics, including mean, standard deviation and variance.
- Covariance and correlation analysis.
- Malignant and benign class distribution.
- Training and testing dataset split.
- Sigmoid function visualization.
  
- Training and testing accuracy.
- Precision, recall and F1-score.
- Confusion matrix.

The original dataset contains 569 samples and 30 numerical features. The training and testing sets are created using an 75:25 stratified split.

The exact model performance values are generated when the program is executed. No fixed accuracy value is reported here because it should correspond to the actual output of the final implementation.

## Presentation and project video

The project presentation and demonstration video are part of the final submission.

- **Presentation:** [View 5-Slide Presentation PDF](https://github.com/AnadiKrishnAgnihotri/breast-cancer-logistic-regression/blob/main/docs/Breast_Cancer_Logistic_Regression_Redesigned.pdf)
- **GitHub Repository:** https://github.com/AnadiKrishnAgnihotri/breast-cancer-logistic-regression.git.
- **Project Video:** https://drive.google.com/file/d/1JO3joQY__Sz5uNas8DRynikdbln_Rnxc/view?usp=drivesdk


## Use of AI tools

AI tools are used during different stages of the project workflow, including understanding Logistic Regression, debugging Python code, understanding statistical concepts, preparing documentation,  improving visualizations for improvising the making of readme

Gemini was used to create the code for the feature importance(coefficients) graph in logistics_model.py

The dataset is based on the Wisconsin Diagnostic Breast Cancer dataset, and model predictions and evaluation results are generated by executing the implemented machine learning pipeline.
