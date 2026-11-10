import matplotlib.pyplot as plt


def run_eda(df):
    # Summarize numeric features, excluding the diagnosis target.
    features = df.drop(columns=["diagnosis"], errors="ignore")
    features = features.select_dtypes(include="number")

    print("\nFeature means:\n", features.mean())
    print("\nFeature standard deviations:\n", features.std())
    print("\nFeature variances:\n", features.var())
    print("\nFeature covariance:\n", features.cov())
    print("\nFeature correlation:\n", features.corr())

    # diagnosis is encoded as 0 = Benign and 1 = Malignant in data_prep.py.
    counts = df["diagnosis"].value_counts().sort_index()
    labels = ["Benign" if value == 0 else "Malignant" for value in counts.index]

    plt.bar(labels, counts.values)
    plt.title("Benign and Malignant Cases")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of cases")
    plt.tight_layout()
    plt.show()
