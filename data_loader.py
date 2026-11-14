"""Load the UCI Heart Disease (Cleveland) dataset from Hugging Face, clean it, split it.

Dataset: skrishna/heart_disease_uci on the Hugging Face Hub (303 patients, 13 features).
Original source: Janosi, Steinbrunn, Pfisterer & Detrano (1988), UCI Machine Learning
Repository, https://doi.org/10.24432/C52P4X
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

HF_DATASET = "skrishna/heart_disease_uci"
TARGET = "target"  # 1 = heart disease present, 0 = absent
RANDOM_STATE = 42

# Plain-English descriptions, used by the interactive demo and for readable charts.
FEATURE_INFO = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 none/asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)",
    "restecg": "Resting ECG (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Max heart rate achieved (bpm)",
    "exang": "Exercise-induced chest pain (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise (mm)",
    "slope": "Slope of peak exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thallium stress test (3 normal, 6 fixed defect, 7 reversible defect)",
}


def load_raw() -> pd.DataFrame:
    """Download the dataset with the `datasets` library and return a pandas DataFrame."""
    ds = load_dataset(HF_DATASET)
    split = list(ds.keys())[0]  # this repo ships a single split named "test"
    return ds[split].to_pandas()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn '?' placeholders into NaN, make every column numeric, drop bad rows.

    Remaining NaNs are NOT filled here: the model Pipeline fills them with the
    training-set median, so no information from the test set leaks into training.
    """
    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df[TARGET] > 0).astype(int)  # safety: works if target is 0-4 too
    return df


def load_data(test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test (stratified so both sets keep the class balance)."""
    df = clean(load_raw())
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


if __name__ == "__main__":
    df = clean(load_raw())
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")
    print("Missing values per column (filled later by the Pipeline):")
    print(df.isna().sum()[df.isna().sum() > 0].to_string() or "  none")
    print("\nClass balance:\n", df[TARGET].value_counts().rename({0: "no disease", 1: "disease"}))
    X_train, X_test, y_train, y_test = load_data()
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")
