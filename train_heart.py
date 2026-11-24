from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "heart.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "heart_model.pkl"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Indices into FEATURE_COLUMNS
NUMERIC_INDICES = [0, 3, 4, 7, 9, 11]  # age, trestbps, chol, thalch, oldpeak, ca
BINARY_INDICES = [1, 5, 8]             # sex, fbs, exang
NOMINAL_INDICES = [2, 6, 10, 12]       # cp, restecg, slope, thal

CATEGORY_MAPPINGS = {
    "sex": {"Male": 1, "Female": 0},
    "cp": {
        "typical angina": 0,
        "atypical angina": 1,
        "non-anginal": 2,
        "asymptomatic": 3,
    },
    "fbs": {False: 0, True: 1, "FALSE": 0, "TRUE": 1},
    "restecg": {
        "normal": 0,
        "st-t abnormality": 1,
        "lv hypertrophy": 2,
    },
    "exang": {False: 0, True: 1, "FALSE": 0, "TRUE": 1},
    "slope": {"upsloping": 0, "flat": 1, "downsloping": 2},
    "thal": {
        "normal": 1,
        "fixed defect": 2,
        "reversable defect": 3,
    },
}


def load_and_prepare_data(csv_path: Path = DATASET_PATH):
    """Load heart.csv, encode categories, mark invalid zeros as NaN, and return (X, y)."""
    df = pd.read_csv(csv_path)

    for col, mapping in CATEGORY_MAPPINGS.items():
        df[col] = df[col].map(mapping)

    X_df = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # In the UCI Cleveland/Switzerland/Hungarian dataset, 0 in trestbps or chol denotes missing values.
    X_df.loc[X_df["trestbps"] <= 0, "trestbps"] = np.nan
    X_df.loc[X_df["chol"] <= 0, "chol"] = np.nan

    y = (pd.to_numeric(df["num"], errors="coerce").fillna(0) > 0).astype(int)
    return X_df.to_numpy(dtype=float), y.to_numpy()


def build_pipeline() -> Pipeline:
    """Create a preprocessing + RandomForest pipeline that imputes missing values cleanly."""
    numeric_transformer = SimpleImputer(strategy="median")
    binary_transformer = SimpleImputer(strategy="most_frequent")
    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_INDICES),
            ("bin", binary_transformer, BINARY_INDICES),
            ("nom", nominal_transformer, NOMINAL_INDICES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save_model(
    csv_path: Path = DATASET_PATH,
    output_path: Path = MODEL_OUTPUT_PATH,
):
    """Train the heart screening pipeline, print evaluation metrics, and save with joblib."""
    X, y = load_and_prepare_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    joblib.dump(pipeline, output_path)
    print(f"Saved model pipeline to {output_path}")
    print(
        "Evaluation on held-out test set (n={}): "
        "Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
        "Recall={recall:.4f}, ROC-AUC={roc_auc:.4f}".format(len(y_test), **metrics)
    )
    return pipeline, metrics


if __name__ == "__main__":
    train_and_save_model()
