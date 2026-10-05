import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from generate_data import generate_dataset

SEED = 42
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "resume_shortlisting.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

def build_preprocessor():
    numeric = ["experience_years", "relevant_skill_count", "certification_count"]
    categorical = ["education_level", "previous_industry"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=8, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=4,
            random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            random_state=SEED
        )
    }

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)

    X = df.drop(columns=["shortlisted"])
    y = df["shortlisted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    rows = []
    fitted = {}

    for name, estimator in build_models().items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator)
        ])

        cv_result = cross_validate(
            pipe, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        rows.append({
            "model": name,
            "cv_accuracy_mean": cv_result["test_accuracy"].mean(),
            "cv_precision_mean": cv_result["test_precision"].mean(),
            "cv_recall_mean": cv_result["test_recall"].mean(),
            "cv_f1_mean": cv_result["test_f1"].mean(),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        })
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values(
        ["cv_f1_mean", "cv_precision_mean"],
        ascending=False
    ).reset_index(drop=True)

    # "Best" is selected by cross-validated F1, with precision as tie-breaker.
    best_name = results.loc[0, "model"]
    best_pipe = fitted[best_name]

    joblib.dump(best_pipe, MODEL_DIR / "best_model.joblib")
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    metadata = {
        "best_model": best_name,
        "selection_rule": "Highest mean 5-fold CV F1; precision used as tie-breaker.",
        "random_seed": SEED,
        "test_size": 0.20
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Group-wise evaluation on the held-out test set.
    test_frame = X_test.copy()
    test_frame["actual"] = y_test.to_numpy()
    test_frame["predicted"] = best_pipe.predict(X_test)

    fairness_rows = []
    for group_col in ["education_level", "previous_industry"]:
        for group_value, g in test_frame.groupby(group_col):
            fairness_rows.append({
                "group_attribute": group_col,
                "group": group_value,
                "n": len(g),
                "precision": precision_score(g["actual"], g["predicted"], zero_division=0),
                "recall": recall_score(g["actual"], g["predicted"], zero_division=0),
                "f1": f1_score(g["actual"], g["predicted"], zero_division=0),
                "selection_rate": g["predicted"].mean()
            })

    pd.DataFrame(fairness_rows).to_csv(
        REPORT_DIR / "group_fairness_metrics.csv", index=False
    )

    print("\nMODEL COMPARISON")
    print(results.round(3).to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("Reports saved in reports/")
    print("Model saved in models/best_model.joblib")

if __name__ == "__main__":
    main()
