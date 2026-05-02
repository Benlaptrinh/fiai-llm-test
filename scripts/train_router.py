"""
Train a lightweight intent classifier for router intent detection.

This script trains a TF-IDF + Logistic Regression classifier using
data/synthetic_queries.csv.

The trained artifact is saved to models/router_model.joblib.

This is intentionally lightweight so it can run quickly on local CPU/Mac.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/synthetic_queries.csv")
MODEL_PATH = Path("models/router_model.joblib")
REPORT_PATH = Path("report/router_training_report.txt")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "query" not in df.columns or "expected_intent" not in df.columns:
        raise ValueError(
            "synthetic_queries.csv must contain query and expected_intent columns"
        )

    features = df["query"].astype(str)
    labels = df["expected_intent"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    text = (
        "Lightweight Router Training Report\n"
        "=================================\n\n"
        f"Dataset: {DATA_PATH}\n"
        f"Train size: {len(x_train)}\n"
        f"Test size: {len(x_test)}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"{report}\n"
    )

    REPORT_PATH.write_text(text, encoding="utf-8")

    print(text)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
