from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import DATASET_PATH, FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH, TARGET_COLUMN
from .data_validation import load_dataset
from .preprocessing import build_preprocessor


def select_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.1, 0.9, 81):
        predictions = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        score = (2 * precision * recall) / (precision + recall + 1e-9)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def evaluate_model(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
    }


def make_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            random_state=42,
            max_iter=250,
        )
    raise ValueError(f"Unknown model name: {name}")


def train_and_save_model(dataset_path: str | Path = DATASET_PATH, model_path: str | Path = MODEL_PATH, metadata_path: str | Path = METADATA_PATH):
    data = load_dataset(dataset_path)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=42,
    )

    results = []
    for model_name in ["logistic_regression", "random_forest", "hist_gradient_boosting"]:
        model = make_model(model_name)
        pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_prob)
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, test_prob, threshold)
        results.append({
            "model_name": model_name,
            "model": pipeline,
            "threshold": threshold,
            "metrics": metrics,
        })

    best = max(results, key=lambda entry: (entry["metrics"]["roc_auc"], entry["metrics"]["f1"]))
    selected_model = best["model"]
    selected_threshold = best["threshold"]
    selected_metrics = best["metrics"]

    joblib.dump(selected_model, model_path)

    metadata = {
        "model_version": f"eduguard-v{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        "training_date": datetime.utcnow().isoformat(timespec="seconds"),
        "selected_model": best["model_name"],
        "selected_threshold": selected_threshold,
        "dataset_version": "sih2026_student_dropout_10000_gpa10",
        "feature_schema": FEATURE_COLUMNS,
        "random_seed": 2026,
        "metrics": {
            "test": selected_metrics,
            "comparison": {entry["model_name"]: entry["metrics"] for entry in results},
        },
        "notes": "Synthetic-data performance only; not evidence of real-world effectiveness.",
    }

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


if __name__ == "__main__":
    train_and_save_model()
    print("Model trained successfully.")
