from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURE_COLUMNS = [
    "Delta_P",
    "Theta_P",
    "Alpha_P",
    "Beta_P",
    "Delta_R",
    "Theta_R",
    "Alpha_R",
    "Beta_R",
    "HFD",
    "SVD_Entropy",
    "Fisher_Info",
    "DFA",
]

TARGET_COLUMN = "Segment_Type"
DEFAULT_MODEL_PATH = Path("concentration_model.joblib")


@dataclass
class TrainingResult:
    accuracy: float
    report: dict[str, Any]
    confusion_matrix: list[list[int]]
    classes: list[str]
    rows_used: int


def validate_feature_frame(df: pd.DataFrame, *, require_target: bool = False) -> None:
    required_columns = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing}")


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_feature_frame(df, require_target=True)
    training_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    training_df = training_df.replace([np.inf, -np.inf], np.nan).dropna()
    if training_df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data must contain at least two classes.")
    return training_df


def balance_classes(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    class_counts = df[TARGET_COLUMN].value_counts()
    min_samples = int(class_counts.min())
    balanced_parts = [
        group.sample(n=min_samples, random_state=random_state)
        for _, group in df.groupby(TARGET_COLUMN)
    ]
    return (
        pd.concat(balanced_parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def train_model(
    df: pd.DataFrame,
    *,
    balance: bool = True,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, Any], TrainingResult]:
    training_df = clean_training_frame(df)
    if balance:
        training_df = balance_classes(training_df, random_state=random_state)

    X = training_df[FEATURE_COLUMNS]
    y = training_df[TARGET_COLUMN]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise ValueError(
                "XGBoost is not installed in this Python environment. Use Random Forest "
                "or run `pip install -r requirements.txt` and restart Streamlit."
            ) from exc
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
        model_name = "XGBoost concentration classifier"
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        model_name = "Random Forest concentration classifier"
    else:
        raise ValueError("Unknown model type. Choose `random_forest` or `xgboost`.")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    classes = label_encoder.classes_.tolist()

    artifact = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metadata": {
            "model_name": model_name,
            "model_type": model_type,
            "accuracy": float(accuracy),
            "classes": classes,
            "rows_used": int(len(training_df)),
        },
    }

    result = TrainingResult(
        accuracy=float(accuracy),
        report=classification_report(
            y_test,
            y_pred,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        classes=classes,
        rows_used=int(len(training_df)),
    )
    return artifact, result


def save_artifact(artifact: dict[str, Any], model_path: Path = DEFAULT_MODEL_PATH) -> None:
    joblib.dump(artifact, model_path)


def load_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(model_path)


def predict(artifact: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    validate_feature_frame(df)
    feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
    X = df[feature_columns]
    X_scaled = artifact["scaler"].transform(X)
    encoded_predictions = artifact["model"].predict(X_scaled)
    labels = artifact["label_encoder"].inverse_transform(encoded_predictions.astype(int))

    output = df.copy()
    output["Prediction"] = labels

    if hasattr(artifact["model"], "predict_proba"):
        probabilities = artifact["model"].predict_proba(X_scaled)
        classes = artifact["label_encoder"].classes_
        for index, class_name in enumerate(classes):
            output[f"Probability_{class_name}"] = probabilities[:, index]

    return output
