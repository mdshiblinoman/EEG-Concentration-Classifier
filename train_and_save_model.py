from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_utils import DEFAULT_MODEL_PATH, save_artifact, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save the EEG concentration classifier from a feature CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="pyeeg_processed_features.csv",
        help="Path to the training feature CSV produced by the notebook.",
    )
    parser.add_argument(
        "--test-csv",
        help="Optional held-out test feature CSV. Use this for the notebook train/test workflow.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing before training.",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost"],
        default="random_forest",
        help="Classifier to train. Random Forest is the default because it only needs scikit-learn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(
            f"Could not find {csv_path}. Export pyeeg_processed_features.csv from the notebook first."
        )

    df = pd.read_csv(csv_path)
    test_df = pd.read_csv(args.test_csv) if args.test_csv else None
    artifact, result = train_model(
        df,
        test_df=test_df,
        balance=not args.no_balance,
        model_type=args.model_type,
    )
    model_path = Path(args.model_path)
    save_artifact(artifact, model_path)

    print(f"Saved model to {model_path}")
    print(f"Evaluation: {result.evaluation_mode}")
    print(f"Rows used: {result.rows_used}")
    print(f"Test rows: {result.test_rows_used}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Classes: {', '.join(result.classes)}")


if __name__ == "__main__":
    main()
