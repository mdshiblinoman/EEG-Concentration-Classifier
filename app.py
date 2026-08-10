from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from eeg_processing import (
    extract_feature_frame,
    extract_labeled_experiment_features,
    preprocess_signal,
    read_edf,
    read_numeric_csv,
)
from model_utils import (
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_artifact,
    predict,
    save_artifact,
    train_model,
)


st.set_page_config(
    page_title="EEG Concentration Classifier",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def get_artifact(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return load_artifact(path)
    except Exception as exc:
        return {"__load_error__": str(exc)}


def format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def single_prediction_form(artifact: dict) -> None:
    st.subheader("Single prediction")
    st.caption("Enter the 12 extracted EEG features used by the notebook model.")

    defaults = {
        "Delta_P": 0.0,
        "Theta_P": 0.0,
        "Alpha_P": 0.0,
        "Beta_P": 0.0,
        "Delta_R": 0.25,
        "Theta_R": 0.25,
        "Alpha_R": 0.25,
        "Beta_R": 0.25,
        "HFD": 1.0,
        "SVD_Entropy": 1.0,
        "Fisher_Info": 0.0,
        "DFA": 0.5,
    }

    values = {}
    columns = st.columns(3)
    for index, feature in enumerate(FEATURE_COLUMNS):
        with columns[index % 3]:
            values[feature] = st.number_input(
                feature,
                value=float(defaults.get(feature, 0.0)),
                format="%.8f",
            )

    if st.button("Predict concentration", type="primary"):
        input_df = pd.DataFrame([values])
        prediction = predict(artifact, input_df)
        label = prediction.loc[0, "Prediction"]
        st.success(f"Predicted class: {label}")

        probability_columns = [
            column for column in prediction.columns if column.startswith("Probability_")
        ]
        if probability_columns:
            probability_data = {
                column.replace("Probability_", ""): format_probability(prediction.loc[0, column])
                for column in probability_columns
            }
            st.dataframe(pd.DataFrame([probability_data]), use_container_width=True)


def batch_prediction_panel(artifact: dict) -> None:
    st.subheader("Batch prediction")
    uploaded_file = st.file_uploader(
        "Upload a CSV with the feature columns",
        type=["csv"],
        key="batch_prediction_file",
    )

    if uploaded_file is None:
        st.info("The CSV must include: " + ", ".join(FEATURE_COLUMNS))
        return

    input_df = pd.read_csv(uploaded_file)
    try:
        predictions = predict(artifact, input_df)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.dataframe(predictions, use_container_width=True)
    st.download_button(
        "Download predictions",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="concentration_predictions.csv",
        mime="text/csv",
    )


def real_eeg_panel(artifact: dict | None) -> None:
    st.subheader("Real EEG signal")
    st.caption("Upload an EDF file or numeric CSV to detect low or high concentration.")

    uploaded_file = st.file_uploader(
        "Upload EEG signal",
        type=["edf", "csv"],
        key="real_eeg_file",
    )

    settings_columns = st.columns(4)
    with settings_columns[0]:
        csv_sfreq = st.number_input(
            "CSV sample rate (Hz)",
            min_value=1.0,
            value=173.0,
            step=1.0,
        )
    with settings_columns[1]:
        window_seconds = st.number_input(
            "Window seconds",
            min_value=1.0,
            value=60.0,
            step=1.0,
        )
    with settings_columns[2]:
        step_seconds = st.number_input(
            "Step seconds",
            min_value=1.0,
            value=60.0,
            step=1.0,
        )
    with settings_columns[3]:
        notch_freq = st.number_input(
            "Notch filter (Hz)",
            min_value=0.0,
            value=50.0,
            step=1.0,
        )

    filter_columns = st.columns(4)
    with filter_columns[0]:
        l_freq = st.number_input("Low cut (Hz)", min_value=0.0, value=1.0, step=0.5)
    with filter_columns[1]:
        h_freq = st.number_input("High cut (Hz)", min_value=0.0, value=40.0, step=0.5)
    with filter_columns[2]:
        apply_car = st.checkbox("Apply CAR", value=True)
    with filter_columns[3]:
        known_label = st.selectbox(
            "Known label",
            ["Unlabeled", "High Concentration", "Low Concentration"],
        )

    if uploaded_file is None:
        st.info("EDF files use their embedded sample rate. CSV files need channel columns plus the sample rate above.")
        return

    if artifact is None:
        st.warning("No saved model found. Train and save the model first, then upload raw EEG here for detection.")

    if st.button("Detect concentration from raw EEG", type="primary"):
        try:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix == ".edf":
                data, sfreq, channel_names = load_edf_upload(uploaded_file)
            else:
                csv_df = pd.read_csv(uploaded_file)
                data, channel_names = read_numeric_csv(csv_df)
                sfreq = float(csv_sfreq)

            processed = preprocess_signal(
                data,
                sfreq,
                l_freq=float(l_freq),
                h_freq=float(h_freq),
                notch_freq=float(notch_freq),
                apply_car=apply_car,
            )
            features = extract_feature_frame(
                processed,
                sfreq,
                channel_names,
                source_name=uploaded_file.name,
                window_seconds=float(window_seconds),
                step_seconds=float(step_seconds),
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        if known_label != "Unlabeled":
            features[TARGET_COLUMN] = known_label

        st.success(f"Created {len(features):,} feature rows from {len(channel_names)} channels at {sfreq:g} Hz.")
        st.dataframe(features.head(100), use_container_width=True)

        st.download_button(
            "Download extracted features",
            data=features.to_csv(index=False).encode("utf-8"),
            file_name="real_eeg_features.csv",
            mime="text/csv",
        )

        if artifact is None:
            return

        predictions = predict(artifact, features)
        counts = predictions["Prediction"].value_counts()
        top_label = counts.index[0]
        confidence = counts.iloc[0] / counts.sum()

        st.subheader("Detection result")
        result_columns = st.columns(3)
        result_columns[0].metric("Overall concentration", top_label)
        result_columns[1].metric("Agreement", f"{confidence:.1%}")
        result_columns[2].metric("Windows/channels checked", f"{len(predictions):,}")

        probability_columns = [
            column for column in predictions.columns if column.startswith("Probability_")
        ]
        if probability_columns:
            avg_probabilities = predictions[probability_columns].mean()
            avg_probabilities.index = [
                index.replace("Probability_", "") for index in avg_probabilities.index
            ]
            st.write("Average class probability")
            st.bar_chart(avg_probabilities)

        window_columns = ["Window_Start_s", "Window_End_s"]
        if all(column in predictions.columns for column in window_columns):
            window_summary = (
                predictions.groupby(window_columns)["Prediction"]
                .agg(lambda labels: labels.value_counts().index[0])
                .reset_index()
                .rename(columns={"Prediction": "Detected_Concentration"})
            )
            st.write("Detection by time window")
            st.dataframe(window_summary, use_container_width=True)

        st.write("All channel predictions")
        st.dataframe(predictions, use_container_width=True)

        st.download_button(
            "Download raw EEG detections",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="raw_eeg_concentration_detections.csv",
            mime="text/csv",
        )


def load_edf_upload(uploaded_file) -> tuple:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)
    try:
        return read_edf(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def training_panel(model_path: Path) -> None:
    st.subheader("Train and save model")
    st.caption("Create the saved model needed for raw EEG detection.")

    feature_tab, raw_tab = st.tabs(["Train from feature CSV", "Train from raw EEG"])

    with feature_tab:
        train_from_feature_csv(model_path)

    with raw_tab:
        train_from_raw_eeg(model_path)


def train_from_feature_csv(model_path: Path) -> None:
    st.write("Use labeled feature CSV files produced by the notebook or this app.")

    uploaded_files = st.file_uploader(
        "Upload feature CSV file(s)",
        type=["csv"],
        key="training_file",
        accept_multiple_files=True,
    )
    balance = st.checkbox("Balance classes before training", value=True)
    model_type_label = st.selectbox(
        "Model type",
        ["Random Forest", "XGBoost"],
        help="Random Forest is recommended when XGBoost is not installed.",
    )

    if not uploaded_files:
        st.info(
            "Required columns: "
            + ", ".join(FEATURE_COLUMNS)
            + f", {TARGET_COLUMN}. If {TARGET_COLUMN} is missing, upload separate files and assign labels below."
        )
        return

    frames = []
    for index, uploaded_file in enumerate(uploaded_files):
        df = pd.read_csv(uploaded_file)
        missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
        if missing_features:
            st.error(f"{uploaded_file.name} is missing feature column(s): {', '.join(missing_features)}")
            continue

        labeled_df = df.copy()
        if TARGET_COLUMN not in labeled_df.columns:
            assigned_label = st.selectbox(
                f"Label for {uploaded_file.name}",
                ["High Concentration", "Low Concentration"],
                key=f"feature_csv_label_{index}",
            )
            labeled_df[TARGET_COLUMN] = assigned_label
        else:
            st.write(
                f"{uploaded_file.name}: found labels "
                + ", ".join(sorted(labeled_df[TARGET_COLUMN].dropna().astype(str).unique()))
            )

        frames.append(labeled_df)

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    class_counts = df[TARGET_COLUMN].value_counts()
    st.write(f"Rows ready for training: {len(df):,}")
    class_count_df = class_counts.rename_axis(TARGET_COLUMN).reset_index(name="Rows")
    st.dataframe(class_count_df, use_container_width=True)

    if class_counts.size < 2:
        st.warning("Training needs both High Concentration and Low Concentration examples. Upload another labeled file or use a CSV that already contains both labels.")
        return

    if st.button("Train and save", type="primary"):
        train_and_persist(df, model_path, balance, model_type_label)


def train_from_raw_eeg(model_path: Path) -> None:
    st.write("Upload raw experiment files where first 60s and last 60s are low concentration, and middle 60s is high concentration.")

    uploaded_files = st.file_uploader(
        "Upload raw EEG training files",
        type=["edf", "csv"],
        accept_multiple_files=True,
        key="raw_training_files",
    )

    columns = st.columns(4)
    with columns[0]:
        csv_sfreq = st.number_input(
            "Training CSV sample rate (Hz)",
            min_value=1.0,
            value=173.0,
            step=1.0,
            key="training_csv_sfreq",
        )
    with columns[1]:
        segment_seconds = st.number_input(
            "Label segment seconds",
            min_value=1.0,
            value=60.0,
            step=1.0,
        )
    with columns[2]:
        balance = st.checkbox("Balance raw classes", value=True)
    with columns[3]:
        model_type_label = st.selectbox(
            "Raw model type",
            ["Random Forest", "XGBoost"],
            help="Random Forest is recommended when XGBoost is not installed.",
        )

    filter_columns = st.columns(4)
    with filter_columns[0]:
        l_freq = st.number_input("Training low cut (Hz)", min_value=0.0, value=1.0, step=0.5)
    with filter_columns[1]:
        h_freq = st.number_input("Training high cut (Hz)", min_value=0.0, value=40.0, step=0.5)
    with filter_columns[2]:
        notch_freq = st.number_input("Training notch (Hz)", min_value=0.0, value=50.0, step=1.0)
    with filter_columns[3]:
        apply_car = st.checkbox("Training CAR", value=True)

    if not uploaded_files:
        st.info("Upload one or more raw EDF/CSV experiment files to create the saved model.")
        return

    if st.button("Extract, train, and save from raw EEG", type="primary"):
        feature_frames = []
        errors = []
        for uploaded_file in uploaded_files:
            try:
                suffix = Path(uploaded_file.name).suffix.lower()
                if suffix == ".edf":
                    data, sfreq, channel_names = load_edf_upload(uploaded_file)
                else:
                    csv_df = pd.read_csv(uploaded_file)
                    data, channel_names = read_numeric_csv(csv_df)
                    sfreq = float(csv_sfreq)

                processed = preprocess_signal(
                    data,
                    sfreq,
                    l_freq=float(l_freq),
                    h_freq=float(h_freq),
                    notch_freq=float(notch_freq),
                    apply_car=apply_car,
                )
                feature_frames.append(
                    extract_labeled_experiment_features(
                        processed,
                        sfreq,
                        channel_names,
                        source_name=uploaded_file.name,
                        segment_seconds=float(segment_seconds),
                    )
                )
            except ValueError as exc:
                errors.append(f"{uploaded_file.name}: {exc}")

        if errors:
            st.error("\n".join(errors))
        if not feature_frames:
            return

        training_df = pd.concat(feature_frames, ignore_index=True)
        st.success(f"Created {len(training_df):,} labeled feature rows.")
        st.dataframe(training_df.head(100), use_container_width=True)
        st.download_button(
            "Download raw training features",
            data=training_df.to_csv(index=False).encode("utf-8"),
            file_name="raw_training_features.csv",
            mime="text/csv",
        )
        train_and_persist(training_df, model_path, balance, model_type_label)


def train_and_persist(
    df: pd.DataFrame,
    model_path: Path,
    balance: bool,
    model_type_label: str,
) -> None:
    try:
        model_type = "random_forest" if model_type_label == "Random Forest" else "xgboost"
        artifact, result = train_model(df, balance=balance, model_type=model_type)
        save_artifact(artifact, model_path)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.cache_resource.clear()
    st.success(f"Model saved to {model_path}")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Accuracy", f"{result.accuracy:.2%}")
    metric_columns[1].metric("Rows used", f"{result.rows_used:,}")
    metric_columns[2].metric("Classes", str(len(result.classes)))

    cm = pd.DataFrame(
        result.confusion_matrix,
        index=result.classes,
        columns=result.classes,
    )
    st.write("Confusion matrix")
    st.dataframe(cm, use_container_width=True)

    st.info("Refresh the page after saving, then use the Real EEG signal tab for detection.")


def main() -> None:
    model_path = DEFAULT_MODEL_PATH
    artifact = get_artifact(str(model_path))
    load_error = artifact.get("__load_error__") if isinstance(artifact, dict) else None
    if load_error:
        artifact = None

    st.title("EEG Concentration Classifier")
    st.write("Predict high or low concentration from extracted EEG features.")

    with st.sidebar:
        st.header("Model")
        if artifact is None:
            st.warning("No saved model found.")
            st.write(f"Expected file: `{model_path}`")
            if load_error:
                st.error(f"Saved model could not be loaded: {load_error}")
        else:
            metadata = artifact.get("metadata", {})
            st.success("Saved model loaded.")
            st.write(metadata.get("model_name", "Concentration classifier"))
            if "accuracy" in metadata:
                st.metric("Notebook-style test accuracy", f"{metadata['accuracy']:.2%}")
            st.write("Classes: " + ", ".join(metadata.get("classes", [])))

    train_tab, real_eeg_tab, single_tab, batch_tab = st.tabs(
        ["Train model", "Real EEG signal", "Single prediction", "Batch prediction"]
    )

    with train_tab:
        training_panel(model_path)

    with real_eeg_tab:
        real_eeg_panel(artifact)

    with single_tab:
        if artifact is None:
            st.info("Train or save a model first, then return to this tab.")
        else:
            single_prediction_form(artifact)

    with batch_tab:
        if artifact is None:
            st.info("Train or save a model first, then return to this tab.")
        else:
            batch_prediction_panel(artifact)


if __name__ == "__main__":
    main()
