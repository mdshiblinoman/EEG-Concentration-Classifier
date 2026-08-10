# EEG Concentration Streamlit App

This project wraps the notebook model in a Streamlit website.

## Run the Website

```bash
streamlit run app.py
```

The app expects a saved model at:

```text
concentration_model.joblib
```

## Train and Save the Model

Export or copy the notebook feature file into this folder:

```text
pyeeg_processed_features.csv
```

Then train and save the classifier:

```bash
python train_and_save_model.py pyeeg_processed_features.csv
```

By default this trains a Random Forest model, which only needs scikit-learn.
To train the notebook-style XGBoost model instead:

```bash
python train_and_save_model.py pyeeg_processed_features.csv --model-type xgboost
```

You can also train from inside the Streamlit app by uploading the same CSV in the
`Train model` tab.

If a feature CSV does not include `Segment_Type`, the app lets you assign one
label to that whole file. To train a useful model, provide examples for both
`High Concentration` and `Low Concentration`.

You can train from raw EEG in the app too:

1. Open `Train model`
2. Choose `Train from raw EEG`
3. Upload EDF or numeric CSV experiment files
4. Click `Extract, train, and save from raw EEG`

This raw-training path assumes each file follows the notebook experiment layout:
the first segment is low concentration, the middle segment is high concentration,
and the last segment is low concentration.

## Required CSV Columns

```text
Delta_P, Theta_P, Alpha_P, Beta_P,
Delta_R, Theta_R, Alpha_R, Beta_R,
HFD, SVD_Entropy, Fisher_Info, DFA,
Segment_Type
```

## Use a Real EEG Signal

After `concentration_model.joblib` exists, open the `Real EEG signal` tab and upload either:

- `.edf`: the app reads the sample rate and EEG channels from the file
- `.csv`: use numeric channel columns and enter the sample rate in Hz

The app applies band-pass filtering, optional 50 Hz notch filtering, optional
common average reference, then extracts the same 12 feature columns used by the
model. It then detects `High Concentration` or `Low Concentration`, shows the
overall result, shows detection by time window, and lets you download the full
channel-level predictions.
