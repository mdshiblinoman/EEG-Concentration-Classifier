from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

from model_utils import FEATURE_COLUMNS


BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 7.0),
    "Alpha": (7.0, 12.0),
    "Beta": (12.0, 30.0),
}


def read_edf(path: Path) -> tuple[np.ndarray, float, list[str]]:
    try:
        import mne
    except Exception as exc:
        raise ValueError(
            "EDF upload needs MNE. Install dependencies with "
            "`pip install -r requirements.txt`, then restart Streamlit."
        ) from exc

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick_types(eeg=True, verbose=False)
    if len(raw.ch_names) == 0:
        raise ValueError("No EEG channels were found in the EDF file.")
    return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)


def read_numeric_csv(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError("CSV must contain numeric EEG columns.")

    time_like = [column for column in numeric_df.columns if column.lower() in {"time", "timestamp"}]
    numeric_df = numeric_df.drop(columns=time_like, errors="ignore")
    if numeric_df.empty:
        raise ValueError("CSV must contain numeric EEG channel columns.")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric_df.empty:
        raise ValueError("No valid numeric samples remain after cleaning the CSV.")

    return numeric_df.to_numpy(dtype=float).T, list(numeric_df.columns)


def preprocess_signal(
    data: np.ndarray,
    sfreq: float,
    *,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float = 50.0,
    apply_car: bool = True,
) -> np.ndarray:
    processed = np.asarray(data, dtype=float).copy()
    nyquist = sfreq / 2.0

    if l_freq > 0 and h_freq > l_freq and h_freq < nyquist:
        sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
        processed = _safe_sosfiltfilt(sos, processed)

    if notch_freq > 0 and notch_freq < nyquist:
        b, a = iirnotch(w0=notch_freq, Q=30, fs=sfreq)
        processed = _safe_sosfiltfilt(tf2sos(b, a), processed)

    if apply_car and processed.shape[0] > 1:
        processed = processed - processed.mean(axis=0, keepdims=True)

    return processed


def extract_feature_frame(
    data: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    *,
    source_name: str,
    window_seconds: float = 60.0,
    step_seconds: float | None = None,
) -> pd.DataFrame:
    if sfreq <= 0:
        raise ValueError("Sampling frequency must be greater than zero.")

    samples_per_window = max(1, int(round(sfreq * window_seconds)))
    samples_per_step = max(1, int(round(sfreq * (step_seconds or window_seconds))))

    if data.shape[1] < samples_per_window:
        raise ValueError(
            f"Signal is too short for a {window_seconds:g}s window at {sfreq:g} Hz."
        )

    rows = []
    for start in range(0, data.shape[1] - samples_per_window + 1, samples_per_step):
        end = start + samples_per_window
        for channel_index, channel_name in enumerate(channel_names):
            segment = data[channel_index, start:end]
            features = extract_features(segment, sfreq)
            rows.append(
                {
                    "File": source_name,
                    "Channel": channel_name,
                    "Window_Start_s": start / sfreq,
                    "Window_End_s": end / sfreq,
                    **features,
                }
            )

    if not rows:
        raise ValueError("No feature windows were created from this signal.")
    return pd.DataFrame(rows)


def extract_labeled_experiment_features(
    data: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    *,
    source_name: str,
    segment_seconds: float = 60.0,
) -> pd.DataFrame:
    if sfreq <= 0:
        raise ValueError("Sampling frequency must be greater than zero.")

    samples_per_segment = max(1, int(round(sfreq * segment_seconds)))
    if data.shape[1] < samples_per_segment * 3:
        raise ValueError(
            f"{source_name} is too short. It needs at least {segment_seconds * 3:g}s "
            "for low/high/low experiment labeling."
        )

    rows = []
    high_start = samples_per_segment
    high_end = samples_per_segment * 2
    low_tail_start = data.shape[1] - samples_per_segment

    for channel_index, channel_name in enumerate(channel_names):
        first_low = data[channel_index, :samples_per_segment]
        last_low = data[channel_index, low_tail_start:]
        combined_low = np.concatenate([first_low, last_low])
        high = data[channel_index, high_start:high_end]

        rows.append(
            {
                "File": source_name,
                "Channel": channel_name,
                "Segment_Type": "Low Concentration",
                **extract_features(combined_low, sfreq),
            }
        )
        rows.append(
            {
                "File": source_name,
                "Channel": channel_name,
                "Segment_Type": "High Concentration",
                **extract_features(high, sfreq),
            }
        )

    return pd.DataFrame(rows)


def extract_features(signal: np.ndarray, sfreq: float) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("Each segment needs at least 8 valid samples.")

    x = x - np.mean(x)
    powers = _band_powers(x, sfreq)
    total_power = sum(powers.values()) or np.finfo(float).eps

    values = {
        "Delta_P": powers["Delta"],
        "Theta_P": powers["Theta"],
        "Alpha_P": powers["Alpha"],
        "Beta_P": powers["Beta"],
        "Delta_R": powers["Delta"] / total_power,
        "Theta_R": powers["Theta"] / total_power,
        "Alpha_R": powers["Alpha"] / total_power,
        "Beta_R": powers["Beta"] / total_power,
        "HFD": _higuchi_fd(x),
        "SVD_Entropy": _svd_entropy(x),
        "Fisher_Info": _fisher_info(x),
        "DFA": _dfa(x),
    }
    return {column: float(values[column]) for column in FEATURE_COLUMNS}


def _safe_sosfiltfilt(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    try:
        return sosfiltfilt(sos, data, axis=1)
    except ValueError:
        return data


def _band_powers(x: np.ndarray, sfreq: float) -> dict[str, float]:
    nperseg = min(len(x), max(8, int(round(sfreq * 2))))
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg)
    powers = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        powers[band_name] = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers


def _higuchi_fd(x: np.ndarray, kmax: int = 5) -> float:
    n = len(x)
    lengths = []
    ks = range(1, min(kmax, n // 2) + 1)
    for k in ks:
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            diff = np.abs(np.diff(x[idx])).sum()
            norm = (n - 1) / (((len(idx) - 1) * k) or 1)
            lm.append((diff * norm) / k)
        if lm:
            lengths.append(np.mean(lm))

    if len(lengths) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(1.0 / np.array(list(ks)[: len(lengths)])), np.log(lengths), 1)
    return float(coeffs[0])


def _svd_entropy(x: np.ndarray, tau: int = 4, embedding_dim: int = 10) -> float:
    matrix = _embed_seq(x, tau=tau, embedding_dim=embedding_dim)
    if matrix.size == 0:
        return 0.0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _fisher_info(x: np.ndarray, tau: int = 4, embedding_dim: int = 10) -> float:
    matrix = _embed_seq(x, tau=tau, embedding_dim=embedding_dim)
    if matrix.size == 0:
        return 0.0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) < 2:
        return 0.0
    return float(np.sum(np.diff(probabilities) ** 2 / probabilities[:-1]))


def _embed_seq(x: np.ndarray, tau: int, embedding_dim: int) -> np.ndarray:
    n_vectors = len(x) - (embedding_dim - 1) * tau
    if n_vectors <= 1:
        return np.empty((0, 0))
    return np.array([x[i : i + embedding_dim * tau : tau] for i in range(n_vectors)])


def _dfa(x: np.ndarray) -> float:
    y = np.cumsum(x - np.mean(x))
    n = len(y)
    scales = np.unique(np.logspace(np.log10(4), np.log10(max(5, n // 4)), num=8).astype(int))
    fluctuations = []
    valid_scales = []

    for scale in scales:
        if scale < 4 or scale >= n:
            continue
        segments = n // scale
        if segments < 2:
            continue

        rms = []
        for segment_index in range(segments):
            segment = y[segment_index * scale : (segment_index + 1) * scale]
            t = np.arange(scale)
            trend = np.polyval(np.polyfit(t, segment, 1), t)
            rms.append(np.sqrt(np.mean((segment - trend) ** 2)))

        fluctuation = np.sqrt(np.mean(np.square(rms)))
        if fluctuation > 0:
            valid_scales.append(scale)
            fluctuations.append(fluctuation)

    if len(fluctuations) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)
    return float(coeffs[0])
