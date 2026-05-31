"""Shared utilities for FSC curve fetching, preprocessing, and classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class FSCModels:
    """Container for the loaded FSC analysis models and cluster metadata."""

    encoder: Any
    kmeans: Any
    cluster_freq: pd.Series
    cluster_percentiles: pd.Series


def resample_curve(curve: np.ndarray, target_length: int = 100) -> np.ndarray:
    """Resample an FSC curve to a fixed length using linear interpolation.

    Handles NaN values and short curves gracefully.
    """
    y = np.asarray(curve, dtype=float)
    y_clean = y[~np.isnan(y)]

    if len(y_clean) == 0:
        return np.full(target_length, np.nan)
    if len(y_clean) == 1:
        return np.full(target_length, y_clean[0])

    x_old = np.linspace(0, 1, len(y_clean))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, y_clean, kind="linear", bounds_error=False,
                 fill_value=(y_clean[0], y_clean[-1]))
    return f(x_new)


def find_crossing_point(y_values: np.ndarray, threshold: float = 0.143) -> float | None:
    """Return the fractional index where the curve first crosses below *threshold*.

    Returns ``None`` if no crossing is found.
    """
    for i in range(1, len(y_values)):
        if y_values[i - 1] >= threshold > y_values[i]:
            y0, y1 = float(y_values[i - 1]), float(y_values[i])
            return (i - 1) + (y0 - threshold) / (y0 - y1)
    return None


def anchor_curve(
    curve: np.ndarray,
    target_idx: int = 50,
    output_length: int = 100,
) -> np.ndarray:
    """Warp a curve so its 0.143 FSC threshold crossing aligns to *target_idx*.

    This mirrors the preprocessing used during model training. If no crossing
    is found the curve is simply resampled to *output_length*.
    """
    crossing = find_crossing_point(curve)
    if crossing is None:
        return resample_curve(curve, output_length)

    old_indices = np.arange(len(curve))
    new_x_coords = np.linspace(0, output_length - 1, output_length)

    # Map: old index 0 → 0, crossing → target_idx, last → output_length-1
    mapping_func = interp1d(
        [0, target_idx, output_length - 1],
        [0, crossing, len(curve) - 1],
        kind="linear",
    )
    source_indices = mapping_func(new_x_coords)

    final_interp = interp1d(old_indices, curve, kind="linear", fill_value="extrapolate")
    return final_interp(source_indices)


def fetch_fsc_curve(emd_id: str) -> list:
    """Fetch the FSC curve for an EMDB entry from the EBI API.

    Args:
        emd_id: EMDB accession code (e.g. ``"EMD-1234"``).

    Returns:
        List of FSC values.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If the response cannot be parsed or contains no FSC data.
    """
    url = f"https://ebi.ac.uk/emdb/api/analysis/{emd_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    try:
        json_data = response.json()
    except ValueError as exc:
        raise ValueError(f"Failed to parse JSON for {emd_id}") from exc

    try:
        data = list(json_data.values())[0]
        return data["fsc"]["curves"]["fsc"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"No FSC data found for {emd_id}") from exc


def load_models(model_dir: Path = _DEFAULT_MODEL_DIR) -> FSCModels:
    """Load the encoder, KMeans model, and cluster metadata from *model_dir*.

    Note:
        The encoder was trained with global min-max normalisation applied to
        the full training set. Inference accuracy requires the same normalisation;
        save ``data_min``/``data_max`` from training and apply them here once
        those artefacts are available.
    """
    from tensorflow.keras.models import load_model  # noqa: PLC0415 – deferred TF import

    encoder = load_model(model_dir / "encoder_model.h5")
    kmeans = joblib.load(model_dir / "kmeans_model.pkl")
    cluster_freq = pd.read_csv(model_dir / "cluster_frequencies.csv", index_col=0)["count"]
    cluster_percentiles = cluster_freq.rank(pct=True)

    return FSCModels(
        encoder=encoder,
        kmeans=kmeans,
        cluster_freq=cluster_freq,
        cluster_percentiles=cluster_percentiles,
    )


def classify_fsc_curve(
    fsc_curve: np.ndarray,
    models: FSCModels,
) -> tuple[int, int, float]:
    """Classify a preprocessed FSC curve using the trained encoder and KMeans model.

    Args:
        fsc_curve: Preprocessed curve array of the length used during training.
        models: Loaded :class:`FSCModels` instance.

    Returns:
        ``(cluster_id, frequency, typicality_percentile)``
    """
    encoded = models.encoder.predict(fsc_curve.reshape(1, -1), verbose=0)
    cluster_id = int(models.kmeans.predict(encoded)[0])
    frequency = int(models.cluster_freq.get(cluster_id, 0))
    percentile = float(models.cluster_percentiles.get(cluster_id, 0.0))
    return cluster_id, frequency, percentile


def draw_typicality_bar(percentile: float, width: int = 40) -> None:
    """Print a terminal bar showing where *percentile* sits on the typicality scale."""
    pos = int(percentile * width)
    bar = ["─"] * width
    if 0 <= pos < width:
        bar[pos] = "O"
    print(f"Least Typical  {''.join(bar)}  Most Typical")
