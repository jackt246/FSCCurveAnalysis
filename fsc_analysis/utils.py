"""Shared utilities for FSC curve fetching, preprocessing, and classification."""

from __future__ import annotations

import json
import logging
import os
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
DEFAULT_CURVE_TYPE = "fsc_masked"


def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) TensorFlow for reproducible runs."""
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        logger.debug("TensorFlow not available; skipping TF seeding.")


@dataclass
class FSCModels:
    """Container for the loaded FSC analysis models and inference metadata.

    Attributes:
        encoder: Trained autoencoder encoder.
        kmeans: Fitted KMeans model (exposes ``cluster_centers_``).
        data_min: Global minimum used to normalise the training data.
        data_max: Global maximum used to normalise the training data.
        cluster_thresholds: Per-cluster distance threshold (indexed by cluster id)
            beyond which a curve is considered atypical for that cluster.
    """

    encoder: Any
    kmeans: Any
    data_min: float
    data_max: float
    cluster_thresholds: pd.Series


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


def normalize_curve(
    curve: np.ndarray,
    data_min: float,
    data_max: float,
) -> np.ndarray:
    """Apply the training-time global min-max normalisation to a curve.

    Uses the ``data_min``/``data_max`` persisted from training so a single
    curve is scaled identically to the data the encoder was trained on.
    """
    arr = np.asarray(curve, dtype=float)
    if data_max == data_min:
        return arr
    return (arr - data_min) / (data_max - data_min)


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


def load_models(
    model_dir: Path = _DEFAULT_MODEL_DIR,
    curve_type: str = DEFAULT_CURVE_TYPE,
) -> FSCModels:
    """Load the encoder, KMeans model, and inference metadata from *model_dir*.

    Loads the artefacts produced by ``train_kmeans`` for *curve_type*:
    the encoder, the KMeans model, the persisted normalisation parameters
    (``data_min``/``data_max``), and the per-cluster centroid-distance
    thresholds used to score typicality.
    """
    from tensorflow.keras.models import load_model  # noqa: PLC0415 – deferred TF import

    encoder = load_model(model_dir / f"encoder_model_{curve_type}.h5")
    kmeans = joblib.load(model_dir / f"kmeans_model_{curve_type}.pkl")

    with open(model_dir / f"normalisation_{curve_type}.json") as fh:
        norm = json.load(fh)

    stats = pd.read_csv(model_dir / f"cluster_distance_stats_{curve_type}.csv")
    cluster_thresholds = stats.set_index("cluster_id")["threshold"]

    return FSCModels(
        encoder=encoder,
        kmeans=kmeans,
        data_min=float(norm["data_min"]),
        data_max=float(norm["data_max"]),
        cluster_thresholds=cluster_thresholds,
    )


def classify_fsc_curve(
    fsc_curve: np.ndarray,
    models: FSCModels,
) -> tuple[int, float, float]:
    """Score a preprocessed FSC curve by its distance from its cluster centroid.

    The curve is normalised with the training-time parameters, encoded, and
    assigned to its nearest KMeans cluster. Typicality is derived from the
    Euclidean distance to that cluster's centroid in latent space, scaled by
    the cluster's distance threshold:
    ``typicality = clip(1 - distance / threshold, 0, 1)`` (1.0 = sits on the
    centroid, 0.0 = at or beyond the atypicality threshold).

    Args:
        fsc_curve: Preprocessed (anchored) curve of the training length.
        models: Loaded :class:`FSCModels` instance.

    Returns:
        ``(cluster_id, distance_from_centroid, typicality)``
    """
    normalized = normalize_curve(fsc_curve, models.data_min, models.data_max)
    embedding = models.encoder.predict(normalized.reshape(1, -1), verbose=0)
    cluster_id = int(models.kmeans.predict(embedding)[0])

    centroid = np.asarray(models.kmeans.cluster_centers_[cluster_id], dtype=float)
    distance = float(np.sqrt(np.sum((np.asarray(embedding[0], dtype=float) - centroid) ** 2)))

    threshold = float(models.cluster_thresholds.get(cluster_id, np.nan))
    if not np.isfinite(threshold) or threshold <= 0:
        typicality = 0.0
    else:
        typicality = float(np.clip(1.0 - distance / threshold, 0.0, 1.0))

    return cluster_id, distance, typicality


def draw_typicality_bar(percentile: float, width: int = 40) -> None:
    """Print a terminal bar showing where *percentile* sits on the typicality scale."""
    pos = int(percentile * width)
    bar = ["─"] * width
    if 0 <= pos < width:
        bar[pos] = "O"
    print(f"Least Typical  {''.join(bar)}  Most Typical")
