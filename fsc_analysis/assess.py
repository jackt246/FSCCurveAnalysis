"""Classify a single FSC curve by EMD ID and report its typicality."""

from __future__ import annotations

import sys

import numpy as np
import requests

from fsc_analysis.utils import anchor_curve, classify_fsc_curve, draw_typicality_bar, fetch_fsc_curve, load_models


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <EMD_ID>")
        sys.exit(1)

    emd_id = sys.argv[1]

    try:
        fsc_curve = fetch_fsc_curve(emd_id)
    except requests.RequestException as exc:
        print(f"Network error fetching {emd_id}: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    models = load_models()
    processed = anchor_curve(np.array(fsc_curve))
    cluster_id, distance, typicality = classify_fsc_curve(processed, models)

    print(f"Cluster ID: {cluster_id}")
    print(f"Distance from centroid: {distance:.4f}")
    print(f"Typicality: {typicality * 100:.2f}%")
    draw_typicality_bar(typicality)


if __name__ == "__main__":
    main()