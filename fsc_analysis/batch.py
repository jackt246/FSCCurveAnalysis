"""Classify a batch of FSC curves and plot their typicality distribution."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import requests

from fsc_analysis.utils import anchor_curve, classify_fsc_curve, draw_typicality_bar, fetch_fsc_curve, load_models


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    if not os.path.isfile(input_csv):
        print(f"Input file {input_csv} does not exist.")
        sys.exit(1)

    with open(input_csv) as f:
        emd_ids = [line.strip() for line in f if line.strip()]

    models = load_models()
    results: list[tuple[str, float]] = []

    for emd_id in emd_ids:
        try:
            fsc_curve = fetch_fsc_curve(emd_id)
        except (requests.RequestException, ValueError) as exc:
            print(f"Skipping {emd_id}: {exc}")
            continue

        processed = anchor_curve(np.array(fsc_curve))
        cluster_id, distance, typicality = classify_fsc_curve(processed, models)
        print(f"{emd_id}: Cluster ID = {cluster_id}, Distance = {distance:.4f}, Typicality = {typicality * 100:.2f}%")
        draw_typicality_bar(typicality)
        results.append((emd_id, typicality))

    if not results:
        print("No curves were successfully processed.")
        return

    sorted_results = sorted(results, key=lambda x: x[1])
    emd_ids_sorted, percentiles_sorted = zip(*sorted_results)

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(percentiles_sorted)), percentiles_sorted, c="blue")
    for i, label in enumerate(emd_ids_sorted):
        y_val = percentiles_sorted[i]
        y_offset = 0.02 if i % 2 == 0 else -0.02
        va = "bottom" if i % 2 == 0 else "top"
        plt.text(i, y_val + y_offset, label, fontsize=6, rotation=90, ha="center", va=va)

    plt.xlabel("FSC Curve Index (sorted by typicality)")
    plt.ylabel("Typicality Percentile")
    plt.title("Typicality of FSC Curves by EMD ID")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("fsc_assessment_batch.png", dpi=300)
    print("Plot saved to fsc_assessment_batch.png")


if __name__ == "__main__":
    main()