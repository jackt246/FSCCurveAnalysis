"""Classify a batch of FSC curves with anchored alignment and generate typicality plots.

Produces three output plots:
- ``fsc_assessment_batch_anchored.png``: scatter of typicality percentiles
- ``fsc_colored_by_typicality.png``: all curves coloured by typicality
- ``fsc_typicality_violin.png``: violin distribution of typicality scores
"""

from __future__ import annotations

import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from matplotlib.cm import ScalarMappable

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
    aligned_curves: list[np.ndarray] = []

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
        aligned_curves.append(processed)

    if not results:
        print("No curves were successfully processed.")
        return

    _plot_typicality_scatter(results)
    _plot_curves_by_typicality(results, aligned_curves)
    _plot_typicality_violin(results)


def _plot_typicality_scatter(results: list[tuple[str, float]]) -> None:
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
    plt.savefig("fsc_assessment_batch_anchored.png", dpi=300)
    plt.close()
    print("Scatter plot saved to fsc_assessment_batch_anchored.png")


def _plot_curves_by_typicality(
    results: list[tuple[str, float]], aligned_curves: list[np.ndarray]
) -> None:
    norm = plt.Normalize(0, 1)
    cmap = cm.get_cmap("gist_rainbow")
    x = np.linspace(0, 1, 100)

    plt.figure(figsize=(12, 6))
    for (_, perc), curve in zip(results, aligned_curves):
        plt.plot(x, curve, color=cmap(norm(perc)), alpha=0.6)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([perc for _, perc in results])
    plt.colorbar(sm, ax=plt.gca(), label="Typicality Percentile")

    plt.title("FSC Curves Colored by Typicality")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("FSC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fsc_colored_by_typicality.png", dpi=300)
    plt.close()
    print("Color plot saved to fsc_colored_by_typicality.png")


def _plot_typicality_violin(results: list[tuple[str, float]]) -> None:
    percentiles = [perc for _, perc in results]

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=percentiles, orient="h", inner="quartile", color="skyblue")
    plt.xlabel("Typicality Percentile")
    plt.title("Distribution of Typicality Scores")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig("fsc_typicality_violin.png", dpi=300)
    plt.close()
    print("Violin plot saved to fsc_typicality_violin.png")


if __name__ == "__main__":
    main()