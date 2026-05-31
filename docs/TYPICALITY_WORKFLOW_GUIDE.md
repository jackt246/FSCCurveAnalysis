# Typicality Scoring Workflow Guide

## Overview

This workflow turns a corpus of FSC curves into a single, quantitative
**typicality score (0–1)** for any curve. The score is defined consistently for
both the population experiment and the single-curve user tool, so the numbers
mean the same thing in both places.

## How typicality is defined

Typicality is the distance from a curve's assigned cluster centroid in the
encoder's latent space, scaled by that cluster's distance threshold:

```
typicality = clip(1 - distance / threshold, 0, 1)
```

- **1.0** — the curve sits exactly on its cluster centroid (most typical).
- **0.5** — the curve is halfway to the cluster's atypicality threshold.
- **0.0** — the curve is at or beyond the threshold (atypical / outlier).

The per-cluster `threshold` is `mean + 1.5*std` of the training curves'
distances within that cluster. Thresholds are computed once at training time and
persisted, so no manual cluster labelling is required and the definition is
identical everywhere.

All randomness is seeded (`set_seeds(42)`), so retraining on the same data
reproduces the same clusters, thresholds, and scores.

## Artefacts produced by training

`train_kmeans` writes the following into `models/` (suffixed by curve type, e.g.
`fsc_masked`):

| File | Purpose |
| --- | --- |
| `encoder_model_{type}.h5` | Trained autoencoder encoder |
| `kmeans_model_{type}.pkl` | Fitted KMeans model (centroids) |
| `normalisation_{type}.json` | Global `data_min`/`data_max` for inference normalisation |
| `cluster_distance_stats_{type}.csv` | Per-cluster distance stats + typicality `threshold` |
| `cluster_frequencies_{type}.csv` | Curve counts per cluster |

These five artefacts are everything the experiment scripts and the `fsc-assess`
tool need; both read the **same** normalisation parameters and thresholds.

## Step 1 — Train the models

```bash
uv run python -m fsc_analysis.training.train_kmeans
```

Run from the repository root (the script reads `data/` and writes to `models/`
and `outputs/`). The curve type is set by the `fsc_curve_type` variable at the
top of the script (default `fsc_masked`).

## Step 2 — (Optional) Inspect cluster distances

```bash
uv run python -m fsc_analysis.training.inspect_clusters
```

Loads the trained models from `models/`, reuses the persisted normalisation, and
writes diagnostics to `outputs/`:

- `cluster_distance_distributions_{type}.png` — distance histograms per cluster
- `cluster_statistics_{type}.csv` — per-cluster distance statistics

Use these to sanity-check that clusters are coherent and thresholds look sensible.

## Step 3 — Score the population

```bash
uv run python -m fsc_analysis.training.typicality
```

Loads the trained models, applies the unified centroid-distance definition to
every curve, and writes to `outputs/`:

- `typicality_scores_{type}.csv` — per-curve cluster, distance, threshold, score, class
- `typicality_distribution_{type}.png` — score distributions (per cluster + overall)
- `typicality_vs_distance_{type}.png` — score vs. centroid distance
- `typicality_by_cluster_{type}.png` — Typical/Atypical breakdown per cluster

A curve is labelled **Typical** when `distance <= threshold` and **Atypical**
otherwise.

## Step 4 — Assess a user's own curve

The single-curve tool uses the identical definition and the same persisted
artefacts:

```bash
uv run fsc-assess EMD-1234
```

It fetches the curve, anchors it, normalises it with the saved
`data_min`/`data_max`, encodes it, finds its nearest cluster centroid, and
reports the centroid distance and the 0–1 typicality score.

For batches:

```bash
uv run fsc-batch
uv run fsc-batch-anchored
```

## Comparing curve types for a paper

To compare how different FSC curve types cluster, retrain per type by setting
`fsc_curve_type` (e.g. `fsc_masked`, `fsc_unmasked`, `fsc_phaserandom`) at the
top of `train_kmeans.py`, then re-run Steps 1–3. Each type produces its own
suffixed artefacts and figures, which you can place side by side.

## File reference

| File | Role | When to run |
| --- | --- | --- |
| `fsc_analysis/training/train_kmeans.py` | Train encoder + KMeans, persist artefacts | First |
| `fsc_analysis/training/inspect_clusters.py` | Inspect cluster distance distributions | Optional, after training |
| `fsc_analysis/training/typicality.py` | Score the whole population | After training |
| `fsc_analysis/assess.py` (`fsc-assess`) | Score a single user curve | Any time after training |
| `models/cluster_distance_stats_{type}.csv` | Per-cluster thresholds | Reference |
| `outputs/typicality_scores_{type}.csv` | Per-curve scores | Output |
