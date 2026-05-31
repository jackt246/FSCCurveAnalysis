# FSC Analysis

FSC Analysis provides command-line tools and training scripts for working with Fourier Shell Correlation (FSC) curves from cryo-EM map validation data. FSC curves summarise how consistently two half-maps agree across spatial frequencies, making them useful for assessing map quality, clustering curve shapes, and estimating how typical a structure looks relative to a reference corpus.

## Features

- Download FSC validation data from EMDB
- Classify single FSC curves by learned cluster typicality
- Process batches of EMDB IDs and generate plots
- Train clustering and neural-network models on curated FSC datasets
- Explore typicality definitions and cluster inspection workflows

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

The repository includes a `.python-version` file pinned to Python 3.11.

## Setup with uv

```bash
uv sync
```

Run tools inside the managed environment with `uv run`:

```bash
uv run fsc-assess EMD-14046
uv run fsc-batch data/examples/curves_example_batch.csv
uv run python -m fsc_analysis.training.train_kmeans
```

## Directory layout

```text
FSCCurveAnalysis/
├── fsc_analysis/          # Installable Python package
│   ├── utils.py           # Shared fetching, preprocessing, and inference helpers
│   ├── assess.py          # Single-curve CLI
│   ├── batch.py           # Batch CLI
│   ├── batch_anchored.py  # Anchored batch CLI with plots
│   ├── downloader.py      # EMDB FSC downloader
│   └── training/          # Model training and inspection scripts
├── data/                  # Input datasets used by training scripts
├── models/                # Trained model artefacts loaded by the CLIs
├── outputs/               # Generated plots and derived arrays
├── docs/                  # Typicality workflow reference material
└── tests/                 # Unit tests (pytest)
```

## Running the tests

Unit tests cover the pure preprocessing and classification helpers
(`resample_curve`, `find_crossing_point`, `anchor_curve`, `classify_fsc_curve`,
`fetch_fsc_curve`, `draw_typicality_bar`, and `is_valid_number_array`). Model
loading and HTTP access are mocked, so the suite runs without TensorFlow,
network access, or trained artefacts.

```bash
uv run --group dev pytest
```

## CLI usage

### Assess a single EMDB entry

```bash
uv run fsc-assess EMD-14046
```

This fetches the FSC curve from EMDB, anchors it to the 0.143 crossing, encodes it with the trained model, and reports the assigned cluster and percentile-based typicality.

### Assess a batch of entries

```bash
uv run fsc-batch data/examples/curves_example_batch.csv
```

The input file should contain one EMDB accession per line. The command prints per-entry results and saves `fsc_assessment_batch.png`.

### Assess a batch with anchored visualisations

```bash
uv run fsc-batch-anchored data/examples/curves_example_batch.csv
```

This variant also generates:

- `fsc_assessment_batch_anchored.png`
- `fsc_colored_by_typicality.png`
- `fsc_typicality_violin.png`

### Download FSC data from EMDB

```bash
uv run fsc-download
```

The downloader resumes from `fsc_curves_partial.json` when present and writes the consolidated export to `data/fsc_curves_all.json`.

## Training workflow

Training scripts live under `fsc_analysis/training/` and are run from the project root with `uv run python -m ...`.

### Building a clustering figure set for a paper

This is the end-to-end workflow for showing how different FSC curve types cluster:

1. **Download the FSC curves from EMDB**

   ```bash
   uv run fsc-download
   ```

   Writes `data/fsc_curves_all.json` (resumes from `fsc_curves_partial.json` if interrupted).

2. **Train clustering on a chosen curve type**

   Edit `fsc_curve_type` near the top of `fsc_analysis/training/train_kmeans.py`:

   ```python
   # Options: 'fsc_unmasked', 'fsc_masked', 'fsc_corrected', 'fsc_phaserandom'
   fsc_curve_type = 'fsc_masked'
   ```

   Then run:

   ```bash
   uv run python -m fsc_analysis.training.train_kmeans
   ```

3. **Compare across curve types**

   Repeat step 2 for each of the four `fsc_curve_type` values. All artefacts are
   suffixed by curve type, so nothing is overwritten and the four runs can be
   compared side by side.

   Generated plots in `outputs/`:

   - `cluster_subplots_<type>.png` — every curve per cluster with its mean
   - `cluster_subplots_dist_colored_<type>.png` — curves coloured by distance from the cluster centroid
   - `cluster_averages_<type>.png` — mean curve per cluster
   - `elbow_method_analysis_<type>.png` — inertia vs. K, for choosing the cluster count

   Model artefacts in `models/`:

   - `encoder_model_<type>.h5`
   - `kmeans_model_<type>.pkl`
   - `cluster_frequencies_<type>.csv`
   - `cluster_summary_<type>.csv`

### Other training and analysis scripts

- Alternative clustering methods: `train_dbscan`, `train_fuzzy`, `train_cnn`, `similarity_search`
- Curve preprocessing: `normalise_curves`
- Cluster inspection: `inspect_clusters`
- Typicality definitions and comparison: `test_configurations`, `typicality`

```bash
uv run python -m fsc_analysis.training.inspect_clusters
```

Supporting workflow notes are in `docs/CONFIGURATION_TEMPLATE.md` and `docs/TYPICALITY_WORKFLOW_GUIDE.md`.

## Model artefacts

The runtime CLI tools (`fsc-assess`, `fsc-batch`, `fsc-batch-anchored`) load trained
artefacts from `models/`:

- `encoder_model.h5`
- `kmeans_model.pkl`
- `cluster_frequencies.csv`
- `cluster_summary.csv`

Training (`train_kmeans`) writes curve-type-suffixed artefacts (e.g.
`encoder_model_fsc_masked.h5`) into `models/`. To use a freshly trained model with
the assessment CLIs, copy or rename the suffixed files to the unsuffixed names above.
