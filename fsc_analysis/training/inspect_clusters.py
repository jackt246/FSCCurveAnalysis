"""Inspect cluster distance distributions for typicality threshold selection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsc_analysis.training.train_kmeans import edit_curve_based_on_crossing
from fsc_analysis.utils import load_models, normalize_curve

fsc_curve_type = 'fsc_masked'

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def main() -> None:
    print(f'Loading models for {fsc_curve_type}...')
    models = load_models(curve_type=fsc_curve_type)
    cluster_summary = pd.read_csv(_MODEL_DIR / f'cluster_frequencies_{fsc_curve_type}.csv')

    cluster_data = pd.read_json('data/fsc_curves_all.json')
    cluster_data[fsc_curve_type] = cluster_data[fsc_curve_type].apply(np.asarray)
    fsc_data = cluster_data[fsc_curve_type].dropna()

    resampled_curves = [edit_curve_based_on_crossing(c) for c in fsc_data]
    resampled_data = np.vstack(resampled_curves).astype(np.float32)

    normalized_data = normalize_curve(resampled_data, models.data_min, models.data_max)

    refined_embeddings = models.encoder.predict(normalized_data)
    labels = models.kmeans.predict(refined_embeddings)

    centroids = models.kmeans.cluster_centers_[labels]
    distances = np.sqrt(np.sum((refined_embeddings - centroids) ** 2, axis=1))

    analysis_df = pd.DataFrame({
        'cluster': labels,
        'distance': distances,
        'curve': list(normalized_data),
    })

    print(f"\n{'=' * 70}")
    print(f'CLUSTER INSPECTION REPORT - {fsc_curve_type}')
    print(f"{'=' * 70}\n")
    print(f'Loaded cluster summary with {len(cluster_summary)} rows.')

    cluster_stats = {}
    for cluster_id in sorted(analysis_df['cluster'].unique()):
        cluster_slice = analysis_df[analysis_df['cluster'] == cluster_id]
        distances_in_cluster = cluster_slice['distance'].values

        mean_dist = np.mean(distances_in_cluster)
        std_dist = np.std(distances_in_cluster)
        min_dist = np.min(distances_in_cluster)
        max_dist = np.max(distances_in_cluster)
        median_dist = np.median(distances_in_cluster)
        suggested_threshold = mean_dist + 1.5 * std_dist

        cluster_stats[cluster_id] = {
            'size': len(cluster_slice),
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'min_distance': min_dist,
            'max_distance': max_dist,
            'median_distance': median_dist,
            'suggested_threshold': suggested_threshold,
            'percentile_90': np.percentile(distances_in_cluster, 90),
            'percentile_95': np.percentile(distances_in_cluster, 95),
        }

        print(f'CLUSTER {cluster_id}:')
        print(f'  Size: {len(cluster_slice)} curves')
        print('  Distance Statistics:')
        print(f'    Mean: {mean_dist:.4f}')
        print(f'    Std:  {std_dist:.4f}')
        print(f'    Range: [{min_dist:.4f}, {max_dist:.4f}]')
        print(f'    Median: {median_dist:.4f}')
        print(f"    90th percentile: {np.percentile(distances_in_cluster, 90):.4f}")
        print(f"    95th percentile: {np.percentile(distances_in_cluster, 95):.4f}")
        print(f'  SUGGESTED THRESHOLD: {suggested_threshold:.4f}')
        print('    (mean + 1.5*std, captures ~93% of curves as typical)')
        print()

    print(f"\n{'=' * 70}")
    print('Creating distance distribution visualizations...')
    print(f"{'=' * 70}\n")

    num_clusters = len(cluster_stats)
    cols = 5
    rows = (num_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, cluster_id in enumerate(sorted(cluster_stats.keys())):
        ax = axes[idx]
        distances_in_cluster = analysis_df[analysis_df['cluster'] == cluster_id]['distance'].values
        stats_dict = cluster_stats[cluster_id]

        ax.hist(distances_in_cluster, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(stats_dict['mean_distance'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(stats_dict['median_distance'], color='green', linestyle='--', linewidth=2, label='Median')
        ax.axvline(stats_dict['suggested_threshold'], color='orange', linestyle='--', linewidth=2, label='Suggested Threshold')
        ax.set_title(f"Cluster {cluster_id} (n={stats_dict['size']})")
        ax.set_xlabel('Distance from Centroid')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(cluster_stats), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Distance Distributions per Cluster - {fsc_curve_type}', fontsize=16, y=1.00)
    plt.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dist_plot_path = _OUTPUT_DIR / f'cluster_distance_distributions_{fsc_curve_type}.png'
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {dist_plot_path}')
    plt.close()

    stats_df = pd.DataFrame(cluster_stats).T
    stats_csv_path = _OUTPUT_DIR / f'cluster_statistics_{fsc_curve_type}.csv'
    stats_df.to_csv(stats_csv_path)
    print(f'✓ Saved: {stats_csv_path}')

    print(f"\n{'=' * 70}")
    print('NEXT STEPS:')
    print(f"{'=' * 70}")
    print(
        f"""
1. INSPECT THE VISUALIZATIONS
   - Look at cluster_distance_distributions_{fsc_curve_type}.png
   - Look at your existing cluster_subplots plots
   - Identify which clusters look like "good" FSC curves

2. CLASSIFY YOUR CLUSTERS
   Edit the dictionaries below based on your visual inspection:

   TYPICAL_CLUSTERS = [0, 3, 5]
   ATYPICAL_CLUSTERS = [1, 9]

3. DEFINE THRESHOLDS
   For each TYPICAL cluster, choose a threshold:
   - Use the "Suggested Threshold" (mean + 1.5*std) as a starting point
   - Or pick a different percentile from the histogram
   - Or pick a specific value from the statistics

   CLASS_THRESHOLDS = {{
       0: 0.45,
       3: 0.38,
       5: 0.52,
   }}

4. RUN THE TYPICALITY SCRIPT
   Once you've defined the classes, run:
   python -m fsc_analysis.training.typicality
"""
    )

    print(f'Reference: cluster_statistics_{fsc_curve_type}.csv contains all statistics')
    print(f'Reference: cluster_distance_distributions_{fsc_curve_type}.png shows the distributions')


if __name__ == '__main__':
    main()
