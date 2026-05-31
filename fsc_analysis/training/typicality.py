"""Calculate typicality scores from trained clustering outputs.

Typicality is defined on a single, shared scale: the Euclidean distance from a
curve's assigned cluster centroid in latent space, scaled by that cluster's
persisted distance threshold (``mean + 1.5*std``):

    typicality = clip(1 - distance / threshold, 0, 1)

A score of 1.0 means the curve sits on its cluster centroid; 0.0 means it lies
at or beyond the atypicality threshold. This is the identical definition used by
the single-curve ``fsc-assess`` tool, so the experiment and the user-facing tool
report typicality the same way.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsc_analysis.training.train_kmeans import edit_curve_based_on_crossing
from fsc_analysis.utils import load_models, normalize_curve

fsc_curve_type = 'fsc_masked'

_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def main() -> None:
    print(f'Loading models for {fsc_curve_type}...')
    models = load_models(curve_type=fsc_curve_type)

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

    print(f"\n{'=' * 70}")
    print(f'CALCULATING TYPICALITY SCORES - {fsc_curve_type}')
    print(f"{'=' * 70}\n")

    thresholds = np.array(
        [float(models.cluster_thresholds.get(int(c), np.nan)) for c in labels]
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        typicality_scores = np.clip(1.0 - distances / thresholds, 0.0, 1.0)
    typicality_scores = np.where(np.isfinite(thresholds), typicality_scores, 0.0)
    typicality_class = np.where(distances <= thresholds, 'Typical', 'Atypical')

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        'cluster': labels,
        'distance': distances,
        'threshold': thresholds,
        'typicality_score': typicality_scores,
        'typicality_class': typicality_class,
        'curve': list(normalized_data),
    })

    scores_path = _OUTPUT_DIR / f'typicality_scores_{fsc_curve_type}.csv'
    results_df.to_csv(scores_path, index=False)
    print(f'✓ Saved typicality scores to: {scores_path}')

    print(f"\n{'=' * 70}")
    print('TYPICALITY SUMMARY STATISTICS')
    print(f"{'=' * 70}\n")

    print(f'Total curves analyzed: {len(results_df)}')
    print('\nCounts by typicality class:')
    for class_name in ['Typical', 'Atypical']:
        count = (results_df['typicality_class'] == class_name).sum()
        pct = 100 * count / len(results_df)
        print(f'  {class_name:20s}: {count:5d} ({pct:5.1f}%)')

    print('\nTypicality score statistics:')
    print(f"  Mean:   {results_df['typicality_score'].mean():.4f}")
    print(f"  Std:    {results_df['typicality_score'].std():.4f}")
    print(f"  Min:    {results_df['typicality_score'].min():.4f}")
    print(f"  Max:    {results_df['typicality_score'].max():.4f}")
    print(f"  Median: {results_df['typicality_score'].median():.4f}")

    print('\nPer-cluster breakdown:')
    for cluster_id in sorted(results_df['cluster'].unique()):
        cluster_slice = results_df[results_df['cluster'] == cluster_id]
        print(f'\n  Cluster {cluster_id}:')
        print(f'    Count: {len(cluster_slice)}')
        print(
            f"    Typicality - Mean: {cluster_slice['typicality_score'].mean():.4f}, "
            f"Std: {cluster_slice['typicality_score'].std():.4f}"
        )
        for class_name in cluster_slice['typicality_class'].unique():
            count = (cluster_slice['typicality_class'] == class_name).sum()
            pct = 100 * count / len(cluster_slice)
            print(f'      {class_name:20s}: {count:3d} ({pct:5.1f}%)')

    print(f"\n{'=' * 70}")
    print('CREATING VISUALIZATIONS')
    print(f"{'=' * 70}\n")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    clusters = sorted(results_df['cluster'].unique())
    data_by_cluster = [results_df[results_df['cluster'] == c]['typicality_score'].values for c in clusters]
    ax.violinplot(data_by_cluster, positions=clusters, showmeans=True, showmedians=True)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Typicality Score')
    ax.set_title(f'Distribution of Typicality Scores per Cluster - {fsc_curve_type}')
    ax.set_xticks(clusters)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.hist(results_df['typicality_score'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(results_df['typicality_score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(results_df['typicality_score'].median(), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Typicality Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Overall Distribution of Typicality Scores - {fsc_curve_type}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    dist_path = _OUTPUT_DIR / f'typicality_distribution_{fsc_curve_type}.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {dist_path}')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {'Typical': 'green', 'Atypical': 'red'}
    for class_name in ['Atypical', 'Typical']:
        class_data = results_df[results_df['typicality_class'] == class_name]
        ax.scatter(
            class_data['distance'],
            class_data['typicality_score'],
            label=f'{class_name} (n={len(class_data)})',
            color=colors[class_name],
            alpha=0.6,
            s=30,
        )
    ax.set_xlabel('Distance from Cluster Centroid')
    ax.set_ylabel('Typicality Score')
    ax.set_title(f'Typicality Score vs. Distance - {fsc_curve_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    vs_dist_path = _OUTPUT_DIR / f'typicality_vs_distance_{fsc_curve_type}.png'
    plt.savefig(vs_dist_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {vs_dist_path}')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_class_counts = pd.crosstab(results_df['cluster'], results_df['typicality_class'])
    cluster_class_counts = cluster_class_counts.reindex(columns=['Typical', 'Atypical'], fill_value=0)
    cluster_class_counts.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red'])
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Count')
    ax.set_title(f'Typicality Class Distribution per Cluster - {fsc_curve_type}')
    ax.legend(title='Typicality Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    by_cluster_path = _OUTPUT_DIR / f'typicality_by_cluster_{fsc_curve_type}.png'
    plt.savefig(by_cluster_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {by_cluster_path}')
    plt.close()

    print(f"\n{'=' * 70}")
    print('✓ COMPLETE!')
    print(f"{'=' * 70}")
    print('\nOutput files:')
    print(f'  - {scores_path}')
    print(f'  - {dist_path}')
    print(f'  - {vs_dist_path}')
    print(f'  - {by_cluster_path}')


if __name__ == '__main__':
    main()
