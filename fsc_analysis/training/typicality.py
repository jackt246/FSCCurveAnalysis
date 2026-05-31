"""Calculate typicality scores from trained clustering outputs."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from fsc_analysis.training.train_kmeans import edit_curve_based_on_crossing

fsc_curve_type = 'fsc_masked'

TYPICAL_CLUSTERS = [0, 3, 5]
ATYPICAL_CLUSTERS = [1, 2, 9]
CLASS_THRESHOLDS = {
    0: 0.45,
    3: 0.38,
    5: 0.52,
}


def main() -> None:
    print(f'Loading models for {fsc_curve_type}...')
    encoder = load_model(f'encoder_model_{fsc_curve_type}.h5')
    kmeans = joblib.load(f'kmeans_model_{fsc_curve_type}.pkl')

    cluster_data = pd.read_json('data/fsc_curves_all.json')
    cluster_data[fsc_curve_type] = cluster_data[fsc_curve_type].apply(np.asarray)
    fsc_data = cluster_data[fsc_curve_type].dropna()

    resampled_curves = [edit_curve_based_on_crossing(c) for c in fsc_data]
    resampled_data = np.vstack(resampled_curves).astype(np.float32)

    data_min = np.min(resampled_data)
    data_max = np.max(resampled_data)
    normalized_data = (resampled_data - data_min) / (data_max - data_min)

    refined_embeddings = encoder.predict(normalized_data)
    labels = kmeans.predict(refined_embeddings)
    centroids = kmeans.cluster_centers_[labels]
    distances = np.sqrt(np.sum((refined_embeddings - centroids) ** 2, axis=1))

    print(f"\n{'=' * 70}")
    print(f'CALCULATING TYPICALITY SCORES - {fsc_curve_type}')
    print(f"{'=' * 70}\n")

    typicality_scores = np.zeros(len(labels))
    typicality_class = [''] * len(labels)

    for i, (cluster_id, distance) in enumerate(zip(labels, distances)):
        if cluster_id in TYPICAL_CLUSTERS:
            threshold = CLASS_THRESHOLDS.get(cluster_id, np.inf)
            if distance <= threshold:
                typicality_scores[i] = 1.0 - (distance / threshold)
                typicality_class[i] = 'Typical'
            else:
                typicality_scores[i] = 0.0
                typicality_class[i] = 'Atypical_Outlier'
        elif cluster_id in ATYPICAL_CLUSTERS:
            typicality_scores[i] = 0.0
            typicality_class[i] = 'Atypical'
        else:
            typicality_scores[i] = 0.5
            typicality_class[i] = 'Mixed'

    results_df = pd.DataFrame({
        'cluster': labels,
        'distance': distances,
        'typicality_score': typicality_scores,
        'typicality_class': typicality_class,
        'curve': list(normalized_data),
    })

    results_df.to_csv(f'typicality_scores_{fsc_curve_type}.csv', index=False)
    print(f'✓ Saved typicality scores to: typicality_scores_{fsc_curve_type}.csv')

    print(f"\n{'=' * 70}")
    print('TYPICALITY SUMMARY STATISTICS')
    print(f"{'=' * 70}\n")

    print(f'Total curves analyzed: {len(results_df)}')
    print('\nCounts by typicality class:')
    for class_name in ['Typical', 'Atypical', 'Atypical_Outlier', 'Mixed']:
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
    plt.savefig(f'typicality_distribution_{fsc_curve_type}.png', dpi=300, bbox_inches='tight')
    print(f'✓ Saved: typicality_distribution_{fsc_curve_type}.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {
        'Typical': 'green',
        'Atypical': 'red',
        'Atypical_Outlier': 'orange',
        'Mixed': 'gray',
    }
    for class_name in ['Mixed', 'Atypical', 'Atypical_Outlier', 'Typical']:
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
    plt.savefig(f'typicality_vs_distance_{fsc_curve_type}.png', dpi=300, bbox_inches='tight')
    print(f'✓ Saved: typicality_vs_distance_{fsc_curve_type}.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_class_counts = pd.crosstab(results_df['cluster'], results_df['typicality_class'])
    cluster_class_counts.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red', 'orange', 'gray'])
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Count')
    ax.set_title(f'Typicality Class Distribution per Cluster - {fsc_curve_type}')
    ax.legend(title='Typicality Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'typicality_by_cluster_{fsc_curve_type}.png', dpi=300, bbox_inches='tight')
    print(f'✓ Saved: typicality_by_cluster_{fsc_curve_type}.png')
    plt.close()

    print(f"\n{'=' * 70}")
    print('✓ COMPLETE!')
    print(f"{'=' * 70}")
    print('\nOutput files:')
    print(f'  - typicality_scores_{fsc_curve_type}.csv')
    print(f'  - typicality_distribution_{fsc_curve_type}.png')
    print(f'  - typicality_vs_distance_{fsc_curve_type}.png')
    print(f'  - typicality_by_cluster_{fsc_curve_type}.png')


if __name__ == '__main__':
    main()
