"""
Quick batch processor for calculating typicality scores with multiple threshold configurations.

This is useful for testing different threshold combinations to see how they affect the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

fsc_curve_type = 'fsc_masked'

# Define multiple configurations to test
# Each configuration is a tuple of (TYPICAL_CLUSTERS, ATYPICAL_CLUSTERS, CLASS_THRESHOLDS)
CONFIGURATIONS = {
    'conservative': {
        'description': 'Strict thresholds - fewer curves classified as typical',
        'typical_clusters': [4, 9, 0, 8, 6, 1, 5],
        'atypical_clusters': [3, 2, 7],
        'thresholds': {
            4: 0.35,
            9: 0.35,
            0: 0.40,
        }
    },
    'moderate': {
        'description': 'Medium thresholds - balanced approach',
        'typical_clusters': [0, 3, 5],
        'atypical_clusters': [1, 2, 9],
        'thresholds': {
            0: 0.50,
            3: 0.40,
            5: 0.55,
        }
    },
    'lenient': {
        'description': 'Loose thresholds - more curves classified as typical',
        'typical_clusters': [0, 3, 5],
        'atypical_clusters': [1, 2, 9],
        'thresholds': {
            0: 0.65,
            3: 0.55,
            5: 0.70,
        }
    },
}

# ============================================================================
# Load data
# ============================================================================

print(f"Loading models for {fsc_curve_type}...")
encoder = load_model(f"encoder_model_{fsc_curve_type}.h5")
kmeans = joblib.load(f"kmeans_model_{fsc_curve_type}.pkl")

cluster_data = pd.read_json('fsc_curves/fsc_curves_all.json')
cluster_data[fsc_curve_type] = cluster_data[fsc_curve_type].apply(np.asarray)
fsc_data = cluster_data[fsc_curve_type].dropna()

from ModelTraining_clustering_Kmeans import edit_curve_based_on_crossing

resampled_curves = [edit_curve_based_on_crossing(c) for c in fsc_data]
resampled_data = np.vstack(resampled_curves).astype(np.float32)

data_min = np.min(resampled_data)
data_max = np.max(resampled_data)
normalized_data = (resampled_data - data_min) / (data_max - data_min)

refined_embeddings = encoder.predict(normalized_data)
labels = kmeans.predict(refined_embeddings)
centroids = kmeans.cluster_centers_[labels]
distances = np.sqrt(np.sum((refined_embeddings - centroids) ** 2, axis=1))

# ============================================================================
# Calculate typicality for all configurations
# ============================================================================

print(f"\n{'='*70}")
print(f"TESTING {len(CONFIGURATIONS)} CONFIGURATIONS")
print(f"{'='*70}\n")

all_results = {}

for config_name, config in CONFIGURATIONS.items():
    print(f"Processing: {config_name}")
    print(f"  Description: {config['description']}")

    TYPICAL_CLUSTERS = config['typical_clusters']
    ATYPICAL_CLUSTERS = config['atypical_clusters']
    CLASS_THRESHOLDS = config['thresholds']

    # Calculate scores
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
    })

    all_results[config_name] = results_df

    # Print summary
    print(f"  Results:")
    print(f"    Total curves: {len(results_df)}")
    for class_name in ['Typical', 'Atypical', 'Atypical_Outlier', 'Mixed']:
        count = (results_df['typicality_class'] == class_name).sum()
        pct = 100 * count / len(results_df)
        print(f"    {class_name:20s}: {count:5d} ({pct:5.1f}%)")
    print(f"    Mean typicality: {results_df['typicality_score'].mean():.4f}")
    print()

# ============================================================================
# Create comparison visualizations
# ============================================================================

print(f"\n{'='*70}")
print(f"CREATING COMPARISON VISUALIZATIONS")
print(f"{'='*70}\n")

# 1. Side-by-side histograms
fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 5))
if len(all_results) == 1:
    axes = [axes]

for idx, (config_name, results_df) in enumerate(all_results.items()):
    ax = axes[idx]
    ax.hist(results_df['typicality_score'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(results_df['typicality_score'].mean(), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Typicality Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f"{config_name.upper()}\n{results_df['typicality_class'].value_counts()['Typical']} Typical")
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f"Typicality Distribution Comparison - {fsc_curve_type}", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"typicality_comparison_distributions_{fsc_curve_type}.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: typicality_comparison_distributions_{fsc_curve_type}.png")
plt.close()

# 2. Comparison table
comparison_data = []
for config_name, results_df in all_results.items():
    comparison_data.append({
        'Configuration': config_name,
        'Typical': (results_df['typicality_class'] == 'Typical').sum(),
        'Atypical': (results_df['typicality_class'] == 'Atypical').sum(),
        'Outliers': (results_df['typicality_class'] == 'Atypical_Outlier').sum(),
        'Mixed': (results_df['typicality_class'] == 'Mixed').sum(),
        'Mean Score': f"{results_df['typicality_score'].mean():.4f}",
        'Median Score': f"{results_df['typicality_score'].median():.4f}",
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(f"typicality_configurations_comparison_{fsc_curve_type}.csv", index=False)
print(f"✓ Saved: typicality_configurations_comparison_{fsc_curve_type}.csv")

print(f"\nComparison Table:")
print(comparison_df.to_string(index=False))

# 3. Save best configuration (you can modify this logic)
print(f"\n{'='*70}")
print(f"RECOMMENDATION")
print(f"{'='*70}")
print(f"""
Review the generated files:
  1. typicality_comparison_distributions_{fsc_curve_type}.png
  2. typicality_configurations_comparison_{fsc_curve_type}.csv

Then decide which configuration best matches your needs:
  - 'conservative': Fewer false positives, stricter typical/atypical boundary
  - 'moderate': Balanced approach
  - 'lenient': More curves classified as typical, looser boundary

Once you've chosen, update CalculateTypicalityScores.py with those thresholds
and run it to generate the final outputs.
""")

