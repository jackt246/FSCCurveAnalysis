"""
Interactive tool to inspect clusters and define typicality classes and thresholds.

This script helps you:
1. Visualize each cluster's characteristics
2. Classify clusters as Typical/Atypical
3. Define distance thresholds for typical classes
4. Understand distance distributions per cluster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import joblib
from tensorflow.keras.models import load_model

# Configuration - modify these based on your analysis
fsc_curve_type = 'fsc_masked'

# Load the models and data
print(f"Loading models for {fsc_curve_type}...")
encoder = load_model(f"encoder_model_{fsc_curve_type}.h5")
kmeans = joblib.load(f"kmeans_model_{fsc_curve_type}.pkl")
cluster_summary = pd.read_csv(f"cluster_summary_{fsc_curve_type}.csv")

# Load cluster data
cluster_data = pd.read_json('fsc_curves/fsc_curves_all.json')
cluster_data[fsc_curve_type] = cluster_data[fsc_curve_type].apply(np.asarray)
fsc_data = cluster_data[fsc_curve_type].dropna()

# Resample curves (use the same function from main script)
from ModelTraining_clustering_Kmeans import edit_curve_based_on_crossing, resample_curve

resampled_curves = [edit_curve_based_on_crossing(c) for c in fsc_data]
resampled_data = np.vstack(resampled_curves).astype(np.float32)

# Normalize
data_min = np.min(resampled_data)
data_max = np.max(resampled_data)
normalized_data = (resampled_data - data_min) / (data_max - data_min)

# Get embeddings and labels
refined_embeddings = encoder.predict(normalized_data)
labels = kmeans.predict(refined_embeddings)

# Calculate distances
centroids = kmeans.cluster_centers_[labels]
distances = np.sqrt(np.sum((refined_embeddings - centroids) ** 2, axis=1))

# Create analysis DataFrame
analysis_df = pd.DataFrame({
    'cluster': labels,
    'distance': distances,
    'curve': list(normalized_data)
})

print(f"\n{'='*70}")
print(f"CLUSTER INSPECTION REPORT - {fsc_curve_type}")
print(f"{'='*70}\n")

# Analyze each cluster
cluster_stats = {}
for cluster_id in sorted(analysis_df['cluster'].unique()):
    cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
    distances_in_cluster = cluster_data['distance'].values

    mean_dist = np.mean(distances_in_cluster)
    std_dist = np.std(distances_in_cluster)
    min_dist = np.min(distances_in_cluster)
    max_dist = np.max(distances_in_cluster)
    median_dist = np.median(distances_in_cluster)

    # Suggested threshold: mean + 1.5*std (captures ~93% if normal distribution)
    suggested_threshold = mean_dist + 1.5 * std_dist

    cluster_stats[cluster_id] = {
        'size': len(cluster_data),
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'median_distance': median_dist,
        'suggested_threshold': suggested_threshold,
        'percentile_90': np.percentile(distances_in_cluster, 90),
        'percentile_95': np.percentile(distances_in_cluster, 95),
    }

    print(f"CLUSTER {cluster_id}:")
    print(f"  Size: {len(cluster_data)} curves")
    print(f"  Distance Statistics:")
    print(f"    Mean: {mean_dist:.4f}")
    print(f"    Std:  {std_dist:.4f}")
    print(f"    Range: [{min_dist:.4f}, {max_dist:.4f}]")
    print(f"    Median: {median_dist:.4f}")
    print(f"    90th percentile: {np.percentile(distances_in_cluster, 90):.4f}")
    print(f"    95th percentile: {np.percentile(distances_in_cluster, 95):.4f}")
    print(f"  SUGGESTED THRESHOLD: {suggested_threshold:.4f}")
    print(f"    (mean + 1.5*std, captures ~93% of curves as typical)")
    print()

# Create visualization of distance distributions per cluster
print(f"\n{'='*70}")
print(f"Creating distance distribution visualizations...")
print(f"{'='*70}\n")

num_clusters = len(cluster_stats)
cols = 5
rows = (num_clusters + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
axes = axes.flatten()

for idx, cluster_id in enumerate(sorted(cluster_stats.keys())):
    ax = axes[idx]
    distances_in_cluster = analysis_df[analysis_df['cluster'] == cluster_id]['distance'].values

    stats_dict = cluster_stats[cluster_id]

    # Plot histogram
    ax.hist(distances_in_cluster, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

    # Add lines for statistics
    ax.axvline(stats_dict['mean_distance'], color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(stats_dict['median_distance'], color='green', linestyle='--', linewidth=2, label='Median')
    ax.axvline(stats_dict['suggested_threshold'], color='orange', linestyle='--', linewidth=2, label='Suggested Threshold')

    ax.set_title(f"Cluster {cluster_id} (n={stats_dict['size']})")
    ax.set_xlabel("Distance from Centroid")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(cluster_stats), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f"Distance Distributions per Cluster - {fsc_curve_type}", fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(f"cluster_distance_distributions_{fsc_curve_type}.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: cluster_distance_distributions_{fsc_curve_type}.png")
plt.close()

# Save the analysis to CSV for reference
stats_df = pd.DataFrame(cluster_stats).T
stats_df.to_csv(f"cluster_statistics_{fsc_curve_type}.csv")
print(f"✓ Saved: cluster_statistics_{fsc_curve_type}.csv")

print(f"\n{'='*70}")
print(f"NEXT STEPS:")
print(f"{'='*70}")
print("""
1. INSPECT THE VISUALIZATIONS
   - Look at cluster_distance_distributions_{fsc_curve_type}.png
   - Look at your existing cluster_subplots plots
   - Identify which clusters look like "good" FSC curves

2. CLASSIFY YOUR CLUSTERS
   Edit the dictionaries below based on your visual inspection:
   
   TYPICAL_CLUSTERS = [0, 3, 5]       # Clusters that represent good FSC curves
   ATYPICAL_CLUSTERS = [1, 9]         # Clusters that represent bad/noisy FSC curves
   
3. DEFINE THRESHOLDS
   For each TYPICAL cluster, choose a threshold:
   - Use the "Suggested Threshold" (mean + 1.5*std) as a starting point
   - Or pick a different percentile from the histogram
   - Or pick a specific value from the statistics
   
   CLASS_THRESHOLDS = {
       0: 0.45,     # Cluster 0 is typical if distance < 0.45
       3: 0.38,     # Cluster 3 is typical if distance < 0.38
       5: 0.52,     # etc.
   }

4. RUN THE TYPICALITY SCRIPT
   Once you've defined the classes, run:
   python CalculateTypicalityScores.py

""")

print(f"Reference: cluster_statistics_{fsc_curve_type}.csv contains all statistics")
print(f"Reference: cluster_distance_distributions_{fsc_curve_type}.png shows the distributions")

