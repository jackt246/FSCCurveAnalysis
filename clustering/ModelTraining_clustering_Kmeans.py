import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
import umap
import pandas as pd
from scipy.interpolate import interp1d
import argparse
from mpl_toolkits.mplot3d import Axes3D
import joblib
from sklearn.metrics.pairwise import euclidean_distances

def elbow_analysis(refined_embeddings):
    print("Performing Elbow Analysis to find optimal number of clusters (K)")
    # Define the range of K values to test (e.g., from 1 to 30)
    max_k = 30
    inertia = []

    for k in range(1, max_k + 1):
        # Ensure n_init=20 is used for consistency
        kmeans_model_elbow = KMeans(n_clusters=k, n_init=20, random_state=42)
        kmeans_model_elbow.fit(refined_embeddings)
        inertia.append(kmeans_model_elbow.inertia_)

    # Plotting the Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title('Elbow Method: Inertia vs. Number of Clusters (K)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.grid(True)
    plt.xticks(np.arange(1, max_k + 1, 2))
    plt.axvline(x=10, color='r', linestyle='--', label=f'Chosen K={10}')  # Show current choice
    plt.legend()
    plt.savefig("elbow_method_analysis.png", dpi=300)
    plt.show()
    plt.close()

    print(f"Elbow analysis plot saved to elbow_method_analysis.png. Review plot to confirm optimal K.")

# Elbow analysis?
do_elbow_analysis = True

# Load an FSC curve
# Load raw FSC curves as list of lists
fsc_data = []
with open("fsc_curves/fsc_curves_normalisedandanchored.csv", "r") as f:
    for line in f:
        try:
            values = [float(x) for x in line.strip().split(",") if x]
            if len(values) > 1:  # skip empty or invalid lines
                fsc_data.append(values)
        except ValueError:
            print(f"⚠️ Skipping bad line: {line.strip()}")

# Step 1: Resample curves to same length

def resample_curve(curve, length=100):
    if len(curve) < 2:
        return np.full(length, np.nan)
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, length)
    # Crucially, change fill_value to 'nan' to avoid bad extrapolation
    f = interp1d(x_old, curve, kind='linear', fill_value='nan', bounds_error=False)
    return f(x_new)

resampled_data = np.array([resample_curve(c, 100) for c in fsc_data])  # Use first 50 points

# --- START: MISSING GLOBAL NORMALIZATION STEP ---

# Calculate the minimum and maximum across the entire dataset
data_min = np.min(resampled_data)
data_max = np.max(resampled_data)

# Scale all data simultaneously to the [0, 1] range
# This is crucial because your decoder's final layer uses 'sigmoid'
if data_max == data_min:
    # Handle the trivial case where all data is the same
    normalized_data = resampled_data
else:
    normalized_data = (resampled_data - data_min) / (data_max - data_min)

# IMPORTANT: Replace resampled_data with normalized_data for all subsequent steps
resampled_data = normalized_data

# --- END: MISSING GLOBAL NORMALIZATION STEP ---

# Step 2: Train a standard autoencoder
input_dim = resampled_data.shape[1]

encoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(64, activation='relu'),
    layers.Dense(20, activation='linear')  # latent vector
])

decoder = models.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(input_dim, activation='linear')  # we have already normalised the data globally to between 0 and 1, so switching from sigmoid to linear
])

autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
print("Training autoencoder")
autoencoder.fit(resampled_data, resampled_data, epochs=30, batch_size=128, verbose=1)

# Step 3: Get embeddings and cluster with KMeans
print("Generating embeddings and clustering with KMeans")
refined_embeddings = encoder.predict(resampled_data)

if do_elbow_analysis:
    elbow_analysis(refined_embeddings)

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
labels = kmeans.fit_predict(refined_embeddings)

# Optional: Visualize with UMAP
print("Projecting with UMAP")
from mpl_toolkits.mplot3d import Axes3D
umap_proj = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3).fit_transform(refined_embeddings)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(umap_proj[:, 0], umap_proj[:, 1], umap_proj[:, 2], c=labels, cmap='tab10', s=10)
ax.set_title("Clustering of FSC Curves (UMAP 3D)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
fig.colorbar(sc, ax=ax, label="Cluster")
plt.tight_layout()
plt.show()

# Store in a DataFrame if you want to explore later
cluster_df = pd.DataFrame({
    'cluster': labels,
    'embedding': list(refined_embeddings),
    'curve': list(resampled_data)
})

# Save encoder model
encoder.save("encoder_model.h5")

# Save KMeans model
import joblib
joblib.dump(kmeans, "kmeans_model.pkl")

# Save cluster metadata (e.g., frequencies for slider)
cluster_counts = cluster_df['cluster'].value_counts().sort_index()
cluster_counts.to_csv("cluster_frequencies.csv", header=["count"])
print("Models and cluster info saved.")


# Compute average curves and typicality for each cluster, save to one CSV
cluster_df = pd.DataFrame({
    'cluster': labels,
    'curve': list(resampled_data)
})

cluster_counts = cluster_df['cluster'].value_counts().sort_index()
total_curves = len(cluster_df)

output_rows = []
for cluster_id in cluster_counts.index:
    cluster_curves = np.stack(cluster_df[cluster_df['cluster'] == cluster_id]['curve'].values)
    avg_curve = cluster_curves.mean(axis=0)
    typicality = len(cluster_curves) / total_curves
    output_rows.append([cluster_id, typicality] + avg_curve.tolist())

# Save all to one CSV file
columns = ['cluster_id', 'typicality'] + [f'fsc_{i}' for i in range(resampled_data.shape[1])]
df_out = pd.DataFrame(output_rows, columns=columns)
df_out.to_csv("cluster_summary.csv", index=False)

# Plot average curves for each cluster
plt.figure(figsize=(12, 8))
for row in output_rows:
    cluster_id, typicality, *avg_curve = row
    label = f"Cluster {cluster_id} (typ={typicality:.2f})"
    plt.plot(avg_curve, label=label, alpha=0.7)
plt.xlabel("Normalized Frequency")
plt.ylabel("FSC Value")
plt.title("Average FSC Curves per Cluster")
plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("cluster_averages.png", dpi=300)
plt.show()

# Plot each cluster in its own subplot with average in red
import math

num_clusters = len(cluster_counts)
cols = 5
rows = math.ceil(num_clusters / cols)
plt.figure(figsize=(cols * 4, rows * 3))

# Sort clusters by size descending
sorted_clusters = cluster_counts.sort_values(ascending=False)

for i, cluster_id in enumerate(sorted_clusters.index):
    cluster_curves = np.stack(cluster_df[cluster_df['cluster'] == cluster_id]['curve'].values)
    avg_curve = cluster_curves.mean(axis=0)

    plt.subplot(rows, cols, i + 1)
    for curve in cluster_curves:
        plt.plot(curve, color='lightgray', alpha=0.3)
    plt.plot(avg_curve, color='blue', linewidth=2)
    plt.title(f"Cluster {cluster_id} (n={len(cluster_curves)})")
    plt.xticks([])
    plt.yticks([])

plt.suptitle("FSC Curves per Cluster (grey) with Average (blue)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("cluster_subplots.png", dpi=300)
plt.close()

# --- PREPARATORY STEP: Calculate Distances and Create a Helper DataFrame ---

# 1. Get the centroid for each point's assigned cluster
centroids = kmeans.cluster_centers_[labels]

# 2. Calculate the Euclidean distance of each embedding to its own centroid
distances = np.sqrt(np.sum((refined_embeddings - centroids) ** 2, axis=1))

# 3. Create a DataFrame to hold cluster info, curve data, and distances
# We use this to easily retrieve the curve and its distance for plotting
distance_df = pd.DataFrame({
    'cluster': labels,
    'curve': list(resampled_data),  # Use the normalized/resampled curve data
    'distance': distances
})

# --- UPDATED PLOTTING LOOP ---

# Ensure cluster_counts and sorted_clusters are defined as before:
cluster_counts = pd.Series(labels).value_counts()
sorted_clusters = cluster_counts.sort_values(ascending=False)

# Define the color map parameters
# We will scale the distance for color intensity or alpha value
# Find the maximum distance across the whole dataset for normalization
max_global_distance = np.max(distance_df['distance'])

num_clusters = len(cluster_counts)
cols = 5
rows = math.ceil(num_clusters / cols)
plt.figure(figsize=(cols * 4, rows * 3))


cmap = plt.colormaps['tab20']

for i, cluster_id in enumerate(sorted_clusters.index):
    # Filter the helper DataFrame for the current cluster
    cluster_data = distance_df[distance_df['cluster'] == cluster_id]

    cluster_curves = np.stack(cluster_data['curve'].values)
    cluster_distances = cluster_data['distance'].values
    avg_curve = cluster_curves.mean(axis=0)

    plt.subplot(rows, cols, i + 1)

    # Plot individual curves, setting color based on distance
    for curve, dist in zip(cluster_curves, cluster_distances):
        # Normalize distance to the [0, 1] range based on global max
        normalized_dist = dist / max_global_distance

        # Option A (Recommended): Use a color map, e.g., 'Reds' or 'Blues'
        # The higher the normalized_dist, the darker the red color
        color_rgb = cmap(normalized_dist)

        plt.plot(curve, color=color_rgb, alpha=0.5, linewidth=0.5)

    # Plot the average curve (Centroid visualization)
    plt.plot(avg_curve, color='black', linewidth=2)
    plt.title(f"Cluster {cluster_id} (n={len(cluster_curves)})")
    plt.xticks([])
    plt.yticks([])

# Add a color bar to explain the distance mapping
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_global_distance))
sm.set_array([])

plt.suptitle("FSC Curves per Cluster (All curves rescaled so that y = 0.143 is the centre x-value) (Average curve shown in black)", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.90, 0.96])

# Add the colorbar to the figure
cbar_ax = plt.gcf().add_axes([0.91, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label('Euclidean Distance from Latent Space Cluster Centroid', rotation=270, labelpad=15)

plt.savefig("cluster_subplots_dist_colored.png", dpi=300)
plt.close()

print('Plotting complete.')