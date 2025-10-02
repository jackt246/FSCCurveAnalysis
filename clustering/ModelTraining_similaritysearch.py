import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
from io import StringIO
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
import umap
import pandas as pd
from scipy.interpolate import interp1d
import argparse

parser = argparse.ArgumentParser(description="Train and cluster FSC curves.")
args = parser.parse_args()

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
    f = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')
    return f(x_new)

resampled_data = np.array([resample_curve(c, 100) for c in fsc_data])

# Remove rows with NaNs
resampled_data = resampled_data[~np.isnan(resampled_data).any(axis=1)]

# =======================
# Autoencoder & Similarity Search for Typicality
# =======================
# Define encoder and decoder
input_dim = resampled_data.shape[1]

encoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='linear')
])

decoder = models.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(resampled_data, resampled_data, epochs=30, batch_size=128, verbose=1)

# Get latent embeddings
embeddings = encoder.predict(resampled_data)

# Fit nearest neighbors model
from sklearn.neighbors import NearestNeighbors
nn_model = NearestNeighbors(n_neighbors=10).fit(embeddings)

# Compute typicality score for each curve
distances, _ = nn_model.kneighbors(embeddings)
typicality_scores = distances.mean(axis=1)

# Normalize typicality scores to 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
typicality_normalized = scaler.fit_transform(typicality_scores.reshape(-1, 1)).flatten()

# Save to CSV
df_typicality = pd.DataFrame({
    "index": np.arange(len(typicality_normalized)),
    "typicality_score": typicality_normalized
})
df_typicality.to_csv("typicality_scores.csv", index=False)

# Optional: plot distribution
import seaborn as sns
plt.figure(figsize=(8, 4))

sns.violinplot(data=typicality_normalized, orient='h')
plt.title("Distribution of Typicality Scores")
plt.xlabel("Typicality Score (0 = typical, 1 = atypical)")
plt.tight_layout()
plt.savefig("typicality_distribution.png", dpi=300)
plt.show()

# Plot sample of original FSC curves color-coded by typicality
import matplotlib.cm as cm

# Define norm for color normalization
norm = plt.Normalize(vmin=typicality_normalized.min(), vmax=typicality_normalized.max())
cmap = plt.colormaps.get_cmap('viridis_r')

plt.figure(figsize=(12, 6))
for i, curve in enumerate(resampled_data[:500]):  # plot up to 500 for performance
    plt.plot(curve, color=cmap(norm(typicality_normalized[i])), alpha=0.6)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
#plt.colorbar(sm, label="Typicality Score (0=typical, 1=atypical)")
plt.title("Original FSC Curves Colored by Typicality")
plt.xlabel("Normalized Frequency")
plt.ylabel("FSC Value")
plt.tight_layout()
plt.savefig("fsc_curve_colormap.png", dpi=300)
plt.show()

# ====================================
# Optional: Evaluate new curves later
# ====================================
def evaluate_new_curves(new_curves):
    from sklearn.neighbors import NearestNeighbors
    from scipy.interpolate import interp1d

    # Load trained encoder and reference
    encoder = tf.keras.models.load_model("encoder_model.keras")
    reference_embeddings = np.load("reference_embeddings.npy")
    reference_curves = np.load("reference_curves.npy")

    # Fit NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=10).fit(reference_embeddings)

    # Resample function
    def resample_curve(curve, length=100):
        x_old = np.linspace(0, 1, len(curve))
        x_new = np.linspace(0, 1, length)
        f = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')
        return f(x_new)

    resampled_new = np.array([resample_curve(c) for c in new_curves])
    embeddings_new = encoder.predict(resampled_new)
    distances, _ = nn.kneighbors(embeddings_new)
    typicality_scores = distances.mean(axis=1)

    return resampled_new, typicality_scores

# Example usage:
# Suppose new FSC curves are in 'new_fsc_curves.csv'
try:
    new_data = []
    with open("new_fsc_curves.csv", "r") as f:
        for line in f:
            try:
                vals = [float(x) for x in line.strip().split(",") if x]
                if len(vals) > 1:
                    new_data.append(vals)
            except ValueError:
                print(f"⚠️ Skipping bad line: {line.strip()}")

    if new_data:
        resampled_new, new_scores = evaluate_new_curves(new_data)

        # Plot new curves with color by typicality
        import matplotlib.cm as cm
        norm = plt.Normalize(vmin=min(new_scores), vmax=max(new_scores))
        cmap = cm.get_cmap('viridis_r')

        plt.figure(figsize=(10, 6))
        for i, score in enumerate(new_scores):
            color = cmap(norm(score))
            plt.plot(resampled_new[i], color=color, alpha=0.7)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Typicality Score")
        plt.title("New FSC Curves (Color = Typicality)")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("FSC")
        plt.tight_layout()
        plt.savefig("new_fsc_typicality.png", dpi=300)
        plt.show()
except Exception as e:
    print("⚠️ Failed to evaluate new FSC curves.")
    traceback.print_exc()
