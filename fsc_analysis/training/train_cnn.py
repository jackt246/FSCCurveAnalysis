
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from tensorflow.keras import layers, models


def resample_curve(curve, length=100):
    if len(curve) < 2:
        return np.full(length, np.nan)
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, length)
    f = interp1d(x_old, curve, kind="linear", fill_value="extrapolate")
    return f(x_new)


def create_cnn_encoder(input_shape=(100, 1), latent_dim=32):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(latent_dim, activation="relu"),
    ])
    return model


def main() -> None:
    fsc_data = []
    with open("data/fsc_curves_normalisedandanchored.csv") as f:
        for line in f:
            try:
                values = [float(x) for x in line.strip().split(",") if x]
                if len(values) > 1:
                    fsc_data.append(values)
            except ValueError:
                print(f"⚠️ Skipping bad line: {line.strip()}")

    resampled_data = np.array([resample_curve(c, 100) for c in fsc_data])
    resampled_data = resampled_data[~np.isnan(resampled_data).any(axis=1)]

    resampled_data_expanded = np.expand_dims(resampled_data, axis=-1)

    latent_dim = 32
    encoder = create_cnn_encoder(input_shape=(100, 1), latent_dim=latent_dim)

    embeddings = encoder.predict(resampled_data_expanded, batch_size=32)

    n_clusters = 100
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_assignments = kmeans.fit_predict(embeddings)

    reducer = umap.UMAP()
    umap_embeddings = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 6))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_assignments, cmap="tab10", s=10)
    plt.title("UMAP of CNN Curve Embeddings with Cluster Assignments")
    plt.show()

    clustered_curves = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_assignments):
        clustered_curves[label].append(resampled_data[idx])

    cols = 5
    rows = math.ceil(n_clusters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))

    for i, (cluster_id, curves) in enumerate(clustered_curves.items()):
        ax = axes[i // cols, i % cols]
        curves = np.array(curves)
        for curve in curves:
            ax.plot(curve, color="gray", alpha=0.1)
        avg_curve = np.mean(curves, axis=0)
        ax.plot(avg_curve, color="blue", linewidth=2)
        ax.set_title(f"Cluster {cluster_id} (n={len(curves)})")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])

    plt.tight_layout()
    plt.suptitle("FSC Curve Clusters with Mean Curves", fontsize=16)
    plt.subplots_adjust(top=0.93)
    plt.show()


if __name__ == "__main__":
    main()