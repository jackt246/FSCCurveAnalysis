import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import tensorflow as tf
import umap
from scipy.interpolate import interp1d
from tensorflow.keras import layers, models


def resample_curve(curve, length=100):
    if len(curve) < 2:
        return np.full(length, np.nan)
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, length)
    f = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')
    return f(x_new)


def main() -> None:
    fsc_data = []
    with open('data/fsc_curves_normalisedandanchored.csv', 'r') as f:
        for line in f:
            try:
                values = [float(x) for x in line.strip().split(',') if x]
                if len(values) > 1:
                    fsc_data.append(values)
            except ValueError:
                print(f"⚠️ Skipping bad line: {line.strip()}")

    resampled_data = np.array([resample_curve(c, 100) for c in fsc_data])
    resampled_data = resampled_data[~np.isnan(resampled_data).any(axis=1)]

    input_dim = resampled_data.shape[1]

    encoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='linear'),
    ])

    decoder = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid'),
    ])

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    print('Training autoencoder')
    autoencoder.fit(resampled_data, resampled_data, epochs=30, batch_size=128, verbose=1)

    embeddings = encoder.predict(resampled_data).T

    n_clusters = 150
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        embeddings, c=n_clusters, m=2.0, error=1e-5, maxiter=1000, init=None
    )
    u = u.T
    labels = np.argmax(u, axis=1)

    np.save('fuzzy_memberships.npy', u)
    np.save('fuzzy_centers.npy', cntr)

    print('Projecting with UMAP')
    umap_proj = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3).fit_transform(embeddings.T)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(umap_proj[:, 0], umap_proj[:, 1], umap_proj[:, 2], c=labels, cmap='tab10', s=10)
    ax.set_title('Clustering of FSC Curves (UMAP 3D)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    fig.colorbar(sc, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.show()

    cluster_df = pd.DataFrame({
        'cluster': labels,
        'embedding': list(embeddings.T),
        'curve': list(resampled_data),
    })

    encoder.save('encoder_model.h5')

    cluster_counts = cluster_df['cluster'].value_counts().sort_index()
    cluster_counts.to_csv('cluster_frequencies.csv', header=['count'])
    print('Models and cluster info saved.')

    cluster_sizes = cluster_df['cluster'].value_counts().sort_values(ascending=False)
    sorted_cluster_ids = cluster_sizes.index.tolist()

    num_clusters = len(sorted_cluster_ids)
    cols = 5
    rows = math.ceil(num_clusters / cols)
    plt.figure(figsize=(cols * 4, rows * 3))

    for i, cluster_id in enumerate(sorted_cluster_ids):
        cluster_curves = np.stack(cluster_df[cluster_df['cluster'] == cluster_id]['curve'].values)
        avg_curve = cluster_curves.mean(axis=0)

        plt.subplot(rows, cols, i + 1)
        for curve in cluster_curves:
            plt.plot(curve, color='lightgray', alpha=0.4)
        plt.plot(avg_curve, color='blue', label='Average')
        plt.title(f'Cluster {cluster_id} (n={len(cluster_curves)})')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    plt.suptitle('FSC Curves by Cluster (ordered by size)', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()


if __name__ == '__main__':
    main()
