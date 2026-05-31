import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from tensorflow.keras import layers, models


def resample_curve(curve, length=100):
    if len(curve) < 2:
        return np.full(length, np.nan)
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, length)
    f = interp1d(x_old, curve, kind='linear', fill_value='nan', bounds_error=False)
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

    data_min = np.min(resampled_data)
    data_max = np.max(resampled_data)
    if data_max == data_min:
        normalized_data = resampled_data
    else:
        normalized_data = (resampled_data - data_min) / (data_max - data_min)
    resampled_data = normalized_data

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
        layers.Dense(input_dim, activation='linear'),
    ])

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    print('Training autoencoder')
    autoencoder.fit(resampled_data, resampled_data, epochs=30, batch_size=128, verbose=1)

    print('Generating embeddings and clustering with DBSCAN')
    refined_embeddings = encoder.predict(resampled_data)
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    labels = dbscan.fit_predict(refined_embeddings)

    print('Projecting with UMAP')
    umap_proj = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3).fit_transform(refined_embeddings)

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
        'embedding': list(refined_embeddings),
        'curve': list(resampled_data),
    })

    encoder.save('encoder_model.h5')
    joblib.dump(dbscan, 'dbscan_model.pkl')

    cluster_counts = cluster_df['cluster'].value_counts().sort_index()
    cluster_counts.to_csv('cluster_frequencies.csv', header=['count'])
    print('Models and cluster info saved.')

    cluster_df = pd.DataFrame({
        'cluster': labels,
        'curve': list(resampled_data),
    })

    cluster_counts = cluster_df['cluster'].value_counts().sort_index()
    total_curves = len(cluster_df)

    output_rows = []
    for cluster_id in cluster_counts.index:
        cluster_curves = np.stack(cluster_df[cluster_df['cluster'] == cluster_id]['curve'].values)
        avg_curve = cluster_curves.mean(axis=0)
        typicality = len(cluster_curves) / total_curves
        output_rows.append([cluster_id, typicality] + avg_curve.tolist())

    columns = ['cluster_id', 'typicality'] + [f'fsc_{i}' for i in range(resampled_data.shape[1])]
    df_out = pd.DataFrame(output_rows, columns=columns)
    df_out.to_csv('cluster_summary.csv', index=False)

    plt.figure(figsize=(12, 8))
    for row in output_rows:
        cluster_id, typicality, *avg_curve = row
        label = f'Cluster {cluster_id} (typ={typicality:.2f})'
        plt.plot(avg_curve, label=label, alpha=0.7)
    plt.xlabel('Normalized Frequency')
    plt.ylabel('FSC Value')
    plt.title('Average FSC Curves per Cluster')
    plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('cluster_averages.png', dpi=300)
    plt.show()

    num_clusters = len(cluster_counts)
    cols = 5
    rows = math.ceil(num_clusters / cols)
    plt.figure(figsize=(cols * 4, rows * 3))

    sorted_clusters = cluster_counts.sort_values(ascending=False)

    for i, cluster_id in enumerate(sorted_clusters.index):
        cluster_curves = np.stack(cluster_df[cluster_df['cluster'] == cluster_id]['curve'].values)
        avg_curve = cluster_curves.mean(axis=0)

        plt.subplot(rows, cols, i + 1)
        for curve in cluster_curves:
            plt.plot(curve, color='lightgray', alpha=0.3)
        plt.plot(avg_curve, color='blue', linewidth=2)
        plt.title(f'Cluster {cluster_id} (n={len(cluster_curves)})')
        plt.xticks([])
        plt.yticks([])

    plt.suptitle('FSC Curves per Cluster (grey) with Average (blue)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('cluster_subplots.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
