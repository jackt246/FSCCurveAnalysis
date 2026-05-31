import traceback

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models


def resample_curve(curve, length=100):
    if len(curve) < 2:
        return np.full(length, np.nan)
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, length)
    f = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')
    return f(x_new)


def evaluate_new_curves(new_curves):
    encoder = tf.keras.models.load_model('encoder_model.keras')
    reference_embeddings = np.load('reference_embeddings.npy')
    reference_curves = np.load('reference_curves.npy')

    nn = NearestNeighbors(n_neighbors=10).fit(reference_embeddings)

    def _resample(curve, length=100):
        x_old = np.linspace(0, 1, len(curve))
        x_new = np.linspace(0, 1, length)
        f = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')
        return f(x_new)

    resampled_new = np.array([_resample(c) for c in new_curves])
    embeddings_new = encoder.predict(resampled_new)
    distances, _ = nn.kneighbors(embeddings_new)
    typicality_scores = distances.mean(axis=1)

    return resampled_new, typicality_scores


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
        layers.Dropout(0.2),
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
    autoencoder.fit(resampled_data, resampled_data, epochs=30, batch_size=128, verbose=1)

    embeddings = encoder.predict(resampled_data)

    nn_model = NearestNeighbors(n_neighbors=10).fit(embeddings)
    distances, _ = nn_model.kneighbors(embeddings)
    typicality_scores = distances.mean(axis=1)

    scaler = MinMaxScaler()
    typicality_normalized = scaler.fit_transform(typicality_scores.reshape(-1, 1)).flatten()

    df_typicality = pd.DataFrame({
        'index': np.arange(len(typicality_normalized)),
        'typicality_score': typicality_normalized,
    })
    df_typicality.to_csv('typicality_scores.csv', index=False)

    plt.figure(figsize=(8, 4))
    sns.violinplot(data=typicality_normalized, orient='h')
    plt.title('Distribution of Typicality Scores')
    plt.xlabel('Typicality Score (0 = typical, 1 = atypical)')
    plt.tight_layout()
    plt.savefig('typicality_distribution.png', dpi=300)
    plt.show()

    norm = plt.Normalize(vmin=typicality_normalized.min(), vmax=typicality_normalized.max())
    cmap = plt.colormaps.get_cmap('viridis_r')

    plt.figure(figsize=(12, 6))
    for i, curve in enumerate(resampled_data[:500]):
        plt.plot(curve, color=cmap(norm(typicality_normalized[i])), alpha=0.6)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.title('Original FSC Curves Colored by Typicality')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('FSC Value')
    plt.tight_layout()
    plt.savefig('fsc_curve_colormap.png', dpi=300)
    plt.show()

    try:
        new_data = []
        with open('new_fsc_curves.csv', 'r') as f:
            for line in f:
                try:
                    vals = [float(x) for x in line.strip().split(',') if x]
                    if len(vals) > 1:
                        new_data.append(vals)
                except ValueError:
                    print(f"⚠️ Skipping bad line: {line.strip()}")

        if new_data:
            resampled_new, new_scores = evaluate_new_curves(new_data)

            norm = plt.Normalize(vmin=min(new_scores), vmax=max(new_scores))
            cmap = cm.get_cmap('viridis_r')

            plt.figure(figsize=(10, 6))
            for i, score in enumerate(new_scores):
                color = cmap(norm(score))
                plt.plot(resampled_new[i], color=color, alpha=0.7)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Typicality Score')
            plt.title('New FSC Curves (Color = Typicality)')
            plt.xlabel('Normalized Frequency')
            plt.ylabel('FSC')
            plt.tight_layout()
            plt.savefig('new_fsc_typicality.png', dpi=300)
            plt.show()
    except Exception:
        print('⚠️ Failed to evaluate new FSC curves.')
        traceback.print_exc()


if __name__ == '__main__':
    main()
