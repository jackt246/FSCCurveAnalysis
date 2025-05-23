import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import requests
from scipy.interpolate import interp1d
import sys
import os
import matplotlib.pyplot as plt

def resample_curve(curve, target_length=100):
    """
    Resample an FSC curve to a fixed length using linear interpolation.
    """
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')
    return interpolator(x_new)

input_csv = sys.argv[1]
if not os.path.isfile(input_csv):
    print(f"Input file {input_csv} does not exist.")
    sys.exit(1)

with open(input_csv, 'r') as f:
    emd_ids = [line.strip() for line in f if line.strip()]

# List to collect results for plotting
results = []

def draw_typicality_bar(percentile, width=40):
    """
    Draw a terminal bar showing where the curve lies in the typicality scale.
    """
    pos = int(percentile * width)
    bar = ["─"] * width
    if 0 <= pos < width:
        bar[pos] = "0"
    bar_str = "".join(bar)
    print(f"Least Typical  {bar_str}  Most Typical")


# Load encoder and KMeans clustering model
encoder = load_model("encoder_model.h5")
kmeans = joblib.load("kmeans_model.pkl")

# Load cluster frequencies
cluster_freq = pd.read_csv("cluster_frequencies.csv", index_col=0)["count"]
total_curves = cluster_freq.sum()
cluster_percentiles = cluster_freq.rank(pct=True)  # Used for slider

def classify_fsc_curve(fsc_curve: np.ndarray):
    """
    fsc_curve: A numpy array with the same length & format as used in training
    Returns: cluster ID, frequency, percentile
    """
    # Preprocess if needed: e.g., normalize or reshape
    encoded = encoder.predict(fsc_curve.reshape(1, -1))
    cluster_id = kmeans.predict(encoded)[0]

    frequency = cluster_freq.get(cluster_id, 0)
    percentile = cluster_percentiles.get(cluster_id, 0.0)

    return cluster_id, frequency, percentile

for emd_id in emd_ids:
    url = f'https://ebi.ac.uk/emdb/api/analysis/{emd_id}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        data = list(json_data.values())[0]
        fsc_curve = data['fsc']['curves']['fsc']
    except Exception as e:
        print(f"Skipping {emd_id} due to error: {e}")
        continue

    resampled_curve = resample_curve(np.array(fsc_curve), target_length=100)
    cluster_id, freq, perc = classify_fsc_curve(resampled_curve)
    print(f"{emd_id}: Cluster ID = {cluster_id}, Frequency = {freq}, Typicality Percentile = {perc*100:.2f}%")
    draw_typicality_bar(perc)
    results.append((emd_id, perc))

# After processing all curves, generate the plot
if results:
    # Sort results by percentile
    sorted_results = sorted(results, key=lambda x: x[1])
    emd_ids_sorted, percentiles_sorted = zip(*sorted_results)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(percentiles_sorted)), percentiles_sorted, c='blue')
    for i, emd_id in enumerate(emd_ids_sorted):
        y_val = percentiles_sorted[i]
        y_offset = 0.02 if i % 2 == 0 else -0.02
        va = 'bottom' if i % 2 == 0 else 'top'
        plt.text(i, y_val + y_offset, emd_id, fontsize=6, rotation=90, ha='center', va=va)

    plt.xlabel("FSC Curve Index (sorted by typicality)")
    plt.ylabel("Typicality Percentile")
    plt.title("Typicality of FSC Curves by EMD ID")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("fsc_assessment_batch_anchored.png", dpi=300)