import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import requests
from scipy.interpolate import interp1d
import sys
import os
import matplotlib.pyplot as plt

def find_crossing_point(y_values, threshold=0.143):
    for i in range(11, len(y_values)):
        if y_values[i-1] >= threshold and y_values[i] < threshold:
            x0, x1 = i-1, i
            y0, y1 = y_values[i-1], y_values[i]
            return x0 + (threshold - y0) / (y1 - y0)
    return None

def resample_curve(curve, target_length=100):
    """
    Resample an FSC curve to a fixed length using linear interpolation.
    """
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')

    return interpolator(x_new)

def align_curve(y_values, crossing_index, target_index=50, output_length=100):
    """
    Piecewise warp curve so that the segment before the crossing_index stretches to [0, target_index]
    and the segment after compresses to [target_index, output_length-1].
    """
    original_length = len(y_values)

    # Create full x and y arrays
    x = np.linspace(0, 1, original_length)
    y = y_values

    # Identify the actual index (rounded) and corresponding x value
    crossing_index_int = int(np.floor(crossing_index))
    x_cross = x[crossing_index_int]

    # Create new x grid
    x_new = np.linspace(0, 1, output_length)

    # Split curve at crossing
    x_left = x[:crossing_index_int + 1]
    y_left = y[:crossing_index_int + 1]
    x_right = x[crossing_index_int:]
    y_right = y[crossing_index_int:]

    # New x segments
    x_target_left = np.linspace(0, x_new[target_index], len(x_left))
    x_target_right = np.linspace(x_new[target_index], 1, len(x_right))

    # Interpolators
    f_left = interp1d(x_target_left, y_left, kind='linear', bounds_error=False, fill_value='extrapolate')
    f_right = interp1d(x_target_right, y_right, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Final y using piecewise assembly
    y_aligned = np.empty_like(x_new)
    y_aligned[:target_index + 1] = f_left(x_new[:target_index + 1])
    y_aligned[target_index:] = f_right(x_new[target_index:])

    return y_aligned

input_csv = sys.argv[1]
if not os.path.isfile(input_csv):
    print(f"Input file {input_csv} does not exist.")
    sys.exit(1)

with open(input_csv, 'r') as f:
    emd_ids = [line.strip() for line in f if line.strip()]

# List to collect results for plotting
results = []
aligned_curves = []

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

    y = np.array(fsc_curve)
    y_resampled = resample_curve(y, target_length=100)
    crossing = find_crossing_point(y_resampled)
    if crossing is not None:
        y_aligned = align_curve(y_resampled, crossing, target_index=50, output_length=100)
    else:
        print(f"{emd_id}: No 0.143 crossing — using unaligned resampling")
        y_aligned = y_resampled
    cluster_id, freq, perc = classify_fsc_curve(y_aligned)
    print(f"{emd_id}: Cluster ID = {cluster_id}, Frequency = {freq}, Typicality Percentile = {perc*100:.2f}%")
    draw_typicality_bar(perc)
    results.append((emd_id, perc))
    aligned_curves.append(y_aligned)

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
    plt.savefig("fsc_assessment_batch.png", dpi=300)

# Optional: plot all aligned FSC curves color-coded by typicality
import matplotlib.cm as cm

plt.figure(figsize=(12, 6))
norm = plt.Normalize(0, 1)
# Use colormap with higher visual granularity (256 discrete colors)
cmap = cm.get_cmap('gist_rainbow')

for i, (emd_id, perc) in enumerate(results):
    color = cmap(norm(perc))
    plt.plot(np.linspace(0, 1, 100), aligned_curves[i], color=color, alpha=0.6)

from matplotlib.cm import ScalarMappable

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([perc for _, perc in results])
plt.colorbar(sm, ax=plt.gca(), label='Typicality Percentile')

plt.title("FSC Curves Colored by Typicality")
plt.xlabel("Normalized Frequency")
plt.ylabel("FSC")
plt.grid(True)
plt.tight_layout()
plt.savefig("fsc_colored_by_typicality.png", dpi=300)

import seaborn as sns

plt.figure(figsize=(8, 6))
sns.violinplot(data=[perc for _, perc in results], orient='h', inner='quartile', color='skyblue')
plt.xlabel("Typicality Percentile")
plt.title("Distribution of Typicality Scores")
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig("fsc_typicality_violin.png", dpi=300)
plt.show()