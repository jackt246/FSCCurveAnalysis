import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import requests
from scipy.interpolate import interp1d
import sys

def resample_curve(curve, target_length=100):
    """
    Resample an FSC curve to a fixed length using linear interpolation.
    """
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_old, curve, kind='linear', fill_value='extrapolate')
    return interpolator(x_new)

# Load an FSC curve
emd_id = sys.argv[1]
url = f'https://ebi.ac.uk/emdb/api/analysis/{emd_id}'

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Will raise an error if status code is not 200
except:
    print(f"Could not connect to {emd_id}")

try:
    json_data = response.json()

except ValueError as e:
    print(f"Failed to parse JSON for entry {emd_id}: {e}")

try:
    data = list(json_data.values())[0]
    fsc_curve = data['fsc']['curves']['fsc']
except KeyError:
    print(f"No FSC data for {emd_id}, skipping.")


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

# Example: assume new_curve is a 1D NumPy array of length N
resampled_curve = resample_curve(np.array(fsc_curve), target_length=100)
cluster_id, freq, perc = classify_fsc_curve(resampled_curve)

print(f"Cluster ID: {cluster_id}")
print(f"Frequency: {freq} curves")
print(f"Typicality Percentile: {perc*100:.2f}%")

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

# Call it
draw_typicality_bar(perc)