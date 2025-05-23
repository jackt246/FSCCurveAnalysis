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
with open("fsc_curves_normalisedandanchored.csv", "r") as f:
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

resampled_data = np.array([resample_curve(c, 100) for c in fsc_data])  # Use first 50 points

# Remove rows with NaNs
resampled_data = resampled_data[~np.isnan(resampled_data).any(axis=1)]