import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def resample_curve(curve, length=100):
    """
    Resamples a pre-cleaned numeric array to a fixed length using linear interpolation.
    """
    # Ensure input is a numpy array for math operations
    y = np.asarray(curve, dtype=float)

    # Remove NaNs to prevent the "squashed" 0.8-1.0 effect caused by trailing NaNs
    y_clean = y[~np.isnan(y)]

    # Safety check: need at least 2 points to interpolate
    if len(y_clean) < 2:
        return np.full(length, np.nan) if len(y_clean) == 0 else np.full(length, y_clean[0])

    # x_old defines the current spacing (0 to 1)
    x_old = np.linspace(0, 1, len(y_clean))
    # x_new defines the target spacing (0 to 1 with 100 steps)
    x_new = np.linspace(0, 1, length)

    # Perform linear interpolation
    f = interp1d(x_old, y_clean, kind='linear', bounds_error=False,
                 fill_value=(y_clean[0], y_clean[-1]))

    return f(x_new)

def find_crossing_point(y_values, threshold=0.143):
    # Standard linear interpolation to find the fractional index
    for i in range(1, len(y_values)):
        if y_values[i - 1] >= threshold and y_values[i] < threshold:
            y0, y1 = y_values[i - 1], y_values[i]
            # fractional index = index_low + (how far we are through the gap)
            return (i - 1) + (y0 - threshold) / (y0 - y1)
    return None

def edit_curve_based_on_crossing(y_values, target_idx=50, output_length=100):
    """
    Correctly warps the curve so y_values[crossing_idx] moves to target_idx. This anchors at 0.143
    """
    old_indices = np.arange(len(y_values))

    # 1. Create a map from the old indices to the new coordinate system
    # We map: 0 -> 0, crossing_idx -> target_idx, max_idx -> output_length - 1
    new_x_coords = np.linspace(0, output_length - 1, output_length)

    # Define the mapping: Original index -> Aligned index
    # To interpolate the values, we actually need the inverse:
    # Where does each "New Index" land in the "Old Index" space?
    crossing = find_crossing_point(y_values)
    if crossing is not None:
        old_x_mapped_to_new = [0, crossing, len(y_values) - 1]
        new_x_mapped_to_new = [0, target_idx, output_length - 1]
    else:
        # If no crossing found, do a simple resample without warping
        return resample_curve(y_values, output_length)

    # This function tells us: for a given index in our 100-pt output,
    # which index should we look up in the original data?
    mapping_func = interp1d(new_x_mapped_to_new, old_x_mapped_to_new, kind='linear')

    # Find the corresponding old indices for every new index
    source_indices = mapping_func(new_x_coords)

    # 2. Interpolate the actual Y values at those calculated source indices
    final_interp = interp1d(old_indices, y_values, kind='linear', fill_value="extrapolate")

    return final_interp(source_indices)

df = pd.read_json('data/fsc_curves_all.json')

df["fsc_unmasked"] = df["fsc_unmasked"].apply(np.asarray)
df["fsc_masked"] = df["fsc_masked"].apply(np.asarray)
df["fsc_corrected"] = df["fsc_corrected"].apply(np.asarray)
df["fsc_phaserandom"] = df["fsc_phaserandom"].apply(np.asarray)
print(df.head())

resampled_curves = [edit_curve_based_on_crossing(c) for c in df["fsc_phaserandom"]]
data_min = np.min(resampled_curves)
data_max = np.max(resampled_curves)
resampled_curves = (resampled_curves - data_min) / (data_max - data_min)

plt.figure()
for fsc in resampled_curves:
    plt.plot(fsc, alpha=0.3)

plt.xlabel("Resolution shell")
plt.ylabel("FSC")
plt.show()