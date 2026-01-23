import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def find_crossing_point(y_values, threshold=0.143):
    # Standard linear interpolation to find the fractional index
    for i in range(1, len(y_values)):
        if y_values[i - 1] >= threshold and y_values[i] < threshold:
            y0, y1 = y_values[i - 1], y_values[i]
            # fractional index = index_low + (how far we are through the gap)
            return (i - 1) + (y0 - threshold) / (y0 - y1)
    return None


def align_curve_properly(y_values, crossing_idx, target_idx=50, output_length=100):
    """
    Correctly warps the curve so y_values[crossing_idx] moves to target_idx.
    """
    old_indices = np.arange(len(y_values))

    # 1. Create a map from the old indices to the new coordinate system
    # We map: 0 -> 0, crossing_idx -> target_idx, max_idx -> output_length - 1
    new_x_coords = np.linspace(0, output_length - 1, output_length)

    # Define the mapping: Original index -> Aligned index
    # To interpolate the values, we actually need the inverse:
    # Where does each "New Index" land in the "Old Index" space?
    old_x_mapped_to_new = [0, crossing_idx, len(y_values) - 1]
    new_x_mapped_to_new = [0, target_idx, output_length - 1]

    # This function tells us: for a given index in our 100-pt output,
    # which index should we look up in the original data?
    mapping_func = interp1d(new_x_mapped_to_new, old_x_mapped_to_new, kind='linear')

    # Find the corresponding old indices for every new index
    source_indices = mapping_func(new_x_coords)

    # 2. Interpolate the actual Y values at those calculated source indices
    final_interp = interp1d(old_indices, y_values, kind='linear', fill_value="extrapolate")

    return final_interp(source_indices)


def process_fsc_dataframe(fsc_df, output_length=100, target_idx=50):
    curve_columns = ['fsc_corrected', 'fsc_masked', 'fsc_unmasked']

    # Work on a copy to avoid SettingWithCopy warnings
    df = fsc_df.copy()

    for col in curve_columns:
        if col not in df.columns: continue

        aligned_list = []
        for curve in df[col]:
            y = np.array(curve)
            crossing = find_crossing_point(y)

            if crossing is not None:
                aligned = align_curve_properly(y, crossing, target_idx, output_length)
                aligned_list.append(aligned.tolist())
            else:
                # If no crossing, just resample normally
                f = interp1d(np.linspace(0, 1, len(y)), y)
                aligned_list.append(f(np.linspace(0, 1, output_length)).tolist())

        df[f"{col}_aligned"] = aligned_list

    return df

def resample_curve(y_values, output_length):
    x_original = np.linspace(0, 1, len(y_values))
    f = interp1d(x_original, y_values, bounds_error=False, fill_value='extrapolate')
    return f(np.linspace(0, 1, output_length))

fsc_df = pd.read_json('fsc_curves/fsc_curves_all.json')
fsc_df["fsc_unmasked"] = fsc_df["fsc_unmasked"].apply(np.asarray)
fsc_df["fsc_masked"] = fsc_df["fsc_masked"].apply(np.asarray)
fsc_df["fsc_corrected"] = fsc_df["fsc_corrected"].apply(np.asarray)
fsc_df["fsc_phaserandom"] = fsc_df["fsc_phaserandom"].apply(np.asarray)
processed_df = process_fsc_dataframe(fsc_df)
processed_df.to_json('fsc_curves/fsc_curves_all_aligned.json', orient='records')