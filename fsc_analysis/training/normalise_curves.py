import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def find_crossing_point(y_values, threshold=0.143):
    for i in range(1, len(y_values)):
        if y_values[i - 1] >= threshold and y_values[i] < threshold:
            y0, y1 = y_values[i - 1], y_values[i]
            return (i - 1) + (y0 - threshold) / (y0 - y1)
    return None


def align_curve_properly(y_values, crossing_idx, target_idx=50, output_length=100):
    old_indices = np.arange(len(y_values))
    new_x_coords = np.linspace(0, output_length - 1, output_length)
    old_x_mapped_to_new = [0, crossing_idx, len(y_values) - 1]
    new_x_mapped_to_new = [0, target_idx, output_length - 1]
    mapping_func = interp1d(new_x_mapped_to_new, old_x_mapped_to_new, kind='linear')
    source_indices = mapping_func(new_x_coords)
    final_interp = interp1d(old_indices, y_values, kind='linear', fill_value='extrapolate')
    return final_interp(source_indices)


def process_fsc_dataframe(fsc_df, output_length=100, target_idx=50):
    curve_columns = ['fsc_corrected', 'fsc_masked', 'fsc_unmasked']
    df = fsc_df.copy()

    for col in curve_columns:
        if col not in df.columns:
            continue

        aligned_list = []
        for curve in df[col]:
            y = np.array(curve)
            crossing = find_crossing_point(y)
            if crossing is not None:
                aligned = align_curve_properly(y, crossing, target_idx, output_length)
                aligned_list.append(aligned.tolist())
            else:
                f = interp1d(np.linspace(0, 1, len(y)), y)
                aligned_list.append(f(np.linspace(0, 1, output_length)).tolist())

        df[f'{col}_aligned'] = aligned_list

    return df


def resample_curve(y_values, output_length):
    x_original = np.linspace(0, 1, len(y_values))
    f = interp1d(x_original, y_values, bounds_error=False, fill_value='extrapolate')
    return f(np.linspace(0, 1, output_length))


def main() -> None:
    fsc_df = pd.read_json('data/fsc_curves_all.json')
    fsc_df['fsc_unmasked'] = fsc_df['fsc_unmasked'].apply(np.asarray)
    fsc_df['fsc_masked'] = fsc_df['fsc_masked'].apply(np.asarray)
    fsc_df['fsc_corrected'] = fsc_df['fsc_corrected'].apply(np.asarray)
    fsc_df['fsc_phaserandom'] = fsc_df['fsc_phaserandom'].apply(np.asarray)
    processed_df = process_fsc_dataframe(fsc_df)
    processed_df.to_json('data/fsc_curves_all_aligned.json', orient='records')


if __name__ == '__main__':
    main()
