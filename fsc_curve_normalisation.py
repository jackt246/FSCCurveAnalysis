import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def find_crossing_point(y_values, threshold=0.143):
    for i in range(6, len(y_values)):
        if y_values[i-1] >= threshold and y_values[i] < threshold:
            # Linear interpolation to find exact crossing
            x0, x1 = i-1, i
            y0, y1 = y_values[i-1], y_values[i]
            return x0 + (threshold - y0) / (y1 - y0)
    return None

def align_curve(y_values, crossing_index, target_index=80, output_length=100):
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

def resample_curve(y_values, output_length):
    x_original = np.linspace(0, 1, len(y_values))
    f_interp = interp1d(x_original, y_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    x_resampled = np.linspace(0, 1, output_length)
    return f_interp(x_resampled)

def process_csv(input_csv, output_csv, output_length=100, target_anchor=0.5):
    with open(input_csv, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        curves = [np.array([float(val) for val in line.split(',') if val]) for line in lines]
    aligned_curves = []
    for idx, y in enumerate(curves):
        y = resample_curve(y, output_length)  # Normalize length first
        crossing = find_crossing_point(y)
        if crossing is None:
            print(f"Row {idx}: No 0.143 crossing — using unaligned resampling")
            aligned = y
        else:
            aligned = align_curve(y, crossing, target_index=int(target_anchor * output_length), output_length=output_length)
        aligned_curves.append(aligned)
    pd.DataFrame(aligned_curves).to_csv(output_csv, index=False, header=False)

    # Plot every 100th aligned curve for sanity checking
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i in range(0, len(aligned_curves), 100):
        plt.plot(aligned_curves[i], alpha=0.5)
    plt.title("Sanity check: every 100th aligned FSC curve")
    plt.xlabel("Normalized frequency")
    plt.ylabel("FSC value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python fsc_curve_normalisation.py input.csv output.csv")
        sys.exit(1)
    process_csv(sys.argv[1], sys.argv[2])