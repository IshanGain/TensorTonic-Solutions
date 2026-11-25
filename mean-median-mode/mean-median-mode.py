from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)

    mean_value = np.mean(x_arr)
    median_value = np.median(x_arr)
    
    mode_counter = Counter(x_arr)
    max_freq = max(mode_counter.values())
    min_freq = min(k for k, v in mode_counter.items() if v == max_freq)

    return {
        "mean": float(mean_value),
        "median": float(median_value),
        "mode": float(min_freq),
    }