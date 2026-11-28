from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x_arr = np.asarray(x, dtype=float)
    
    mode_counter = Counter(x_arr)
    max_freq = max(mode_counter.values())
    mode_value = min(k for k, v in mode_counter.items() if v == max_freq)

    return {
        "mean": float(np.mean(x_arr)),
        "median": float(np.median(x_arr)),
        "mode": float(mode_value),
    }