import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    
    centered = x - np.mean(x)
    var = np.sum(centered ** 2) / (x.size - 1)
    std_dev = np.sqrt(var)

    return {
        "variance": float(var),
        "standard_deviation": float(std_dev),
    }