import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    # Write code here
    y_arr = np.array(y)

    if num_classes is None:
        num_classes = np.max(y_arr) + 1

    encoded = np.zeros((y_arr.size, num_classes), dtype=float)
    encoded[np.arange(y_arr.size), y_arr] = 1.0

    return encoded