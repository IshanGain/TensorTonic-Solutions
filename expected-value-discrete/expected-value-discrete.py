import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    x_array = np.asarray(x, dtype=float)
    p_array = np.asarray(p, dtype=float)

    expected_value = np.dot(x_array, p_array)

    return float(expected_value)

