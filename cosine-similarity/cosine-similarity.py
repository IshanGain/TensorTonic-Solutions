import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    dot_product = np.dot(a_arr, b_arr)

    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)

    if a_norm == 0.0 or b_norm ==0.0:
        return 0.0

    result = dot_product / (a_norm * b_norm)

    return float(result)