import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    # Write code here
    result = []

    for val in x:
        if val > 0:
            result.append(float(val))
        else:
            result.append(float(alpha * (math.exp(val) - 1)))

    return result        