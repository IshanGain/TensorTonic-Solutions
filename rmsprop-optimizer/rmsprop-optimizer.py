import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    g = np.array(g, dtype=float)
    s_prev = np.array(s, dtype=float)
    w_prev = np.array(w, dtype=float)

    # step 1
    new_s = beta * s_prev + (1 - beta) * (g * g)

    # step 2
    new_w = w_prev - lr / np.sqrt(new_s + eps) * g

    return new_w, new_s