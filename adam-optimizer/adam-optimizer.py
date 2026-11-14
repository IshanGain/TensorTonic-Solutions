import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    theta = np.array(param, dtype=float)
    g = np.array(grad, dtype=float)
    m_prev = np.array(m, dtype=float)
    v_prev = np.array(v, dtype=float)

    # Step 1: Update biased first moment estimate
    m_new = beta1 * m_prev + (1 - beta1) * g

    # Step 2: Update biased second raw moment estimate
    v_new = beta2 * v_prev + (1 - beta2) * (g ** 2)

    # Step 3: Compute bias-corrected first and second moment estimates
    m_hat = m_new / (1 - (beta1 ** t))
    v_hat = v_new / (1 - (beta2 ** t))

    # Step 4: Update parameters
    param_new = theta - lr * (m_hat / (np.sqrt(v_hat) + eps))

    return param_new, m_new, v_new