import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)

    # 1. Pairwise similarity logits scaled by temperature
    logits = (Z1 @ Z2.T) / temperature

    # 2. Row-wise max subtraction for numerical stability
    shifted = logits - np.max(logits, axis=1, keepdims=True)

    # 3. Log-sum-exp denominator and positive-pair (diagonal) numerators
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    pos_logits = np.diag(shifted)

    # 4. Mean negative log-likelihood across the batch
    loss = log_sum_exp - pos_logits
    return float(np.mean(loss))    