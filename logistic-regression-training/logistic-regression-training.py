import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        linear_model = np.dot(X, w) + b

        y_cap = _sigmoid(linear_model)
        error = y_cap - y

        dw = (1/n_samples) * np.dot(X.T, error)
        db = (1/n_samples) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    return w, b    