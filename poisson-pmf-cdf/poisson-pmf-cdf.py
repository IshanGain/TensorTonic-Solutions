import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here

    # Base case: probability of 0 events, P(X = 0) = e^(-lam)
    pmf = math.exp(-lam)
    cdf = pmf

    # Use recurrence relation: P(X = i) = P(X = i - 1) * (lam / i)
    for i in range(1, k + 1):
        pmf *= lam / i
        cdf += pmf

    return {"pmf": float(pmf), "cdf": float(cdf)}    