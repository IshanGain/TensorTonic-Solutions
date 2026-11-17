def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    top_k_set = set(recommended[:k])
    relevant_set = set(relevant)

    hits = len(top_k_set & relevant_set)

    precision = float(hits / k)
    recall = float(hits / len(relevant_set))

    return [precision, recall]