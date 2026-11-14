def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    """
    Returns a dictionary of metrics.
    """
    # Write code here
    n = len(y_true)

    if system_type == "classification":
        tp = tn = fp = fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 0:
                tn += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 1 and yp == 0:
                fn += 1

        accuracy = (tp + tn) / n if n > 0 else 0.0

        precision_denom = tp + fp
        precision = tp / precision_denom if precision_denom > 0 else 0.0

        recall_denom = tp + fn
        recall = tp / recall_denom if recall_denom > 0 else 0.0

        f1_denom = precision + recall
        f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    elif system_type == "regression":
        mae_sum = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred))
        mae = mae_sum / n if n > 0 else 0.0

        mse_sum = sum(abs(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        rmse = math.sqrt(mse_sum / n) if n > 0 else 0.0

        return {
            "mae": mae,
            "rmse": rmse
        }


    elif system_type == "ranking":
        sorted_indices = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)
        top_3_indices = sorted_indices[:3]

        rel_in_top_3 = sum(1 for i in top_3_indices if y_true[i] == 1)
        precision_at_3 = rel_in_top_3 / 3.0

        total_relevent = sum(1 for yt in y_true if yt == 1)
        recall_at_3 = rel_in_top_3 / total_relevent if total_relevent > 0 else 0.0

        return {
            "precision_at_3": precision_at_3,
            "recall_at_3": recall_at_3
        }

    return {}    