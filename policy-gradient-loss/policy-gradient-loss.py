def policy_gradient_loss(log_probs: list, rewards: list, gamma: float) -> float:
    """
    Returns the REINFORCE loss with a mean-return baseline.
    """
    # Write code here
    T = len(rewards)
    returns = [0.0] * T
    
    # 1. Compute discounted returns backward
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
        
    # 2. Subtract the mean return as a baseline
    mean_return = sum(returns) / T
    advantages = [g - mean_return for g in returns]
    
    # 3. Compute the negative mean of (log_prob * advantage)
    loss = -sum(lp * adv for lp, adv in zip(log_probs, advantages)) / T
    
    return float(loss)
    