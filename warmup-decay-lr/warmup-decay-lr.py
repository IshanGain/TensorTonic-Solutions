def warmup_decay_schedule(base_lr: float, warmup_steps: int, total_steps: int, current_step: int) -> float:
    """
    Returns the learning rate for the requested training step.
    """
    # Write code here

    # warmup phase
    warmup_lr = base_lr * (current_step / warmup_steps)

    # decay phase
    decay_lr = base_lr * ((total_steps - current_step) / (total_steps - warmup_steps))
    
    if (current_step < warmup_steps):
        return warmup_lr
    elif (current_step >= warmup_steps):
        return decay_lr    