import numpy as np

class DebugTracker:
    """
    Helper class to track model internal states like action distribution,
    reward sources, and gradient norms during training to debug collapses.
    """
    def __init__(self):
        self.reward_logs = []
        self.action_history = []
        self.grad_norms = []

    def log_reward(self, step: int, total_reward: float, shaped_components: dict):
        """
        Logs decomposed reward (e.g., PnL vs Strategy Penalty).
        """
        log = {"step": step, "total": total_reward, **shaped_components}
        self.reward_logs.append(log)

    def log_action_distribution(self, actions: np.ndarray):
        """
        Tracks mean action and variance to check for policy collapse (e.g., all 0s).
        """
        self.action_history.append({
            "mean": np.mean(actions),
            "std": np.std(actions),
            "zeros": np.sum(actions == 0)
        })

    def log_gradient_norm(self, model):
        """
        Tracks the gradient norm of the Neural Network to detect vanishing/exploding gradients.
        Needs to be called during backward pass.
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        return total_norm
