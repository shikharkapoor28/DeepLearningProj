import numpy as np
import pandas as pd
from typing import Dict, Any

class Evaluator:
    """
    Evaluates trading agents using walk-forward validation and financial metrics.
    """
    def __init__(self, env):
        self.env = env
        self.history = []

    def run_backtest(self, model) -> Dict[str, Any]:
        """
        Runs a deterministic evaluation episode and logs portfolio performance.
        """
        obs, _ = self.env.reset()
        done = False
        
        portfolio_trajectory = [self.env.portfolio_value]
        returns = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            portfolio_trajectory.append(info.get("portfolio_value", 0))
            if len(portfolio_trajectory) > 1:
                ret = (portfolio_trajectory[-1] - portfolio_trajectory[-2]) / portfolio_trajectory[-2]
                returns.append(ret)
                
        metrics = self._calculate_metrics(returns, portfolio_trajectory)
        return metrics

    def walk_forward_validation(self, trainer, train_windows, test_windows):
        """
        Implements expanding or rolling window validation.
        """
        # Pseudo-code for WF logic:
        # For each fold:
        #   1. Set env to train_window dates
        #   2. trainer.train()
        #   3. Set env to test_window dates
        #   4. metrics = self.run_backtest(trainer.model)
        #   5. Store fold test metrics
        pass

    def _calculate_metrics(self, returns: list, portfolio_trajectory: list) -> Dict[str, float]:
        returns_arr = np.array(returns)
        
        if len(returns_arr) == 0:
            return {}
            
        cumulative_return = (portfolio_trajectory[-1] - portfolio_trajectory[0]) / portfolio_trajectory[0]
        
        # Assume 252 trading days for crypto annualized (or 365)
        annualization_factor = 365 
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)
        
        sharpe = (mean_return / std_return) * np.sqrt(annualization_factor) if std_return > 0 else 0
        
        # Max Drawdown
        peaks = np.maximum.accumulate(portfolio_trajectory)
        drawdowns = (peaks - portfolio_trajectory) / peaks
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns_arr[returns_arr < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = (mean_return / downside_std) * np.sqrt(annualization_factor) if downside_std > 0 else 0
        
        return {
            "cumulative_return": cumulative_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": float(max_drawdown)
        }
