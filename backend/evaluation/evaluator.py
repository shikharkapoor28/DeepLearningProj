"""
Evaluation module for trading agents.

Runs deterministic backtests and computes standard financial performance
metrics using 252 trading days for annualization (US equities convention).
"""

import numpy as np
from typing import Dict, Any


class Evaluator:
    """
    Evaluates trading agents by running deterministic episodes and computing
    financial metrics: cumulative return, Sharpe ratio, Sortino ratio, and
    maximum drawdown.
    """

    # US equities trade ~252 days/year; used for annualizing ratios.
    ANNUALIZATION_FACTOR: int = 252

    def __init__(self, env):
        self.env = env

    def run_backtest(self, model) -> Dict[str, Any]:
        """
        Runs a single deterministic evaluation episode.

        Args:
            model: Any object with a .predict(obs, deterministic=True) method
                   (SB3 model, BuyAndHoldBaseline, RandomAgentBaseline, etc.).

        Returns:
            Dictionary of financial performance metrics.
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
                prev = portfolio_trajectory[-2]
                curr = portfolio_trajectory[-1]
                ret = (curr - prev) / (prev + 1e-12)
                returns.append(ret)

        metrics = self._calculate_metrics(returns, portfolio_trajectory)
        return metrics

    def _calculate_metrics(
        self, returns: list, portfolio_trajectory: list
    ) -> Dict[str, float]:
        """
        Computes financial performance metrics.

        Formulas:
            Cumulative Return = (final_value - initial_value) / initial_value

            Sharpe Ratio = (mean_daily_return / std_daily_return) * sqrt(252)
                Measures risk-adjusted return. Higher is better.

            Sortino Ratio = (mean_daily_return / downside_std) * sqrt(252)
                Like Sharpe but penalizes only downside volatility.

            Max Drawdown = max over t of (peak_t - value_t) / peak_t
                Worst peak-to-trough decline. Lower is better.
        """
        returns_arr = np.array(returns)

        if len(returns_arr) == 0:
            return {
                "cumulative_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
            }

        # Cumulative return
        cumulative_return = (
            (portfolio_trajectory[-1] - portfolio_trajectory[0])
            / (portfolio_trajectory[0] + 1e-12)
        )

        # Annualized Sharpe ratio (252 trading days for US equities)
        mean_return = float(np.mean(returns_arr))
        std_return = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0.0
        sharpe = (
            (mean_return / std_return) * np.sqrt(self.ANNUALIZATION_FACTOR)
            if std_return > 0
            else 0.0
        )

        # Max Drawdown
        traj = np.array(portfolio_trajectory)
        peaks = np.maximum.accumulate(traj)
        drawdowns = (peaks - traj) / (peaks + 1e-12)
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Annualized Sortino ratio (downside deviation only)
        downside_returns = returns_arr[returns_arr < 0]
        downside_std = (
            float(np.std(downside_returns, ddof=1))
            if len(downside_returns) > 1
            else 0.0
        )
        sortino = (
            (mean_return / downside_std) * np.sqrt(self.ANNUALIZATION_FACTOR)
            if downside_std > 0
            else 0.0
        )

        return {
            "cumulative_return": float(cumulative_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
        }
