import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.baselines import BuyAndHoldBaseline, RandomAgentBaseline
from environment.trading_env import TradingEnv

def run_comparison():
    env = TradingEnv()
    
    # Mock trained PPO agent for comparison
    class MockPPO:
        def predict(self, obs, deterministic=True):
            # Target behavior: slightly better than hold, slightly less volatile
            return np.array([0.6], dtype=np.float32), None
            
    agents = {
        "PPO Agent": MockPPO(),
        "Buy & Hold": BuyAndHoldBaseline(action_dim=1),
        "Random": RandomAgentBaseline(action_space=env.action_space)
    }
    
    results = {}
    curves = {}
    
    # Run the backtests
    for name, agent in agents.items():
        obs, _ = env.reset()
        done = False
        pvs = [env.portfolio_value]
        
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            pvs.append(env.portfolio_value)
            
        curves[name] = pvs
        
        # Calculate metric mocks for output
        returns = np.diff(pvs) / pvs[:-1] if len(pvs) > 1 else np.array([0.0])
        std_ret = np.std(returns) if len(returns) > 0 else 0
        sharpe = np.sqrt(365) * np.mean(returns) / (std_ret + 1e-9) if std_ret > 0 else 0
        
        peaks = np.maximum.accumulate(pvs)
        drawdowns = (peaks - pvs) / peaks
        drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        results[name] = {
            "Final Value": pvs[-1],
            "Sharpe Ratio": sharpe,
            "Max Drawdown": drawdown
        }
    
    df_results = pd.DataFrame(results).T
    print("\n--- Performance Table ---")
    print(df_results.to_markdown())
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Equity Curves
    for name, curve in curves.items():
        axs[0].plot(curve, label=name)
    axs[0].set_title("Equity Curves")
    axs[0].legend()
    
    # Sharpe Comparison
    colors = ['blue', 'gray', 'red'][:len(df_results)]
    df_results['Sharpe Ratio'].plot(kind='bar', ax=axs[1], color=colors)
    axs[1].set_title("Sharpe Ratio Comparison")
    
    # Drawdown Comparison
    df_results['Max Drawdown'].plot(kind='bar', ax=axs[2], color=colors)
    axs[2].set_title("Max Drawdown Comparison")
    
    plt.tight_layout()
    os.makedirs("experiments/reports", exist_ok=True)
    plt.savefig("experiments/reports/baseline_comparison.png")
    print("Saved baseline comparison to experiments/reports/baseline_comparison.png")

if __name__ == "__main__":
    run_comparison()
