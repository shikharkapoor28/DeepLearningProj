import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.trading_env import TradingEnv

def simulate_strategy(env, name, action_fn, steps=100):
    obs, _ = env.reset()
    rewards = []
    portfolio_values = []
    
    for _ in range(steps):
        action = action_fn()
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        portfolio_values.append(env.portfolio_value)
        if term or trunc:
            break
            
    return rewards, portfolio_values

if __name__ == "__main__":
    env = TradingEnv()
    
    strategies = {
        "Always Hold (Cash)": lambda: np.array([0.0], dtype=np.float32),
        "Always Buy (100%)": lambda: np.array([1.0], dtype=np.float32),
        "Random Actions": lambda: env.action_space.sample()
    }
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    for name, action_fn in strategies.items():
        print(f"Running sanity check for: {name}")
        rewards, pvs = simulate_strategy(env, name, action_fn)
        
        axs[0].plot(rewards, label=name)
        axs[1].plot(pvs, label=name)
        
    axs[0].set_title("Reward Over Time")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    
    axs[1].set_title("Portfolio Value Over Time")
    axs[1].set_ylabel("Value ($)")
    axs[1].set_xlabel("Steps")
    axs[1].legend()
    
    os.makedirs("experiments/reports", exist_ok=True)
    plt.savefig("experiments/reports/reward_sanity.png")
    print("Saved reward sanity plot to experiments/reports/reward_sanity.png")
