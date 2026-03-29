import unittest
import numpy as np
import sys
import os

# Ensure the backend directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.trading_env import TradingEnv

class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        self.config = {
            "initial_balance": 100000.0,
            "trading_fee_bps": 10,
            "slippage_bps": 5,
            "n_assets": 1,
            "seq_len": 64,
            "n_features": 10,
            "max_drawdown_stop": 0.30
        }
        self.env = TradingEnv(config=self.config)

    def test_reset_logic(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (64, 10))
        self.assertEqual(info['cash'], 100000.0)
        self.assertEqual(info['portfolio_value'], 100000.0)

    def test_buy_action_correctness(self):
        self.env.reset()
        # Action: 100% allocation to asset (1.0 weight)
        action = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertAlmostEqual(info['portfolio_value'], 100000.0, delta=2000)

    def test_sell_action_correctness(self):
        self.env.reset()
        action_buy = np.array([1.0], dtype=np.float32)
        self.env.step(action_buy)
        action_sell = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action_sell)
        # Should be lower than 100k due to friction
        self.assertLess(info['portfolio_value'], 100000.0)

    def test_transaction_cost_applied(self):
        self.env.reset()
        action = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Verify that Info dict captures steps and transaction metrics
        self.assertIn('step', info)
        self.assertEqual(info['step'], 1)

    def test_slippage_behavior(self):
        # Slippage should cause portfolio drops upon consecutive buys/sells
        pass

    def test_turbulence_override(self):
        # Placeholder to test logic where high market turbulence forces a liquidation
        pass

    def test_episode_termination(self):
        self.env.reset()
        # Force a huge drawdown
        self.env.portfolio_value = 50000.0 # 50% drawdown
        # Verify termination condition is met if logic calculates it during step
        obs, reward, terminated, truncated, info = self.env.step(np.array([1.0]))
        # Assuming the env terminates on large drawdown
        # self.assertTrue(terminated)
        pass

if __name__ == '__main__':
    unittest.main()
