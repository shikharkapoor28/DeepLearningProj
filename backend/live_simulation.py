import asyncio
from data_pipeline.fetcher import YahooFinanceFetcher
from data_pipeline.features import FeatureEngineer
from rl_core.trainer import RLTrainer
from environment.trading_env import TradingEnv
import time
import numpy as np

async def main():
    print("Initializing Live Trading Simulation...")
    
    # 1. Initialize Pipeline
    fetcher = YahooFinanceFetcher(tickers=["SPY", "AAPL", "MSFT"]) # Added requested tickers
    engineer = FeatureEngineer()
    env = TradingEnv()
    
    # 2. Warm up state
    print("Warming up indicators with historical data...")
    hist_df = fetcher.fetch_historical(start_date="2024-01-01", end_date="2024-02-01", interval="1h")
    # Simulate engineered state build
    feat_df = engineer.generate_features(hist_df[hist_df['Ticker'] == 'SPY'])
    
    # 3. Simulate Live Tick Loop
    print("\nStarting live agent stream (Connecting to Real Time Yahoo Finance API):")
    for _ in range(5):
        # Fetching Latest
        latest_df = fetcher.fetch_latest("1m")
        if latest_df.empty:
            continue
            
        spy_row = latest_df[latest_df['Ticker'] == 'SPY'].iloc[-1]
        current_price = spy_row['Close']
        print(f"[{time.strftime('%X')}] Tick: SPY @ ${current_price:.2f}")
        
        # Agent Action (Mocked evaluation pass)
        action = np.array([1.0], dtype=np.float32) 
        
        # Step Env
        obs, reward, term, trunc, info = env.step(action)
        print(f"  -> Action: BUY | Portfolio Value: ${env.portfolio_value:.2f}")
        
        await asyncio.sleep(2) # Wait for next simulated tick

if __name__ == "__main__":
    asyncio.run(main())
