import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class DatasetBuilder:
    """
    Aligns multiple assets and builds the state vector required for the RL Agent.
    """
    def __init__(self, fetcher, engineer):
        self.fetcher = fetcher
        self.engineer = engineer

    def build_dataset(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fetches, engineers, aligns, and outputs a ready-to-train sequence.
        """
        raw_df = self.fetcher.fetch_historical(start_date, end_date)
        
        # Group by ticker and engineer features
        processed_dfs = []
        for ticker in raw_df['Ticker'].unique():
            ticker_data = raw_df[raw_df['Ticker'] == ticker].copy()
            feat_data = self.engineer.generate_features(ticker_data)
            feat_data['Ticker'] = ticker
            processed_dfs.append(feat_data)
            
        if not processed_dfs:
             return pd.DataFrame(), np.array([])

        final_df = pd.concat(processed_dfs)
        
        # Pivot to align timestamps across assets
        final_df = final_df.reset_index()
        # Ensure Datetime column exists cleanly
        dt_col = 'Date' if 'Date' in final_df.columns else 'Datetime'
        
        # Pivot normalized features
        norm_cols = [c for c in final_df.columns if c.endswith('_norm')]
        
        # Create state vector array mapping [time, asset, features]
        # For a single asset 'SPY' (default assumption for base RL scaffold):
        spy_data = final_df[final_df['Ticker'] == 'SPY'].sort_values(dt_col)
        state_vectors = spy_data[norm_cols].values
        
        return spy_data, state_vectors

    def build_trading_env(
        self,
        start_date: str,
        end_date: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        End-to-end: fetch Yahoo history, engineer features, return a TradingEnv on real bars.
        Requires `seq_len` in config to match or be less than available rows (needs seq_len+1 rows).
        """
        from environment.trading_env import TradingEnv

        spy_df, _ = self.build_dataset(start_date, end_date)
        if spy_df.empty:
            raise ValueError("No data returned for the given date range / ticker.")

        norm_cols = [c for c in spy_df.columns if c.endswith("_norm")]
        if not norm_cols:
            raise ValueError("No normalized feature columns; run feature engineering first.")

        feature_matrix = spy_df[norm_cols].to_numpy(dtype=np.float32)
        close_prices = spy_df["Close"].to_numpy(dtype=np.float64)

        return TradingEnv(
            config=config,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            feature_columns=norm_cols,
        )
