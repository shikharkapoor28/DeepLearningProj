import pandas as pd
import pandas_ta as ta
import numpy as np

class FeatureEngineer:
    """
    Applies technical indicators and normalizes features to [-1, 1].
    """
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates RSI, MACD, Returns, Volatility, and Turbulence.
        Assumes df contains ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        df = df.copy()
        
        # Flatten multi-index if from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        # Returns
        df['return_1p'] = df['Close'].pct_change()
        
        # Volatility (Rolling standard deviation of returns)
        df['volatility'] = df['return_1p'].rolling(window=20).std()
        
        # RSI (Relative Strength Index) to measure momentum and overbought/oversold conditions
        df.ta.rsi(length=14, append=True)
        rsi_col = [c for c in df.columns if 'RSI' in c][0]
        
        # MACD (Moving Average Convergence Divergence) for trend following
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        macd_col = [c for c in df.columns if 'MACD_12_26_9' in c][0]
        
        # Turbulence (Mahalanobis distance analog for multivariate, simplified here as vol spike)
        df['turbulence'] = (df['volatility'] - df['volatility'].rolling(100).mean()) / \
                           (df['volatility'].rolling(100).std() + 1e-9)
                           
        # Clean NaNs caused by rolling windows
        df.dropna(inplace=True)
        
        # Normalization to [-1, 1] using tanh clipping to gracefully bound extreme outliers
        df_norm = self._normalize_features(df, [rsi_col, macd_col, 'volatility', 'return_1p', 'turbulence', 'Volume'])
        
        return df_norm
        
    def _normalize_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Normalizes specified columns to [-1, 1] bounds using robust scaling + tanh.
        """
        for col in columns:
            if col in df.columns:
                # Robust standardization
                median = df[col].median()
                mad = (df[col] - median).abs().median()
                if mad == 0: mad = 1e-9
                
                # Zero-centered score mapped to tanh bounds [-1, 1]
                z_score = (df[col] - median) / (mad * 1.4826)
                df[f'{col}_norm'] = np.tanh(z_score)
                
        return df
