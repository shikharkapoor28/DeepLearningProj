import yfinance as yf
import pandas as pd
from typing import List

class YahooFinanceFetcher:
    """
    Handles fetching OHLCV data from Yahoo Finance for a given list of tickers.
    Supports historical loading and incremental updates.
    """
    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    def fetch_historical(self, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """
        Fetches historical data for standard tickers.
        """
        print(f"Fetching historical data for {self.tickers}...")
        try:
            # yfinance returns multi-index columns if multiple tickers are passed
            # We fetch one by one and unify them for robustness
            df_list = []
            for ticker in self.tickers:
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
                data['Ticker'] = ticker
                df_list.append(data)
            
            combined_df = pd.concat(df_list)
            return combined_df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_latest(self, interval: str = "1m") -> pd.DataFrame:
        """
        Fetches the latest available candle (useful for live streams).
        """
        try:
            df_list = []
            for ticker in self.tickers:
                # '1d' period, '1m' interval gets intra-day latest data
                data = yf.download(ticker, period="1d", interval=interval, progress=False)
                data['Ticker'] = ticker
                df_list.append(data.iloc[[-1]]) # only last row
            
            return pd.concat(df_list)
        except Exception as e:
            print(f"Error fetching latest data: {e}")
            return pd.DataFrame()
