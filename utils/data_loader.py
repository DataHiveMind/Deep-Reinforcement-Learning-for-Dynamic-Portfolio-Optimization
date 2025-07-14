import pandas as pd 
import numpy as np
import yfinance as yf 
from dataclasses import dataclass

@dataclass
class DataLoader:
    ticker: str
    start_date: str = '2010-01-01'
    end_date: str = '2023-10-01'
    
    def load_data(self) -> pd.DataFrame:
        """
        Load historical stock data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: DataFrame containing the stock data with Date as index.
        """
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df.reset_index(inplace=True)
        df.set_index('Date', inplace=True)
        return df
    
    def get_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns from the adjusted close prices.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data with 'Adj Close' column.
        
        Returns:
            pd.Series: Series containing daily returns.
        """
        if 'Adj Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Adj Close' column.")
        
        returns = df['Adj Close'].pct_change().dropna()
        return returns
    