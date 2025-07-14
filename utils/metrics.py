import pandas as pd 
import numpy as np
from dataclasses import dataclass

@dataclass
class MetricsCalculator:
    """
    A class to calculate various financial metrics for a given DataFrame of stock data.
    """
    
    df: pd.DataFrame
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.01) -> float:
        """
        Calculate the Sharpe Ratio of the stock returns.
        
        Args:
            risk_free_rate (float): The risk-free rate to use in the calculation.
        
        Returns:
            float: The Sharpe Ratio.
        """
        returns = self.df['Adj Close'].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.01) -> float:
        """
        Calculate the Sortino Ratio of the stock returns.
        
        Args:
            risk_free_rate (float): The risk-free rate to use in the calculation.
        
        Returns:
            float: The Sortino Ratio.
        """
        returns = self.df['Adj Close'].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return sortino_ratio
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown of the stock returns.
        Returns:
            float: The maximum drawdown as a percentage.
        """
        cumulative_returns = (1 + self.df['Adj Close'].pct_change().dropna()).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown * 100
    
    def calculate_volatility(self) -> float:
        """
        Calculate the annualized volatility of the stock returns.
        Returns:
            float: The annualized volatility as a percentage.
        """
        returns = self.df['Adj Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        return volatility * 100
    
    def calculate_beta(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate the beta of the stock returns relative to a benchmark.
        Args:
            benchmark_returns (pd.Series): Series containing the benchmark returns.
        Returns:
            float: The beta of the stock returns.
        """
        stock_returns = self.df['Adj Close'].pct_change().dropna()
        covariance = np.cov(stock_returns, benchmark_returns)[0][1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance
        return beta
    
    def calculate_alpha(self, benchmark_returns: pd.Series, risk_free_rate: float = 0.01) -> float:
        """
        Calculate the alpha of the stock returns relative to a benchmark.
        Args:
            benchmark_returns (pd.Series): Series containing the benchmark returns.
            risk_free_rate (float): The risk-free rate to use in the calculation.
        Returns:

            float: The alpha of the stock returns.
        """
        stock_returns = self.df['Adj Close'].pct_change().dropna()
        excess_stock_returns = stock_returns - risk_free_rate / 252
        excess_benchmark_returns = benchmark_returns - risk_free_rate / 252
        beta = self.calculate_beta(benchmark_returns)
        alpha = excess_stock_returns.mean() - beta * excess_benchmark_returns.mean()
        return alpha * 252
    
    def calculate_correlation(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate the correlation between the stock returns and benchmark returns.
        Args:
            benchmark_returns (pd.Series): Series containing the benchmark returns.
        Returns:
            float: The correlation coefficient.
        """
        stock_returns = self.df['Adj Close'].pct_change().dropna()
        correlation = stock_returns.corr(benchmark_returns)
        return correlation
    
    def calculate_metrics(self, benchmark_returns: pd.Series, risk_free_rate: float = 0.01) -> dict:
        """ 
        Calculate all financial metrics and return them as a dictionary.
        Args:
            benchmark_returns (pd.Series): Series containing the benchmark returns.
            risk_free_rate (float): The risk-free rate to use in the calculations.
        Returns:
            dict: A dictionary containing all calculated metrics.
        """
        metrics = {
            'Sharpe Ratio': self.calculate_sharpe_ratio(risk_free_rate),
            'Sortino Ratio': self.calculate_sortino_ratio(risk_free_rate),
            'Max Drawdown': self.calculate_max_drawdown(),
            'Volatility': self.calculate_volatility(),
            'Beta': self.calculate_beta(benchmark_returns),
            'Alpha': self.calculate_alpha(benchmark_returns, risk_free_rate),
            'Correlation': self.calculate_correlation(benchmark_returns)
        }
        return metrics
    