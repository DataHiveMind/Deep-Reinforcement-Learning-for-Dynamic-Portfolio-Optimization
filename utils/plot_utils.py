import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass

@dataclass
class PlotUtils:
    """
    A class to provide utility functions for plotting financial data.
    """
    
    df: pd.DataFrame

    
    def plot_price(self, title: str = "Stock Price", xlabel: str = "Date", ylabel: str = "Price") -> None:
        """
        Plot the stock price over time.
        
        Args:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['Adj Close'], label='Adjusted Close Price')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_returns(self, title: str = "Daily Returns", xlabel: str = "Date", ylabel: str = "Returns") -> None:
        """
        Plot the daily returns of the stock.
        
        Args:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        returns = self.df['Adj Close'].pct_change().dropna()
        
        plt.figure(figsize=(14, 7))
        sns.histplot(returns, bins=50, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def plot_correlation_matrix(self, title: str = "Correlation Matrix") -> None:
        """
        Plot the correlation matrix of the stock data.
        Args:
            title (str): Title of the plot.
        """
        correlation_matrix = self.df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title(title)
        plt.show()

    def plot_moving_average(self, window: int = 20, title: str = "Moving Average", xlabel: str = "Date", ylabel: str = "Price") -> None:
        """
        Plot the moving average of the stock price.
        Args:
            window (int): The window size for the moving average.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        moving_average = self.df['Adj Close'].rolling(window=window).mean()
        
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['Adj Close'], label='Adjusted Close Price', alpha=0.5)
        plt.plot(self.df.index, moving_average, label=f'{window}-Day Moving Average', color='orange')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_volatility(self, window: int = 20, title: str = "Volatility", xlabel: str = "Date", ylabel: str = "Volatility") -> None:
        """
        Plot the rolling volatility of the stock price.
        Args:
            window (int): The window size for calculating volatility.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        volatility = self.df['Adj Close'].pct_change().rolling(window=window).std()
        
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, volatility, label=f'{window}-Day Rolling Volatility', color='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_histogram(self, column: str, title: str = "Histogram", xlabel: str = "Value", ylabel: str = "Frequency") -> None:
        """
        Plot a histogram of a specified column in the DataFrame.
        Args:
            column (str): The column to plot.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        
        plt.figure(figsize=(14, 7))
        sns.histplot(self.df[column], bins=50, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def plot_boxplot(self, column: str, title: str = "Boxplot", xlabel: str = "Value", ylabel: str = "Frequency") -> None:
        """
        Plot a boxplot of a specified column in the DataFrame.
        Args:
            column (str): The column to plot.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        
        plt.figure(figsize=(14, 7))
        sns.boxplot(x=self.df[column])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def plot_time_series(self, columns: list, title: str = "Time Series Plot", xlabel: str = "Date", ylabel: str = "Value") -> None:
        """
        Plot multiple time series from specified columns in the DataFrame.
        
        Args:
            columns (list): List of columns to plot.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        if not all(col in self.df.columns for col in columns):
            raise ValueError("One or more columns not found in DataFrame.")
        
        plt.figure(figsize=(14, 7))
        for col in columns:
            plt.plot(self.df.index, self.df[col], label=col)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_scatter(self, x_column: str, y_column: str, title:
            str = "Scatter Plot", xlabel: str = "X-axis", ylabel: str = "Y-axis") -> None:
        """
        Plot a scatter plot of two specified columns in the DataFrame.
        
        Args:
            x_column (str): The column for the x-axis.
            y_column (str): The column for the y-axis.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        if x_column not in self.df.columns or y_column not in self.df.columns:
            raise ValueError("One or both columns not found in DataFrame.")
        
        plt.figure(figsize=(14, 7))
        plt.scatter(self.df[x_column], self.df[y_column], alpha=0.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def plot_heatmap(self, title: str = "Heatmap", xlabel: str =
            "X-axis", ylabel: str = "Y-axis") -> None:
        """
        Plot a heatmap of the correlation matrix of the DataFrame.
        
        Args:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """
        correlation_matrix = self.df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()