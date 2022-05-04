import pandas as pd
from pandas import DataFrame

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def get_correlation(df:DataFrame, column1:str, column2:str) -> float:
    """
    Calculates the correlation between two columns in a dataframe.

    :param df: The dataframe containing the columns.
    :param column1: The first column.
    :param column2: The second column.

    :return: The correlation between the two columns.
    """
    return df[column1].corr(df[column2])

def get_lagging_correlation(df:DataFrame, column1:str, column2:str, lag: int) -> float:
    """
    Calculates the correlation between two columns in a dataframe with certain lag

    :param df: The dataframe containing the columns.
    :param column1: The first column.
    :param column2: The lagging column.

    :return: The correlation between the two columns.
    """
    return df[column1].corr(df[column2].shift(lag))

def impute_values_with_linear_regression(df:DataFrame, column: str) -> pd.Series:
    """
    Fills NaN values using linear regression interpolation.

    :param df: The dataframe containing the column.
    :param column: The column to be imputed.

    :return: The dataframe with the imputed values.
    """
    return df[column].interpolate()

def get_relative_strength_index(df:DataFrame, column:str, periods:int) -> pd.DataFrame:
    """
    Returns the Relative Strength Index for a given column in a dataframe.
    """
    close_delta = df[column].diff()

    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        
    rs = ma_up / ma_down
    df['{}_rsi'.format(column)] = 100 - (100/(1 + rs))
    
    return df

def get_bollinger_bands(df:DataFrame, column:str, period:int, std_dev:float) -> pd.DataFrame:
    """
    Calculates the Bollinger Bands for a given column in a dataframe.

    :param df: The dataframe containing the column.
    :param column: The column to be used.
    :param period: The period to be used.

    :return: The dataframe with the Bollinger Bands columns.
    """

    df['{}_upper_band'.format(column)] = df[column].rolling(period).mean() + (df[column].rolling(period).std() * std_dev)
    df['{}_lower_band'.format(column)] = df[column].rolling(period).mean() - (df[column].rolling(period).std() * std_dev)
    return df

def get_macd(df:DataFrame, column:str, period1:int, period2:int, period3:int) -> DataFrame:
    """
    Calculates the MACD for a given column in a dataframe.

    :param df: The dataframe containing the column.
    :param column: The column to be used.
    :param period1: The first period to be used.
    :param period2: The second period to be used.
    :param period3: The third period to be used.
    """

    df['{}_macd'.format(column)] = df[column].ewm(span=period1).mean() - df[column].ewm(span=period2).mean()
    df['{}_signal'.format(column)] = df['{}_macd'.format(column)].ewm(span=period3).mean()
    return df

def get_exponential_moving_average_over_period(df:DataFrame, column:str, period:int) -> pd.Series:
    """
    Calculates the exponential moving average over a given period.

    :param df: The dataframe containing the column.
    :param column: The column to be used.
    :param period: The period to be used.

    :return: The series with the moving average column.
    
    """
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_annual_return_percentage(starting_value, ending_value, period):
    """
    Calculates the annual return percentage.

    :param starting_value: The starting value.
    :param ending_value: The ending value.
    :param period: The period.

    :return: The annual return percentage.
    """
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    return_value = (ending_value - starting_value) / starting_value
    return (1 + return_value) ** (1 / period) - 1

def get_log_returns(df, column):
    """
    Calculates the log returns for a given column in a dataframe.

    :param df: The dataframe containing the column.
    :param column: The column to be used.

    :return: The series with the log returns column.
    """
    returns = np.log(df[column]/df[column].shift(1))
    return returns.fillna(0, inplace=True)
    
def calculate_volatility(df, periods, column):
    """
    Calculates the volatility for a given column in a dataframe.
    
    df: The dataframe containing the column.
    periods: The periods to be used.
    column: The column to be used.

    :return: The volatility series column and mean volatility.
    """
    returns = get_log_returns(df, column)

    volatility_series = returns.rolling(window=periods).std() * np.sqrt(periods)

    return volatility_series, volatility_series.mean()

def get_sharpe_ratio(df, periods, column):
    """
    Calculates the Sharpe ratio for a given column in a dataframe.

    :param df: The dataframe containing the column.
    :param periods: The periods to be used.
    :param column: The column to be used.

    :return: The Sharpe ratio.
    """
    returns = get_log_returns(df, column)
    volatility = calculate_volatility(df, periods, column)[0]

    return returns.mean()/volatility

def plot_bollinger_bands(df, column, period, std_dev):
    """
    Plots the Bollinger Bands for a given column in a dataframe.
    """
    df = get_bollinger_bands(df, column, period, std_dev)
    df.plot()
    plt.show()

def plot_macd(df, column, period1, period2, period3):
    """
    Plots the MACD for a given column in a dataframe.
    """
    df = get_macd(df, column, period1, period2, period3)
    df.plot()
    plt.show()

def plot_rsi(df, column, periods):
    """
    Plots the RSI for a given column in a dataframe.
    """
    df = get_relative_strength_index(df, column, periods)
    df.plot()
    plt.show()

def plot_against_benchmark(series, benchmark):
    """
    Plots a series against a benchmark.

    :param series: The series to be plotted.
    :param benchmark: The benchmark to be plotted.

    :return: The plot.
    """

    ax1 = series.plot(label=series.name, color='blue')
    benchmark.plot(label=benchmark.name, color='red', ax=ax1)

    ax1.show()

def plot_autocorrelation(series):
    """
    Plots the autocorrelation of a series.

    :param series: The series to be plotted.

    :return: The plot.
    """

    ax1 = series.autocorr().plot(label=series.name, color='blue')

    ax1.show()

def plot_std_dev(series):
    """
    Plots the standard deviation of a series.

    :param series: The series to be plotted.

    :return: The plot.
    """
    mean = series.mean()
    std = series.std()

    ax1 = plt.plot(series.index, (series - mean) / std)
    plt.fill_between(
        series.index,
        (series - mean) / std,
        (series - mean) / std,
        color='blue',
        alpha=0.2,
        ax = ax1
    )

    ax1.show()

def plot_fibonacci_retracement(df, output_file = "fibonacci_retracement.png"):
    """
    Plots the Fibonacci retracement of a pricing data.

    :param df: The pricing data.

    :return: The plot.
    """
    highest_swing = -1
    lowest_swing = -1

    for i in range(1,df.shape[0]-1):
        if df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1] and (highest_swing == -1 or df['High'][i] > df['High'][highest_swing]):
            highest_swing = i
        if df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1] and (lowest_swing == -1 or df['Low'][i] < df['Low'][lowest_swing]):
            lowest_swing = i

    ratios = [0,0.236, 0.382, 0.5 , 0.618, 0.786,1]
    colors = ["black","r","g","b","cyan","magenta","yellow"]
    levels = []
    max_level = df['High'][highest_swing]
    min_level = df['Low'][lowest_swing]
    for ratio in ratios:
        if highest_swing > lowest_swing: # Uptrend
            levels.append(max_level - (max_level-min_level)*ratio)
        else: # Downtrend
            levels.append(min_level + (max_level-min_level)*ratio)

    plt.rcParams['figure.figsize'] = [12, 7]
    plt.rc('font', size=14)
    plt.plot(df['Close'])
    start_date = df.index[min(highest_swing,lowest_swing)]
    end_date = df.index[max(highest_swing,lowest_swing)]
    for i in range(len(levels)):
        plt.hlines(levels[i],start_date, end_date,label="{:.1f}%".format(ratios[i]*100),colors=colors[i], linestyles="dashed")
        plt.legend()
    
    plt.savefig(output_file)

if __name__ == '__main__':
    df = yf.download('BTC-USD', start='2021-01-01', end='2022-01-01')

    plot_fibonacci_retracement(df)