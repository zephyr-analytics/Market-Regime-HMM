"""
Utilities module for calculating portfolio statistics.
"""

import numpy as np
import pandas as pd


def calculate_cagr(portfolio_value):
    """
    Calculates the Compound Annual Growth Rate (CAGR) of the portfolio for Monte Carlo simulation.

    Parameters
    ----------
    portfolio_value : Series
        Series containing the portfolio value over time.

    Returns
    -------
    float
        CAGR value.
    """
    total_months = len(portfolio_value) - 1
    total_years = total_months / 12
    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / total_years) - 1

    return cagr


def calculate_average_annual_return(returns):
    """
    Calculates the average annual return of the portfolio.

    Parameters
    ----------
    returns : Series
        Series containing the portfolio returns over time.
    trading_frequency : str, optional
        The frequency of trades. Either 'monthly' or 'bi-monthly'. Default is 'monthly'.

    Returns
    -------
    float
        Average annual return.
    """
    average_periodic_return = returns.mean()
    average_annual_return = (1 + average_periodic_return) ** 12 - 1

    return average_annual_return


def calculate_max_drawdown(portfolio_value):
    """
    Calculates the maximum drawdown of the portfolio.

    Parameters
    ----------
    portfolio_value : Series
        Series containing the portfolio value over time.

    Returns
    -------
    float
        Maximum drawdown value.
    """
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value / running_max) - 1
    max_drawdown = drawdown.min()

    return max_drawdown


def calculate_var_cvar(returns, confidence_level=0.95):
    """
    Calculates the Value at Risk (VaR) and Conditional Value at Risk (CVaR) of the portfolio.

    Parameters
    ----------
    returns : Series
        Series containing the portfolio returns over time.
    confidence_level : float, optional
        The confidence level for calculating VaR and CVaR. Default is 0.95.

    Returns
    -------
    tuple
        Tuple containing VaR and CVaR values.
    """
    sorted_returns = np.sort(returns.dropna())
    index = int(np.floor((1 - confidence_level) * len(sorted_returns)))
    index = max(0, min(index, len(sorted_returns) - 1))
    var = sorted_returns[index]
    cvar = sorted_returns[:index + 1].mean()

    return var, cvar


def calculate_annual_volatility(portfolio_returns):
    """
    Calculates the annual volatility based on backtest results.

    Parameters
    ----------
    backtest : BacktestStaticPortfolio
        An instance of BacktestStaticPortfolio with completed backtest.

    Returns
    -------
    float
        The annual volatility of the portfolio.
    """
    annual_volatility = portfolio_returns.std() * np.sqrt(12)

    return annual_volatility


def calculate_standard_deviation(returns: pd.Series) -> float:
    """
    Calculates the standard deviation of the portfolio returns.

    Parameters
    ----------
    returns : Series
        Series containing the portfolio returns over time.

    Returns
    -------
    float
        Standard deviation of returns.
    """
    return returns.std()


def compute_portfolio_value(returns: pd.Series, initial_value: float = 10000.0) -> pd.Series:
    """
    Computes the portfolio value series by manually compounding returns.

    Parameters:
    ----------
    returns : pd.Series
        Series of periodic returns (e.g., daily, weekly, monthly), indexed by date.
    initial_value : float
        Starting value of the portfolio.

    Returns:
    -------
    pd.Series
        Series of portfolio values over time.
    """
    portfolio_values = []
    current_value = initial_value

    for r in returns:
        current_value *= (1 + r)
        portfolio_values.append(current_value)

    return pd.Series(portfolio_values, index=returns.index)

def compound_returns(returns):
    """
    Computes compounded yearly return from monthly returns.
    """
    compounded_yearly_returns = (returns + 1).prod() - 1

    return compounded_yearly_returns
