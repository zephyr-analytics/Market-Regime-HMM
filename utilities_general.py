import pandas as pd
import scipy.stats as stats

def compounded_return(series, window):
    daily_returns = series.pct_change().fillna(0) + 1
    return daily_returns.rolling(window).apply(lambda x: x.prod() - 1, raw=True)


def smooth_states(states, window=5):
    """
    """
    return pd.Series(states).rolling(window, center=True, min_periods=1).apply(
        lambda x: stats.mode(x)[0][0], raw=False
    ).astype(int).values
