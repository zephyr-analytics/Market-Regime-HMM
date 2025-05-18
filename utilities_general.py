

def compounded_return(series, window):
    daily_returns = series.pct_change().fillna(0) + 1
    return daily_returns.rolling(window).apply(lambda x: x.prod() - 1, raw=True)
