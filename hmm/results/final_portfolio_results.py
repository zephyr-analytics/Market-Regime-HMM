"""
"""
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import yfinance as yf

import logger_config
from hmm import utilities

logger = logging.getLogger(__name__)


class FinalResultsPortfolio:
    """
    """
    def __init__(self, results: pd.DataFrame, config: dict):
        self.results = results
        self.config = config

    def process(self):
        """
        """
        returns = pd.Series(
            data=[res['portfolio_return'] for res in self.results],
            index=pd.to_datetime([res['return_end'] for res in self.results])
        )
        returns.index.name = 'Date'
        returns.name = 'Return'
        portfolio_value = utilities.compute_portfolio_value(returns=returns)
        print(portfolio_value[-1])
        var, cvar = utilities.calculate_var_cvar(returns=returns)
        print(var, cvar)
        average_annual_return = utilities.calculate_average_annual_return(returns=returns)
        logger.info(f"Average Annual Return: {average_annual_return * 100:.2f}%")
        max_drawdown = utilities.calculate_max_drawdown(portfolio_value=portfolio_value)
        logger.info(f"Max Drawdown: {max_drawdown * 100:.2f}%")

        self.plot_portfolio_value(dates=returns.index, portfolio_value=portfolio_value, config=self.config)
        self.plot_var_cvar(returns=returns, var=var, cvar=cvar)
        self.plot_returns_heatmaps(returns=returns, average_annual_return=average_annual_return)

    @staticmethod
    def plot_portfolio_value(dates, portfolio_value, config, price_data=None, metrics=None, filename='portfolio_value'):
        """
        Plots the portfolio value over time, optionally including a benchmark and performance metrics.

        Parameters
        ----------
        dates : pd.Series
            Date index for the time series.
        portfolio_value : pd.Series
            Portfolio value time series aligned with `dates`.
        config : dict
            Configuration dictionary containing at least 'benchmark_ticker'.
        price_data : pd.DataFrame, optional
            Optional DataFrame with adjusted close prices including benchmark. Used if available.
        metrics : dict, optional
            Dictionary of performance metrics to annotate.
        filename : str
            Output filename (without extension) for saving the HTML plot.
        """
        fig = go.Figure()

        # Plot portfolio
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_value,
            mode='lines',
            name='Portfolio',
            line=dict(color='green')
        ))

        # Handle benchmark
        benchmark_ticker = config.get('benchmark_ticker')
        if benchmark_ticker:
            try:
                if price_data is not None and benchmark_ticker in price_data.columns:
                    benchmark_series = price_data[benchmark_ticker].loc[dates]
                else:
                    # Download from yfinance, ensure multi-index support
                    benchmark_data = yf.download(
                        benchmark_ticker,
                        start=dates.min(),
                        end=dates.max(),
                        auto_adjust=False,
                        group_by='ticker'
                    )

                    if isinstance(benchmark_data.columns, pd.MultiIndex):
                        benchmark_series = benchmark_data[(benchmark_ticker, 'Adj Close')]
                    else:
                        benchmark_series = benchmark_data['Adj Close']

                    benchmark_series = benchmark_series.reindex(dates).fillna(method='ffill')

                # Normalize benchmark to match portfolio starting value
                benchmark_series = benchmark_series / benchmark_series.iloc[0] * portfolio_value.iloc[0]

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=benchmark_series,
                    mode='lines',
                    name=benchmark_ticker,
                    line=dict(color='blue', dash='dash')
                ))
            except Exception as e:
                print(f"Failed to fetch or process benchmark data for {benchmark_ticker}: {e}")

        # Add annotations
        annotations = []
        if metrics:
            positions = [0.05 + i * 0.1 for i in range(len(metrics))]
            for (key, val), x in zip(metrics.items(), positions):
                annotations.append(dict(
                    xref='paper', yref='paper', x=x, y=1,
                    xanchor='center', yanchor='bottom',
                    text=f"{key}: {val:.2%}" if "Return" in key or "Drawdown" in key else f"{key}: ${val:,.2f}",
                    showarrow=False,
                    font=dict(size=12)
                ))

        fig.update_layout(
            template="plotly",
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            annotations=annotations,
            legend=dict(orientation="h", yanchor="bottom", y=0.1, xanchor="center", x=0.5)
        )

        utilities.save_html(fig, filename)


    @staticmethod
    def plot_var_cvar(returns, var, cvar, cagr=None, avg_return=None, max_dd=None, filename='var_cvar'):
        """
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns.dropna(),
            nbinsx=30,
            name="Returns",
            marker_color="#ab47bc",
            opacity=0.75
        ))

        fig.add_shape(
            type='line',
            x0=var, x1=var, y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color='red', dash='dash'),
            name='VaR'
        )

        fig.add_shape(
            type='line',
            x0=cvar, x1=cvar, y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color='orange', dash='dash'),
            name='CVaR'
        )

        annotations = []
        if cagr:
            annotations.append(dict(xref='paper', yref='paper', x=0.1, y=1, text=f'CAGR: {cagr:.2%}', showarrow=False))
        if avg_return:
            annotations.append(dict(xref='paper', yref='paper', x=0.3, y=1, text=f'Avg Return: {avg_return:.2%}', showarrow=False))
        if max_dd:
            annotations.append(dict(xref='paper', yref='paper', x=0.5, y=1, text=f'Max Drawdown: {max_dd:.2%}', showarrow=False))
        annotations.append(dict(xref='paper', yref='paper', x=0.7, y=1, text=f'VaR: {var:.2%}', showarrow=False))
        annotations.append(dict(xref='paper', yref='paper', x=0.9, y=1, text=f'CVaR: {cvar:.2%}', showarrow=False))

        fig.update_layout(
            template='plotly',
            title="Portfolio Returns with VaR and CVaR",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            showlegend=False,
            annotations=annotations
        )

        utilities.save_html(fig, filename)

    @staticmethod
    def plot_returns_heatmaps(returns, average_annual_return, filename='returns_heatmap'):
        """
        Plots a combined heatmap of monthly and yearly
        returns with values shown as percentages on each cell.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the plot. Default is 'returns_heatmap.html'.
        """
        monthly_returns = returns.copy()
        monthly_returns_df = monthly_returns.to_frame(name='Monthly Return')
        monthly_returns_df['Monthly Return'] *= 100
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month

        # Step 2: Identify the first year in the dataset
        first_year = monthly_returns_df['Year'].iloc[0]

        # Step 3: Check if the first year is partial (i.e., has fewer than 12 months)
        month_counts = monthly_returns_df.groupby('Year')['Month'].nunique()
        if month_counts.get(first_year, 0) < 12:
            # Remove data from the first year
            monthly_returns_df = monthly_returns_df[monthly_returns_df['Year'] != first_year]

        # Step 4: Recompute yearly returns from the trimmed monthly returns
        filtered_returns = monthly_returns_df['Monthly Return'] / 100  # convert back to decimal
        filtered_returns.index = pd.to_datetime(monthly_returns_df.index)

        yearly_returns = filtered_returns.resample('Y').apply(utilities.compound_returns)
        yearly_returns_df = yearly_returns.to_frame(name='Yearly Return')
        yearly_returns_df['Yearly Return'] *= 100  # Convert to percentage
        yearly_returns_df['Year'] = yearly_returns_df.index.year
        yearly_returns_df = yearly_returns_df.sort_values('Year')

        sharpe_ratio = average_annual_return / yearly_returns_df["Yearly Return"].std()
        logger.info(f"Sharpe Ratio: {sharpe_ratio * 100:.2f}")

        monthly_heatmap_data = monthly_returns_df.pivot('Year', 'Month', 'Monthly Return')
        monthly_heatmap_data = monthly_heatmap_data.reindex(columns=np.arange(1, 13))

        all_returns = np.concatenate([
            monthly_heatmap_data.values.flatten(),
            yearly_returns_df['Yearly Return'].values
        ])
        zmin, zmax = np.nanmin(all_returns), np.nanmax(all_returns)

        fig = sp.make_subplots(
            rows=2, cols=1,
            subplot_titles=("Monthly Returns Heatmap", "Yearly Returns Heatmap"),
            shared_xaxes=False,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.1
        )

        fig.add_trace(go.Heatmap(
            z=monthly_heatmap_data.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=monthly_heatmap_data.index,
            colorscale=[
                [0.0, 'red'],
                [(0 - zmin) / (zmax - zmin), 'white'],
                [1.0, 'green']
            ],
            zmin=zmin,
            zmax=zmax,
            showscale=False,
        ), row=1, col=1)
        monthly_annotations = []
        for i in range(monthly_heatmap_data.shape[0]):
            for j in range(monthly_heatmap_data.shape[1]):
                value = monthly_heatmap_data.iloc[i, j]
                if not np.isnan(value):
                    monthly_annotations.append(
                        dict(
                            text=f"{value:.2f}%",
                            x=[
                                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                            ][j],
                            y=monthly_heatmap_data.index[i],
                            xref='x1',
                            yref='y1',
                            font=dict(color="black"),
                            showarrow=False
                        )
                    )
        fig.add_trace(go.Heatmap(
            z=[yearly_returns_df['Yearly Return'].values],
            x=yearly_returns_df['Year'],
            y=["Yearly Returns"],
            colorscale=[
                [0.0, 'red'],
                [(0 - zmin) / (zmax - zmin), 'white'],
                [1.0, 'green']
            ],
            zmin=zmin,
            zmax=zmax,
            showscale=False,
        ), row=2, col=1)
        yearly_annotations = []
        for i in range(yearly_returns_df.shape[0]):
            value = yearly_returns_df['Yearly Return'].iloc[i]
            yearly_annotations.append(
                dict(
                    text=f"{value:.2f}%",
                    x=yearly_returns_df['Year'].iloc[i],
                    y="Yearly Returns",
                    xref='x2',
                    yref='y2',
                    font=dict(color="black"),
                    showarrow=False
                )
            )

        chart_theme = "plotly_dark"

        fig.update_layout(
            template=chart_theme,
            annotations=monthly_annotations + yearly_annotations + [
                dict(
                    xref='paper', yref='paper', x=0.5, y=0.5,
                    text="© Zephyr Analytics",
                    showarrow=False,
                    font=dict(size=80, color="#f8f9f9"),
                    xanchor='center',
                    yanchor='bottom',
                    opacity=0.5
                )
            ]
        )

        utilities.save_html(
            fig,
            filename
        )
