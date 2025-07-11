"""
PyBacktest Extensions for Rust PyBacktest Engine

This module extends the Rust PyBacktest class with plotting functionality
and progress bar support using the same methods from the original BacktestingEngine.py.

Author: Haoqing Wu
Date: 2025-01-17
"""

import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
import polars as pl
from typing import Optional

try:
    from rustypy_backtest_engine import PyBacktest
    RUST_AVAILABLE = True
except ImportError:
    print("Warning: Rust bindings not available. Run 'maturin develop' to build.")
    PyBacktest = None
    RUST_AVAILABLE = False


class PyBacktestPlotter:
    """
    A class that adds plotting functionality to PyBacktest results.
    
    This class works with the Rust PyBacktest engine to provide
    - equity curve plotting, trade visualization,
    - daily PnL breakdown by instrument,
    - and drawdown analysis.
    """
    
    def __init__(self, backtest_engine, market_data: dict, start_time=None, end_time=None):
        """
        Initialize the plotter with backtest engine and market data.
        
        Args:
            backtest_engine: The PyBacktest engine instance
            market_data: Original market data dict (polars DataFrames)
            start_time: Start time for filtering plots
            end_time: End time for filtering plots
        """
        self.engine = backtest_engine
        self.market_data = market_data
        self.tradables = list(market_data.keys())
        self.start_time = start_time
        self.end_time = end_time

    def _filter_time_series_data(self, data, is_timestamp_tuple=False):
        """Filter time series data to the specified time range."""
        if self.start_time is None and self.end_time is None:
            return data
        
        start_timestamp = None
        end_timestamp = None
        
        if self.start_time is not None:
            if isinstance(self.start_time, dt.datetime):
                start_timestamp = self.start_time.timestamp()
            else:
                start_timestamp = self.start_time
                
        if self.end_time is not None:
            if isinstance(self.end_time, dt.datetime):
                end_timestamp = self.end_time.timestamp()
            else:
                end_timestamp = self.end_time
        
        if is_timestamp_tuple:
            # Filter list of (timestamp, value) tuples
            filtered_data = []
            for timestamp, value in data:
                if start_timestamp is not None and timestamp < start_timestamp:
                    continue
                if end_timestamp is not None and timestamp > end_timestamp:
                    continue
                filtered_data.append((timestamp, value))
            return filtered_data
        else:
            # For pandas Series/DataFrame with datetime index
            if hasattr(data, 'index'):
                mask = pd.Series(True, index=data.index)
                if start_timestamp is not None:
                    start_dt = pd.Timestamp(start_timestamp, unit='s')
                    mask &= (data.index >= start_dt)
                if end_timestamp is not None:
                    end_dt = pd.Timestamp(end_timestamp, unit='s')
                    mask &= (data.index <= end_dt)
                return data[mask]
            return data

    def plotEquityCurve(self, 
                       plot_trades: bool = False,
                       plot_btc_benchmark: bool = True) -> None:
        """ 
        Plot the equity curve of the backtest with dollar formatting on the y-axis, 
        trade markers, drawdown subplot, and optional BTC buy-and-hold comparison
        """
        
        # Get the portfolio value time series from Rust engine
        portfolio_values_data = self.engine.get_portfolio_values()
        
        # Apply time filtering
        portfolio_values_data = self._filter_time_series_data(portfolio_values_data, is_timestamp_tuple=True)
        
        # Convert to pandas Series
        timestamps = []
        values = []
        for timestamp, value in portfolio_values_data:
            timestamps.append(pd.Timestamp(timestamp, unit='s'))
            values.append(value)
            
        portfolio_value_series = pd.Series(data=values, index=timestamps, name='Portfolio Value')
        
        # Calculate drawdown series
        cum_max = portfolio_value_series.cummax()
        drawdown_series = (portfolio_value_series - cum_max) / cum_max * 100
        
        # Create BTC buy-and-hold benchmark if requested
        btc_equity_series = None
        btc_drawdown_series = None
        
        if plot_btc_benchmark and 'BTCUSDT' in self.tradables:
            # Get BTC price data for the same period
            start_time = portfolio_value_series.index[0]
            end_time = portfolio_value_series.index[-1]
            
            # Convert Polars DataFrame to Pandas for plotting
            btc_df = self.market_data['BTCUSDT'].to_pandas()
            btc_df = btc_df.set_index('Open Time')
            
            # Get BTC prices (daily close prices)
            btc_prices = btc_df['Close'].resample('D').last()
            btc_prices = btc_prices.loc[start_time:end_time]
            
            # Calculate BTC buy-and-hold equity curve
            initial_cash = portfolio_value_series.iloc[0]
            initial_btc_price = btc_prices.iloc[0]
            btc_shares = initial_cash / initial_btc_price  # Number of BTC shares bought
            btc_equity_series = btc_prices * btc_shares
            btc_equity_series.name = 'BTC Buy & Hold'
            
            # Calculate BTC drawdown
            btc_cum_max = btc_equity_series.cummax()
            btc_drawdown_series = (btc_equity_series - btc_cum_max) / btc_cum_max * 100
        
        # Create subplots: 2 rows, 1 column with shared x-axis
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Portfolio Equity vs BTC Buy & Hold", "Drawdown Comparison (%)")
        )
        
        # Add the strategy equity curve to the top subplot
        fig.add_trace(
            go.Scatter(
                x=portfolio_value_series.index,
                y=portfolio_value_series.values,
                mode='lines',
                name='Strategy',
                hoverinfo='x+y',
                line=dict(color='royalblue', width=2)
            ),
            row=1, col=1
        )
        
        # Add BTC buy-and-hold equity curve if available
        if btc_equity_series is not None:
            fig.add_trace(
                go.Scatter(
                    x = btc_equity_series.index,
                    y = btc_equity_series.values,
                    mode = 'lines',
                    name = 'BTC Buy & Hold',
                    hoverinfo = 'x+y',
                    line = dict(color='orange', width=2, dash='dash')
                ),
                row = 1, col = 1
            )
        
        # Add strategy drawdown to the bottom subplot
        fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values,
                mode='lines',
                name='Strategy (intraday) Drawdown',
                fill='tozeroy',
                fillcolor='rgba(65,105,225,0.2)',
                line=dict(color='royalblue'),
                hoverinfo='x+y',
                hovertemplate='%{x}<br>Strategy DD: %{y:.2f}%'
            ),
            row=2, col=1
        )
        
        # Add BTC drawdown to the bottom subplot
        if btc_drawdown_series is not None:
            fig.add_trace(
                go.Scatter(
                    x=btc_drawdown_series.index,
                    y=btc_drawdown_series.values,
                    mode='lines',
                    name='BTC (intraday) Drawdown',
                    line=dict(color = 'orange', dash='dash'),
                    hoverinfo='x+y',
                    hovertemplate='%{x}<br>BTC DD: %{y:.2f}%'
                ),
                row=2, col=1
            )
        
        # Calculate and display performance comparison
        initial_cash = portfolio_value_series.iloc[0]
        strategy_total_return = (portfolio_value_series.iloc[-1] / initial_cash - 1) * 100
        strategy_max_dd = drawdown_series.min()
        
        if btc_equity_series is not None:
            btc_total_return = (btc_equity_series.iloc[-1] / initial_cash - 1) * 100
            btc_max_dd = btc_drawdown_series.min()
            
            # Calculate daily returns for both
            strategy_returns = portfolio_value_series.pct_change().dropna()
            btc_returns = btc_equity_series.pct_change().dropna()
            corr = strategy_returns.corr( btc_returns )

            # Add performance comparison annotation
            comparison_text = (
                f"Strategy: {strategy_total_return:.1f}% return, {strategy_max_dd:.1f}% MDD <br>"
                f"BTC B&H: {btc_total_return:.1f}% return, {btc_max_dd:.1f}% MDD <br>"
                f"Strategy BTC Correlation: {corr:.2f}"
            )
            
            fig.add_annotation(
                text=comparison_text,
                xref="paper", yref="paper",
                x=0.95, y=0.98,  # Adjust these values to change position:
                                 # x: 0 (left) to 1 (right)
                                 # y: 0 (bottom) to 1 (top)
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
            )
        
        # Format y-axis on equity curve subplot
        fig.update_yaxes(
            title_text='Portfolio Value ($)',
            tickformat='$,.0f',
            row=1, col=1
        )
        
        # Format y-axis on drawdown subplot
        fig.update_yaxes(
            title_text='Drawdown (%)',
            tickformat='.1f',
            row=2, col=1
        )
        
        # Add horizontal line at y=0 on drawdown subplot
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Display the maximum drawdown value as a horizontal line
        max_dd = drawdown_series.min()
        fig.add_shape(
            type="line",
            x0=drawdown_series.index[0],
            y0=max_dd,
            x1=drawdown_series.index[-1],
            y1=max_dd,
            line=dict(color="red", width=1, dash="dash"),
            row=2, col=1
        )
        
        # Add annotation for maximum drawdown
        max_dd_date = drawdown_series.idxmin()
        fig.add_annotation(
            x=max_dd_date,
            y=max_dd,
            text=f"Max Drawdown: {max_dd:.2f}%",
            showarrow=True,
            arrowhead=1,
            row=2, col=1,
            arrowcolor="red",
            font=dict(color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1
        )
        

        # Update layout
        fig.update_layout(
            title='Equity Curve and Drawdown',
            hovermode='x unified',
            height=800,
            margin=dict(t=80, b=50, l=50, r=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Show the plot
        fig.show()

    def plotEntryExitTrades(self) -> None:
        """ Plot the entry and exit trades on the price graph of each instrument using Plotly """
        
        # Get trade history from Rust engine as dict of Polars DataFrames
        trade_history_dict = self.engine.get_trade_history()
        
        # Apply time filtering to trade history
        if self.start_time is not None or self.end_time is not None:
            start_timestamp = None
            end_timestamp = None
            
            if self.start_time is not None:
                start_timestamp = self.start_time.timestamp() if isinstance(self.start_time, dt.datetime) else self.start_time
            if self.end_time is not None:
                end_timestamp = self.end_time.timestamp() if isinstance(self.end_time, dt.datetime) else self.end_time
            
            # Filter trade history by time range
            for instrument in trade_history_dict:
                if hasattr(trade_history_dict[instrument], 'to_pandas'):
                    trades_df = trade_history_dict[instrument].to_pandas()
                else:
                    trades_df = trade_history_dict[instrument]
                
                if not trades_df.empty:
                    # Convert Open Time to timestamp for filtering
                    trades_df['Open Time'] = pd.to_datetime(trades_df['Open Time'])
                    mask = pd.Series(True, index=trades_df.index)
                    
                    if start_timestamp is not None:
                        start_dt = pd.Timestamp(start_timestamp, unit='s')
                        mask &= (trades_df['Open Time'] >= start_dt)
                    if end_timestamp is not None:
                        end_dt = pd.Timestamp(end_timestamp, unit='s')
                        mask &= (trades_df['Open Time'] <= end_dt)
                    
                    trade_history_dict[instrument] = trades_df[mask]

        for imnt in self.tradables:
            # Check if this instrument has trades
            if imnt not in trade_history_dict:
                continue
                
            # Get the DataFrame for this instrument
            instrument_trades_df = trade_history_dict[imnt]
            
            # Convert to pandas for easier manipulation
            if hasattr(instrument_trades_df, 'to_pandas'):
                instrument_trades = instrument_trades_df.to_pandas()
            else:
                instrument_trades = instrument_trades_df  # Already pandas
            
            # Check if DataFrame is empty
            if instrument_trades.empty:
                continue
                
            # Convert Polars DataFrame to Pandas for plotting
            imnt_df = self.market_data[imnt].to_pandas()
            imnt_df = imnt_df.set_index('Open Time')
            
            # Apply time filtering to price data
            if self.start_time is not None or self.end_time is not None:
                start_timestamp = None
                end_timestamp = None
                
                if self.start_time is not None:
                    start_timestamp = self.start_time.timestamp() if isinstance(self.start_time, dt.datetime) else self.start_time
                if self.end_time is not None:
                    end_timestamp = self.end_time.timestamp() if isinstance(self.end_time, dt.datetime) else self.end_time
                
                mask = pd.Series(True, index=imnt_df.index)
                if start_timestamp is not None:
                    start_dt = pd.Timestamp(start_timestamp, unit='s')
                    mask &= (imnt_df.index >= start_dt)
                if end_timestamp is not None:
                    end_dt = pd.Timestamp(end_timestamp, unit='s')
                    mask &= (imnt_df.index <= end_dt)
                
                imnt_df = imnt_df[mask]
            
            # Get the price data for the instrument
            price_data = imnt_df['Close']

            # Create a plotly figure
            fig = go.Figure()

            # Add the price data to the figure
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data.values,
                mode='lines',
                name='Price',
                hoverinfo='x+y',
                line=dict(color='blue')
            ))

            # Plot trades with different colors and symbols based on direction
            for _, trade in instrument_trades.iterrows():
                # Get trade direction and determine colors/symbols
                direction = trade['Direction']
                is_long = 'Long' in direction or 'LONG' in direction
                
                # Set colors and symbols based on trade direction
                if is_long:
                    open_color = 'green'
                    close_color = 'green'
                    open_symbol = 'circle'
                    close_symbol = 'x'
                    trade_type = 'LONG'
                else:  # Short trade
                    open_color = 'red'
                    close_color = 'red'
                    open_symbol = 'circle'
                    close_symbol = 'x'
                    trade_type = 'SHORT'
                
                # Plot the entry (open) of the trade
                open_time = pd.Timestamp(trade['Open Time'])
                entry_price = trade['Entry Price']
                
                fig.add_trace(go.Scatter(
                    x=[open_time],
                    y=[entry_price],
                    mode='markers',
                    marker=dict(color=open_color, symbol=open_symbol, size=10),
                    name=f'{trade_type} Open',
                    hoverinfo='text',
                    text=f'Open {trade_type} {imnt}<br>Open Time: {open_time}<br>Entry Price: ${entry_price:.2f}',
                    showlegend=False  # Don't clutter legend with individual trades
                ))

                # Plot the exit (close) of the trade only if it's closed
                close_time_str = trade['Close Time']
                if close_time_str != "Open":  # Only plot if the trade is closed
                    close_time = pd.Timestamp(close_time_str)
                    
                    # Get market close price at close_time
                    def get_market_price_at_time(price_series, target_time):
                        """Get the market price closest to the target time"""
                        try:
                            # Method 1: Try exact match
                            if target_time in price_series.index:
                                return price_series.loc[target_time]
                            
                            # Method 2: Use forward fill (get last available price before target time)
                            before_prices = price_series.loc[:target_time]
                            if len(before_prices) > 0:
                                return before_prices.iloc[-1]
                            
                            # Method 3: Use nearest neighbor
                            closest_idx = price_series.index.get_indexer([target_time], method='nearest')[0]
                            return price_series.iloc[closest_idx]
                            
                        except (IndexError, KeyError):
                            # Fallback: return entry price
                            return entry_price
                    
                    exit_price = get_market_price_at_time(price_data, close_time)
                    
                    fig.add_trace(go.Scatter(
                        x=[close_time],
                        y=[exit_price],
                        mode='markers',
                        marker=dict(color=close_color, symbol=close_symbol, size=10),
                        name=f'{trade_type} Close',
                        hoverinfo='text',
                        text=f'Close {trade_type} {imnt}<br>Close Time: {close_time}<br>Market Price: ${exit_price:.2f}<br>PnL: ${trade["PnL"]:.2f}',
                        showlegend=False  # Don't clutter legend with individual trades
                    ))

            # Add legend entries manually to show what the symbols mean
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color='green', symbol='circle', size=10),
                name='LONG Open',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color='green', symbol='x', size=10),
                name='LONG Close',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color='red', symbol='circle', size=10),
                name='SHORT Open',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color='red', symbol='x', size=10),
                name='SHORT Close',
                showlegend=True
            ))

            # Set y-axis label to emphasize that values are in dollars
            fig.update_layout(
                yaxis=dict(
                    title='Price ($)',
                    tickformat='$,.2f'
                ),
                title=f'Price and Trades for {imnt}',
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )

            # Show the plot
            fig.show()
            
    def plotDailyPnL(self) -> None:
        """ Plot each instrument's daily PnL as a stacked bar chart with different colors using Plotly """

        # Get daily PnL data from Rust engine
        cum_pnl_data = self.engine.get_cum_pnl_by_instrument()
        
        # Convert to DataFrame and apply time filtering
        daily_pnl_dict = {}
        for instrument, pnl_series in cum_pnl_data.items():
            # Apply time filtering first
            pnl_series = self._filter_time_series_data(pnl_series, is_timestamp_tuple=True)
            
            timestamps, pnl_values = zip(*pnl_series) if pnl_series else ([], [])
            timestamps = pd.to_datetime(timestamps, unit='s')
            daily_pnl_dict[instrument] = pd.Series(data=pnl_values, index=timestamps).diff()
            # Fill NaN values with 0 for the first entry
            daily_pnl_dict[instrument].fillna(0, inplace=True)
        
        daily_pnl_df = pd.DataFrame(daily_pnl_dict)
        
        # Resample to daily if needed
        daily_pnl_df = daily_pnl_df.resample('D').sum()

        # Create a plotly figure
        fig = go.Figure()

        # Define colors for each instrument using plotly's built-in colorscales
        colorscale = px.colors.qualitative.Plotly[:len(daily_pnl_df.columns)]

        # Iterate over each instrument (column) and add a bar trace - EXACT SAME AS ORIGINAL
        for i, instrument in enumerate(daily_pnl_df.columns):
            fig.add_trace(go.Bar(
                x=daily_pnl_df.index,
                y=daily_pnl_df[instrument],
                name=instrument,
                marker_color=colorscale[i % len(colorscale)]
            ))

        # Configure the layout for a stacked bar chart - EXACT SAME AS ORIGINAL
        fig.update_layout(
            title="Daily PnL by Instrument (Stacked Bar Chart)",
            xaxis_title="Date",
            yaxis_title="PnL (in $)",
            barmode='stack',
            yaxis=dict(
                tickformat="$,.0f",
            ),
            legend=dict(
                title="Instruments",
                x=1.05,
                y=1,
                xanchor='left'
            ),
        )

        # Show the plot
        fig.show()

    def get_daily_pnl_dataframe(self) -> pd.DataFrame:
        """
        Get daily PnL by instrument as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with dates as index and instruments as columns,
                         containing daily PnL values for each instrument.
        """
        # Get cumulative PnL data from Rust engine
        cum_pnl_data = self.engine.get_cum_pnl_by_instrument()
        
        # Convert to DataFrame and apply time filtering
        cum_pnl_dict = {}
        for instrument, pnl_series in cum_pnl_data.items():
            # Apply time filtering first
            pnl_series = self._filter_time_series_data(pnl_series, is_timestamp_tuple=True)
            
            if not pnl_series:  # Handle empty series
                continue
                
            timestamps, cum_pnl_values = zip(*pnl_series)
            timestamps = pd.to_datetime(timestamps, unit='s')
            cum_pnl_dict[instrument] = pd.Series(data=cum_pnl_values, index=timestamps)
        
        if not cum_pnl_dict:  # Handle case with no data
            return pd.DataFrame()
            
        # Create DataFrame from cumulative PnL
        cum_pnl_df = pd.DataFrame(cum_pnl_dict)
        
        # Resample to daily frequency first (take last value of each day for cumulative PnL)
        daily_cum_pnl_df = cum_pnl_df.resample('D').last()
        
        # Take difference to get daily PnL changes
        daily_pnl_df = daily_cum_pnl_df.diff()
        
        # Fill NaN values with 0 for the first entry (or use the first day's cumulative PnL)
        daily_pnl_df.iloc[0] = daily_cum_pnl_df.iloc[0]  # First day PnL = first day cumulative PnL
        
        return daily_pnl_df


class EnhancedPyBacktest:
    """
    Enhanced PyBacktest wrapper that includes plotting functionality.
    
    This class wraps the Rust PyBacktest and adds plotting methods
    while maintaining the same interface as the original BacktestingEngine.py.
    """
    
    def __init__(self, market_data, initial_cash: float, data_frequency: str = "4h", 
                 log_level: int = 3, log_file: Optional[str] = None):
        """
        Initialize enhanced PyBacktest with plotting capabilities.
        
        Args:
            market_data: Market data dictionary (polars DataFrames)
            initial_cash: Initial cash amount
            data_frequency: Data frequency string
            log_level: Logging level
            log_file: Optional log file path
        """
        if not RUST_AVAILABLE:
            raise ImportError("Rust bindings not available. Run 'maturin develop' to build.")
            
        # Initialize the Rust engine
        self.engine = PyBacktest(market_data, initial_cash, data_frequency, log_level, log_file)
        self.initial_cash = initial_cash
        self.market_data = market_data
        self.tradables = list(market_data.keys())
        
        # Store results for plotting
        self._last_strategy_result = None
        self._last_metrics = None
        self._start_time = None
        self._end_time = None
        
    def run_strategy( self, 
                      strategy_func, 
                      start_time = None, 
                      end_time = None, 
                      lookback_periods: int = 50, 
                      progress: bool = True ):
        """
        Run strategy with optional progress bar and store results for plotting.
        
        Args:
            strategy_func: Strategy function
            start_time: Start time (datetime object or timestamp)
            end_time: End time (datetime object or timestamp)
            lookback_periods: Number of lookback periods
            progress: Whether to show progress bar (default: True)
            
        Returns:
            dict: Strategy results
        """
        # Convert datetime objects to timestamps for Rust engine
        start_timestamp = None
        end_timestamp = None
        
        if start_time is not None:
            if isinstance( start_time, dt.datetime ):
                start_timestamp = start_time.timestamp()
            else:
                start_timestamp = start_time  # Assume it's already a timestamp
        
        if end_time is not None:
            if isinstance( end_time, dt.datetime ):
                end_timestamp = end_time.timestamp()
            else:
                end_timestamp = end_time  # Assume it's already a timestamp
        
        if not progress:
            # No progress bar - use original method
            result = self.engine.run_strategy( strategy_func, start_timestamp, end_timestamp, lookback_periods )
            self._last_strategy_result = result
            self._start_time = start_time
            self._end_time = end_time
            return result
        
        # Get total periods from market data between start_time and end_time
        total_periods = 1000  # Default fallback
        if self.market_data:
            first_instrument = next( iter( self.market_data.keys() ), None )
            if first_instrument and hasattr( self.market_data[ first_instrument ], 'height' ):
                df = self.market_data[ first_instrument ]
                
                # If start_time and end_time are provided, filter to get actual period count
                if start_timestamp is not None and end_timestamp is not None:
                    # Convert timestamps to datetime for filtering
                    import pandas as pd
                    start_dt = pd.Timestamp(start_timestamp, unit='s')
                    end_dt = pd.Timestamp(end_timestamp, unit='s')
                    
                    # Filter the dataframe to the time range
                    time_mask = (df['Open Time'] >= start_dt) & (df['Open Time'] <= end_dt)
                    filtered_df = df.filter(time_mask)
                    total_periods = filtered_df.height
                else:
                    # Use full dataset if no time range specified
                    total_periods = df.height
        
        # Import progress bar dependencies
        try:
            from tqdm import tqdm
            import time
        except ImportError:
            print( "Warning: tqdm not available. Running without progress bar." )
            result = self.engine.run_strategy( strategy_func, start_timestamp, end_timestamp, lookback_periods )
            self._last_strategy_result = result
            return result
        
        print( f"Starting backtest with progress tracking..." )
        print( f"Total periods: {total_periods:,}" )
        if start_time and end_time:
            print( f"Period: {start_time} to {end_time}" )
        
        # Create progress bar
        pbar = tqdm(
            total = total_periods,
            desc = "Backtesting",
        )
        
        # Counter for progress updates
        call_count = [ 0 ]
        update_frequency = max( 1, total_periods // 200 )  # Update every 0.5% of progress
        
        def strategy_with_progress( context ):
            """Wrapper that updates progress and calls original strategy"""
            call_count[ 0 ] += 1
            
            # Update progress bar periodically
            if call_count[ 0 ] % update_frequency == 0 or call_count[ 0 ] == 1:
                pbar.update( update_frequency )
                
                # Update progress bar postfix with current stats
                portfolio_value = context.get( "portfolio_value", 0 )
                cash = context.get( "cash", 0 )
                
                postfix_dict = {}
                if portfolio_value:
                    postfix_dict[ 'Portfolio' ] = f"${portfolio_value:,.0f}"
                if cash:
                    postfix_dict[ 'Cash' ] = f"${cash:,.0f}"
                
                if postfix_dict:
                    pbar.set_postfix( postfix_dict )
            
            # Call the original strategy function
            return strategy_func( context )
        
        try:
            # Run the backtest with progress tracking
            start_time_exec = time.time()
            result = self.engine.run_strategy(
                strategy_with_progress, start_timestamp, end_timestamp, lookback_periods
            )
            execution_time = time.time() - start_time_exec
            
            # Complete the progress bar and show final stats
            pbar.n = total_periods  # Ensure we reach 100%
            final_postfix = {
                'Final': f"${result.get( 'final_portfolio_value', 0 ):,.0f}",
                'Return': f"{result.get( 'total_return', 0 ) * 100:.2f}%",
                'Time': f"{execution_time:.1f}s"
            }
            pbar.set_postfix( final_postfix )
            pbar.close()
            
            print( f"Backtest completed in {execution_time:.2f} seconds" )
            print( f"Final portfolio value: ${result.get( 'final_portfolio_value', 0 ):,.2f}" )
            print( f"Total return: {result.get( 'total_return', 0 ) * 100:.2f}%" )
            
            self._last_strategy_result = result
            self._start_time = start_time
            self._end_time = end_time
            return result
            
        except Exception as e:
            pbar.close()
            print( f"Backtest failed: {e}" )
            raise e
    
    def calculate_performance_metrics(self):
        """Calculate and store performance metrics."""
        metrics = self.engine.calculate_performance_metrics()
        self._last_metrics = metrics
        return metrics
    
    def _filter_time_series_data(self, data, is_timestamp_tuple=False):
        """
        Filter time series data to only include the specified time range.
        
        Args:
            data: Time series data (list of tuples or pandas Series/DataFrame)
            is_timestamp_tuple: Whether data is list of (timestamp, value) tuples
            
        Returns:
            Filtered data in the same format
        """
        if self._start_time is None and self._end_time is None:
            return data
        
        start_timestamp = None
        end_timestamp = None
        
        if self._start_time is not None:
            if isinstance(self._start_time, dt.datetime):
                start_timestamp = self._start_time.timestamp()
            else:
                start_timestamp = self._start_time
                
        if self._end_time is not None:
            if isinstance(self._end_time, dt.datetime):
                end_timestamp = self._end_time.timestamp()
            else:
                end_timestamp = self._end_time
        
        if is_timestamp_tuple:
            # Filter list of (timestamp, value) tuples
            filtered_data = []
            for timestamp, value in data:
                if start_timestamp is not None and timestamp < start_timestamp:
                    continue
                if end_timestamp is not None and timestamp > end_timestamp:
                    continue
                filtered_data.append((timestamp, value))
            return filtered_data
        else:
            # For pandas Series/DataFrame with datetime index
            if hasattr(data, 'index'):
                mask = pd.Series(True, index=data.index)
                if start_timestamp is not None:
                    start_dt = pd.Timestamp(start_timestamp, unit='s')
                    mask &= (data.index >= start_dt)
                if end_timestamp is not None:
                    end_dt = pd.Timestamp(end_timestamp, unit='s')
                    mask &= (data.index <= end_dt)
                return data[mask]
            return data

    def get_plotter(self) -> PyBacktestPlotter:
        """
        Get a plotter instance for the last strategy run.
        
        Returns:
            PyBacktestPlotter: Plotter instance
        """
        if self._last_strategy_result is None:
            raise ValueError("No strategy results available. Run a strategy first.")
            
        return PyBacktestPlotter(self.engine, self.market_data, self._start_time, self._end_time)
    
    def plotEquityCurve(self, **kwargs) -> None:
        """Plot equity curve using the last strategy results."""
        plotter = self.get_plotter()
        plotter.plotEquityCurve(**kwargs)
    
    def plotEntryExitTrades(self) -> None:
        """Plot entry/exit trades using the last strategy results."""
        plotter = self.get_plotter()
        plotter.plotEntryExitTrades()
    
    def plotDailyPnL(self) -> None:
        """Plot daily PnL using the last strategy results."""
        plotter = self.get_plotter()
        plotter.plotDailyPnL()
    
    def get_daily_pnl_dataframe(self) -> pd.DataFrame:
        """
        Get daily PnL by instrument as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with dates as index and instruments as columns,
                         containing daily PnL values for each instrument.
        """
        plotter = self.get_plotter()
        return plotter.get_daily_pnl_dataframe()
    
    def plot_all(self) -> None:
        """Plot all available charts."""
        # Calculate metrics if not done already
        if self._last_metrics is None:
            self.calculate_performance_metrics()
            
        print("📊 Plotting all charts...")
        
        # Plot equity curve
        print("1. Equity Curve and Drawdown")
        self.plotEquityCurve()
        
        # Plot entry/exit trades
        print("2. Entry/Exit Trades")
        self.plotEntryExitTrades()
        
        # Plot daily PnL
        print("3. Daily PnL by Instrument")
        self.plotDailyPnL()
        
        print("✅ All plots generated!")
    
    # Delegate all other methods to the underlying engine
    def __getattr__(self, name):
        """Delegate unknown methods to the underlying engine."""
        return getattr(self.engine, name)


# Convenience function to create enhanced backtest
def create_enhanced_backtest(market_data, initial_cash: float, **kwargs) -> EnhancedPyBacktest:
    """
    Create an enhanced PyBacktest instance with plotting capabilities.
    
    Args:
        market_data: Market data dictionary (polars DataFrames)
        initial_cash: Initial cash amount
        **kwargs: Additional arguments for PyBacktest
        
    Returns:
        EnhancedPyBacktest: Enhanced backtest instance
    """
    return EnhancedPyBacktest(market_data, initial_cash, **kwargs)




def demo_plotting():
    """Demonstrate the plotting functionality."""
    if not RUST_AVAILABLE:
        print("Demo requires Rust bindings. Run 'maturin develop' first.")
        return
        
    print("Enhanced PyBacktest with Plotting Demo")
    print("=====================================")
    print()
    print("Usage example:")
    print("""
    from pybacktest_extensions import create_enhanced_backtest
    
    # Create enhanced backtest with plotting capabilities
    backtest = create_enhanced_backtest(market_data, 1000000.0, log_level=0)
    
    # Run your strategy
    result = backtest.run_strategy(your_strategy_function, start_time, end_time)
    
    # Calculate metrics
    metrics = backtest.calculate_performance_metrics()
    
    # Plot individual charts
    backtest.plotEquityCurve()              # Equity curve with drawdown
    backtest.plotEntryExitTrades()          # Trade entry/exit points
    backtest.plotDailyPnL()                 # Daily PnL by instrument
    
    # Or plot everything at once
    backtest.plot_all()
    
    # You can also access the underlying Rust engine directly
    portfolio_values = backtest.get_portfolio_values()
    trade_history = backtest.get_trade_history()
    daily_pnl = backtest.get_cum_pnl_by_instrument()
    """)
    
    print("\nKey Features:")
    print("Same plotting interface as original BacktestingEngine.py")
    print("Uses actual portfolio values from Rust engine")
    print("Plotly interactive charts with hover information")
    print("BTC buy-and-hold benchmark comparison")
    print("Trade entry/exit visualization")
    print("Daily PnL breakdown by instrument")
    print("Automatic drawdown calculation and visualization")


if __name__ == "__main__":
    demo_plotting()