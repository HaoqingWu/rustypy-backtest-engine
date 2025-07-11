# RustyPy Backtesting Engine

**This is a project growing out of my own prop trading, with the need of efficient parameter-tuning in high-dimensional parameter space.**

A light-weight and high-performance backtesting engine implemented in Rust with Python bindings, designed for binance perpetual trading.


## Workflow

```
Python Strategy Logic → Rust Execution Engine → Python Analysis
```

## Features

- **High Performance**: Core backtesting logic implemented in Rust for performance. ~60% faster in most use cases than my Python backtest engine.

- **Python Integration**: Strategy development and performance analysis in Python so that you can still use your favorite python package.

## Installation

### Prerequisites

This project requires Rust to build the Python bindings. If you don't have Rust installed, installed it from rustup.

**Install Rust:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**Verify installation:**
```bash
rustc --version
cargo --version
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rustypy-backtest-engine

# Install dependencies and build
uv sync
uv run maturin develop
```

## Quick Start

### Basic Usage

```python
import polars as pl
from rustypy_backtest_engine import PyBacktest
from pybacktest_extensions import create_enhanced_backtest, PyBacktestPlotter

import datetime as dt

# Load sample data
market_data = {
    "BTCUSDT": pl.read_csv("../../tests/test_data/sample_data.csv")
        .with_columns(
            pl.col("Open Time").cast(pl.Datetime)
        )
}

# Create backtest engine
backtest = create_enhanced_backtest(market_data, initial_cash=1000000.0)

# Define your strategy with debug output
def buy_and_hold(status):
    orders = []
    
    # Simple buy-and-hold example
    positions = status.get("positions", {})
    current_prices = status.get("current_prices", {})
        
    if not positions.get("BTCUSDT", {}).get("long"):
        if "BTCUSDT" in current_prices:
            order = {
                "imnt": "BTCUSDT",
                "direction": "LONG",
                "order_type": "OPEN", 
                "size": 1.0,
                "leverage": 1,
                "price": current_prices["BTCUSDT"]
            }
            orders.append(order)

    return orders

# Check the datetime format and data

# Check the timestamp conversion
start_time = dt.datetime(2018, 1, 1)
end_time = dt.datetime(2018, 4, 1)

print(f"\nStart timestamp: {start_time}")
print(f"End timestamp: {end_time}")

# Run backtest
result = backtest.run_strategy(
            buy_and_hold,
            start_time,
            end_time,
            lookback_periods = 1,
            progress = True  # Disable progress to see debug output
        )
```

## Performance Metrics

The engine provides comprehensive performance analysis:

```python
metrics = backtest.calculate_performance_metrics()

# Risk-adjusted returns
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")

# Risk metrics
print(f"Max Drawdown: {metrics['max_drawdown_pct']*100:.2f}%")
print(f"VaR (95%): {metrics['cvar_5']:.2f}")
print(f"Volatility: {metrics['volatility']*100:.2f}%")

# Trading metrics
print(f"Total Trades: {metrics['total_trades']}")
print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
```

## Visualization and Analysis

```python
# Plot Equity Curve v.s. buy-and-hold
backtest.plotEquityCurve()

# Get Daily PnL by instrument
backtest.plotDailyPnL()

# Get detailed trade history
trade_history = backtest.get_trade_history()
```

## Data Format Requirements

Market data should be in OHLCV format with these columns:
- `Open Time`: Timestamp (datetime)
- `Open`, `High`, `Low`, `Close`: Price data (float)
- `Volume`: Trading volume (float)

Compatible with Binance API data format.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
