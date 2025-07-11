"""
Rust Backtesting Engine - High-performance backtesting framework

A high-performance backtesting engine implemented in Rust with Python bindings,
designed for cryptocurrency and financial trading strategy evaluation.

Key Features:
- High-performance Rust backend for order execution and portfolio management
- Python interface for strategy development and analysis
- Support for multiple instruments, leverage, and order types
- Comprehensive performance metrics and visualization tools
- Compatible with Polars and Pandas for data handling

Example Usage:
    >>> from rustypy_backtest_engine import PyBacktest
    >>> backtest = PyBacktest(market_data, initial_cash=1000000.0)
    >>> results = backtest.run_strategy(my_strategy, start_time, end_time)
    >>> metrics = backtest.calculate_performance_metrics()
"""

# Import the Rust bindings
try:
    from .rustypy_backtest_engine import PyBacktest
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    PyBacktest = None

# Import Python interface classes
try:
    from .simple_rust_interface import PolarsBacktestEngine, RustExecutionEngine
except ImportError:
    PolarsBacktestEngine = None
    RustExecutionEngine = None

# Version information
__version__ = "0.1.0"
__author__ = "Haoqing Wu"
__license__ = "MIT"

# Public API
__all__ = [
    # Main backtesting engine
    "PyBacktest",
    # Alternative interfaces
    "PolarsBacktestEngine",
    "RustExecutionEngine", 
    # Utility
    "RUST_AVAILABLE",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]

# Ensure the main class is available
if not RUST_AVAILABLE:
    import warnings
    warnings.warn(
        "Rust bindings not available. Run 'uv run maturin develop' to build the engine.",
        ImportWarning
    )