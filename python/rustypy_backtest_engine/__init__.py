# Re-export PyBacktest class for cleaner imports
try:
    from .rustypy_backtest_engine import PyBacktest
    __all__ = ['PyBacktest']
except ImportError:
    # Fallback if the Rust module is not built yet
    print("Warning: Rust backtesting engine not built. Run 'maturin develop' to build.")
    PyBacktest = None
    __all__ = []