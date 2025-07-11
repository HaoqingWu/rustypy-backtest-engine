# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance backtesting engine migration project from Python to Rust. The system allows users to implement trading strategies in Python while executing the performance-critical backtesting logic in Rust, then returning results to Python for visualization and analysis.

**Architecture:** Python strategy interface → Rust backtesting engine → Python visualization

## Key Commands

### Building and Development
- `cargo build` - Build the Rust project in debug mode
- `cargo build --release` - Build optimized release version for Python bindings
- `cargo test` - Run all Rust unit tests
- `cargo test -- --nocapture` - Run tests with stdout output
- `cargo run` - Run the standalone Rust application
- `cargo check` - Fast compile check without building

### Python Integration
- `pip install -r requirements.txt` - Install Python dependencies
- `python -m pytest python/test_rust_integration.py` - Run Python tests
- `maturin develop` - Build and install Python module for development (if using maturin)

### Linting and Formatting
- `cargo fmt` - Format Rust code
- `cargo clippy` - Run Rust linter
- `cargo clippy -- -D warnings` - Run clippy with warnings as errors

## Code Architecture

### Core Components

**Rust Core (`src/backtesting/`)**
- `engine.rs` - Main backtesting execution engine with performance optimization
- `order.rs` - Order management and OrderTicket system
- `trade.rs` - Individual trade lifecycle and PnL calculations
- `position.rs` - Position management per instrument (long/short)
- `status.rs` - Backtesting state management and portfolio metrics

**Python Interface (`python/`)**
- `BacktestingEngine.py` - Original Python implementation (reference)
- `rustypy_backtest_engine/` - Python bindings to Rust engine
- `examples/` - Strategy implementation examples

**Key Data Flow:**
1. Strategy logic executes in Python and generates OrderTicket
2. OrderTicket passed to Rust engine for execution simulation
3. Market data processing and PnL calculations happen in Rust
4. Results returned to Python for analysis and visualization

### Dependencies and Features
- **PyO3** for Python-Rust bindings with `extension-module` feature
- **Polars** for high-performance DataFrame operations (replacing pandas)
- **Chrono** for timestamp handling with timezone support
- **Serde** for serialization between Python and Rust

### Important Design Patterns

**Order Execution Model:**
- Orders filled if execution price is between High/Low of next candle
- Supports market orders (None price) and limit orders
- Transaction fees (2 BPS) applied on notional value
- Margin requirements calculated as notional/leverage

**Position Management:**
- At most one LONG and one SHORT position per instrument
- Same direction orders aggregate positions with price averaging
- Opposite direction orders close existing positions first

**Performance Optimization:**
- Core backtesting loop runs in Rust for maximum performance
- Data structures optimized for financial calculations
- Minimal memory allocations during backtesting execution

## Development Workflow

### Adding New Features
1. Implement core logic in Rust modules under `src/backtesting/`
2. Add unit tests in the same file using `#[cfg(test)]`
3. Update Python bindings if needed in `src/python_bindings/`
4. Add integration tests in `tests/` directory
5. Update Python interface in `python/rustypy_backtest_engine/`

### Strategy Development
- Users implement strategies by overriding `myStrategy()` method
- Strategy returns `OrderTicket` with new orders for current time step
- Access to market data and current portfolio status provided
- Historical data available through `getCurrentAvailableData(lookBack)`

### Data Format Requirements
- Market data expected in OHLCV format with "Open Time" timestamp column
- All instruments must have matching timestamp arrays
- Polars DataFrame format used internally for performance
- Supports funding rate data for crypto instruments

## Testing Strategy
- Unit tests for individual components (Order, Trade, Position)
- Integration tests for complete backtesting workflows
- Python tests for interface compatibility
- Performance benchmarks comparing Python vs Rust execution