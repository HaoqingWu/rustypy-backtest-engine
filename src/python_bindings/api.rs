// Python Wrapper for the Rust backtesting engine.
// This module provides a Python interface to the high-performance Rust backtesting core,
// enabling users to implement trading strategies in Python while benefiting from Rust's speed.

use crate::backtesting::engine::Backtest;
use crate::backtesting::order::OrderTicket;
use crate::backtesting::status::BacktestStatus;
use crate::utils::types::{Direction, OrderType};
use crate::utils::logger::{LogLevel, init_logger};
use chrono::{DateTime, Utc};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

const TRANSACTION_FEE: f64 = 0.0002; // 2 BPS


/// Parse data frequency from string format to minutes
/// Supports: "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
fn parse_frequency_string(freq_str: &str) -> Result<f64, String> {
    let freq_lower = freq_str.to_lowercase();
    
    if freq_lower.ends_with("m") {
        // Minutes format: "1m", "5m", "15m", "30m"
        let num_str = &freq_lower[..freq_lower.len()-1];
        num_str.parse::<f64>()
            .map_err(|_| format!("Invalid minute format: {}", freq_str))
    } else if freq_lower.ends_with("h") {
        // Hours format: "1h", "2h", "4h", "6h", "12h"
        let num_str = &freq_lower[..freq_lower.len()-1];
        let hours: f64 = num_str.parse()
            .map_err(|_| format!("Invalid hour format: {}", freq_str))?;
        Ok(hours * 60.0) // Convert to minutes
    } else if freq_lower.ends_with("d") {
        // Days format: "1d"
        let num_str = &freq_lower[..freq_lower.len()-1];
        let days: f64 = num_str.parse()
            .map_err(|_| format!("Invalid day format: {}", freq_str))?;
        Ok(days * 24.0 * 60.0) // Convert to minutes
    } else {
        Err(format!("Unsupported frequency format: {}. Use formats like '1m', '5m', '1h', '4h', '1d'", freq_str))
    }
}

#[pyclass]
pub struct PyBacktest {
    inner: Backtest,
    last_results: Option<HashMap<DateTime<Utc>, BacktestStatus>>,
}

#[pymethods]
impl PyBacktest {
    /// Create a new backtesting instance.
    /// 
    /// # Arguments
    /// * `market_data` - HashMap of instrument symbols to Polars DataFrames containing OHLCV data
    /// * `initial_cash` - Starting cash amount for the backtest
    /// * `data_frequency` - Data frequency as string ("1m", "5m", "1h", "4h", "1d") or minutes as float (default: "4h")
    /// * `log_level` - Optional logging level: 0=Silent, 1=Error, 2=Warn, 3=Info, 4=Debug (default: 3)
    /// * `log_file` - Optional path to log file. If None, logs to console only
    /// 
    /// # Returns
    /// * `PyResult<Self>` - New PyBacktest instance or error
    /// 
    /// # Errors
    /// * Returns error if market data is empty or invalid
    /// * Returns error if DataFrames have mismatched lengths
    /// * Returns error if data_frequency format is invalid
    #[new]
    #[pyo3(signature = (market_data, initial_cash, data_frequency="4h", log_level=3, log_file=None))]
    pub fn new(
        market_data: HashMap<String, PyDataFrame>, 
        initial_cash: f64,
        data_frequency: &str,
        log_level: Option<u8>,
        log_file: Option<String>,
    ) -> PyResult<Self> {
        // Initialize logger with provided settings
        let level = LogLevel::from(log_level.unwrap_or(3));
        let log_file_path = log_file.as_deref();
        
        let _ = init_logger(level, log_file_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to initialize logger: {}", e)))?;
        
        // Convert PyDataFrame to Polars DataFrame
        let mut rust_market_data = HashMap::new();

        for (symbol, py_df) in market_data {
            let df: DataFrame = py_df.into();
            rust_market_data.insert(symbol, df);
        }

        // Parse data frequency
        let frequency_minutes = parse_frequency_string(data_frequency)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        let backtest = Backtest::new_with_frequency(rust_market_data, initial_cash, frequency_minutes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(PyBacktest {
            inner: backtest,
            last_results: None,
        })
    }

    /// Run a trading strategy on the backtesting engine.
    /// 
    /// # Arguments
    /// * `py` - Python interpreter reference
    /// * `strategy_func` - Python function that implements the trading strategy
    /// * `start_time_opt` - Optional start time as Unix timestamp in seconds
    /// * `end_time_opt` - Optional end time as Unix timestamp in seconds
    /// * `lookback_periods` - Optional number of historical periods to provide to strategy (default: 50)
    /// 
    /// # Returns
    /// * `PyResult<PyObject>` - Dictionary containing backtest results
    /// 
    /// # Strategy Function Interface
    /// The strategy function should accept a context dictionary and return a list of order dictionaries.
    /// Context includes 'current_data', a slice of your input dataframe with customized features.
    /// Each order dictionary should contain:
    /// * `imnt` - Instrument symbol (string)
    /// * `direction` - "LONG" or "SHORT"
    /// * `order_type` - "OPEN" or "CLOSE"
    /// * `size` - Order size (float)
    /// * `price` - Order price (float, optional - uses market price if not specified)
    /// * `leverage` - Leverage multiplier (int, optional - defaults to 1)
    /// * `duration` - Order duration in bars (int, optional - GTC if not specified)
    #[pyo3(signature = (strategy_func, start_time_opt=None, end_time_opt=None, lookback_periods=50))]
    pub fn run_strategy(
        &mut self,
        py: Python,
        strategy_func: PyObject,
        start_time_opt: Option<f64>, // Unix timestamp in seconds
        end_time_opt: Option<f64>,   // Unix timestamp in seconds
        lookback_periods: Option<usize>, // Number of historical periods to include
    ) -> PyResult<PyObject> {
        // Convert Python timestamps to DateTime<Utc>
        let start_datetime = start_time_opt.and_then(|ts| DateTime::from_timestamp(ts as i64, 0));

        let end_datetime = end_time_opt.and_then(|ts| DateTime::from_timestamp(ts as i64, 0));

        // Define the Rust closure that calls the Python strategy function
        let strategy = |backtest: &Backtest, status: &BacktestStatus| -> OrderTicket {
            Python::with_gil(|py| -> OrderTicket {
                // Create strategy context data
                let data_dict = PyDict::new(py);
                let _ = data_dict.set_item("current_time_index", backtest.current_time_index);
                let _ = data_dict.set_item(
                    "current_time",
                    backtest.get_cur_time().map(|t| t.timestamp()).unwrap_or(0),
                );
                let _ = data_dict.set_item("tradables", backtest.tradables.clone());
                let _ = data_dict.set_item("current_cash", status.cur_cash);
                let _ = data_dict.set_item("portfolio_value", status.cur_portfolio_mtm_value);

                // Add current prices
                let prices_dict = PyDict::new(py);
                for (instrument, price) in &status.cur_price_vector {
                    let _ = prices_dict.set_item(instrument, *price);
                }
                let _ = data_dict.set_item("current_prices", prices_dict);

                // Add available market data (configurable lookback window) - OPTIMIZED
                let lookback = lookback_periods.unwrap_or(50); // Default to 50 periods
                let available_data = match backtest.get_current_available_data(Some(lookback)) {
                    Ok(data) => data,
                    Err(_) => std::collections::HashMap::new(), // Return empty if error
                };
                let market_dict = PyDict::new(py);
                for (instrument, data_frame) in available_data {
                    // Return Polars DataFrame slice directly using PyO3-Polars integration
                    // This avoids expensive row-by-row conversion and leverages Polars' native PyO3 support
                    let py_dataframe = pyo3_polars::PyDataFrame(data_frame);
                    let _ = market_dict.set_item(instrument, py_dataframe.into_py(py));
                }
                let _ = data_dict.set_item("current_data", market_dict); // Add current_data alias for strategy compatibility

                // Add positions info
                let positions_dict = PyDict::new(py);
                for (instrument, pos_manager) in &status.cur_positions {
                    let pos_dict = PyDict::new(py);
                    if let Some(long_pos) = &pos_manager.long_position {
                        let long_dict = PyDict::new(py);
                        let _ = long_dict.set_item("size", long_pos.size);
                        let _ = long_dict.set_item("entry_price", long_pos.entry_price);
                        let _ = long_dict.set_item("open_pnl", long_pos.open_pnl);
                        let _ = pos_dict.set_item("long", long_dict);
                    }
                    if let Some(short_pos) = &pos_manager.short_position {
                        let short_dict = PyDict::new(py);
                        let _ = short_dict.set_item("size", short_pos.size);
                        let _ = short_dict.set_item("entry_price", short_pos.entry_price);
                        let _ = short_dict.set_item("open_pnl", short_pos.open_pnl);
                        let _ = pos_dict.set_item("short", short_dict);
                    }
                    let _ = positions_dict.set_item(instrument, pos_dict);
                }
                let _ = data_dict.set_item("positions", positions_dict);

                // Call Python strategy function
                match strategy_func.call1(py, (data_dict,)) {
                    Ok(result) => {
                        // Parse Python orders and create OrderTicket
                        let mut order_ticket = OrderTicket::new();

                        // Extract orders from Python result (should be a list of order dicts)
                        if let Ok(orders_list) = result.downcast::<PyList>(py) {
                            for order_item in orders_list.iter() {
                                if let Ok(order_dict) = order_item.downcast::<PyDict>() {
                                    // Parse order from dictionary
                                    if let (
                                        Ok(Some(imnt_val)),
                                        Ok(Some(direction_val)),
                                        Ok(Some(order_type_val)),
                                        Ok(Some(size_val)),
                                    ) = (
                                        order_dict.get_item("imnt"),
                                        order_dict.get_item("direction"),
                                        order_dict.get_item("order_type"),
                                        order_dict.get_item("size"),
                                    ) {
                                        if let (
                                            Ok(imnt),
                                            Ok(direction_str),
                                            Ok(order_type_str),
                                            Ok(size),
                                        ) = (
                                            imnt_val.extract::<String>(),
                                            direction_val.extract::<String>(),
                                            order_type_val.extract::<String>(),
                                            size_val.extract::<f64>(),
                                        ) {
                                            // Convert string enums to Rust enums
                                            let direction = match direction_str.as_str() {
                                                "LONG" => Direction::Long,
                                                "SHORT" => Direction::Short,
                                                _ => continue,
                                            };

                                            let order_type = match order_type_str.as_str() {
                                                "OPEN" => OrderType::Open,
                                                "CLOSE" => OrderType::Close,
                                                _ => continue,
                                            };

                                            // Extract optional fields
                                            let leverage = order_dict
                                                .get_item("leverage")
                                                .ok()
                                                .and_then(|v| {
                                                    v.and_then(|val| val.extract::<u8>().ok())
                                                })
                                                .unwrap_or(1);
                                            let price = order_dict
                                                .get_item("price")
                                                .ok()
                                                .and_then(|v| {
                                                    v.and_then(|val| val.extract::<f64>().ok())
                                                })
                                                .unwrap_or_else(|| {
                                                    // Use current market price if not specified
                                                    status
                                                        .cur_price_vector
                                                        .get(&imnt)
                                                        .copied()
                                                        .unwrap_or(0.0)
                                                });
                                            let duration =
                                                order_dict.get_item("duration").ok().and_then(
                                                    |v| v.and_then(|val| val.extract::<u32>().ok()),
                                                );

                                            // Create order
                                            if let Some(current_time) = backtest.get_cur_time() {
                                                match crate::backtesting::order::Order::new(
                                                    imnt,
                                                    price,
                                                    direction,
                                                    order_type,
                                                    size,
                                                    leverage,
                                                    current_time,
                                                    duration,
                                                ) {
                                                    Ok(order) => order_ticket.add_order(order),
                                                    Err(_) => {
                                                        // Skip invalid orders
                                                        continue;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        order_ticket
                    }
                    Err(_) => OrderTicket::new(),
                }
            })
        };

        // Run the strategy using the engine's run_strategy method
        let results = self
            .inner
            .run_strategy(start_datetime, end_datetime, strategy)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Store results for other methods
        self.last_results = Some(results.clone());

        // Convert results to Python format
        let results_dict = PyDict::new(py);
        results_dict.set_item(
            "final_portfolio_value",
            self.inner.status.cur_portfolio_mtm_value,
        )?;
        results_dict.set_item("initial_cash", self.inner.initial_cash)?;
        results_dict.set_item(
            "total_return",
            (self.inner.status.cur_portfolio_mtm_value - self.inner.initial_cash)
                / self.inner.initial_cash,
        )?;

        // Add timestamps
        let mut timestamps = Vec::new();
        for (timestamp, _) in &results {
            timestamps.push(timestamp.timestamp() as f64);
        }
        results_dict.set_item("timestamps", timestamps)?;
        results_dict.set_item("num_periods", results.len())?;

        // Add execution period info
        if let Some(start_dt) = start_datetime {
            results_dict.set_item("start_time", start_dt.timestamp() as f64)?;
        }
        if let Some(end_dt) = end_datetime {
            results_dict.set_item("end_time", end_dt.timestamp() as f64)?;
        }

        Ok(results_dict.into())
    }

    /// Calculate comprehensive performance metrics from the last backtest run.
    /// 
    /// # Arguments
    /// * `py` - Python interpreter reference
    /// 
    /// # Returns
    /// * `PyResult<PyObject>` - Dictionary containing performance metrics including:
    ///   - `total_pnl` - Total profit/loss
    ///   - `total_return_pct` - Total return percentage
    ///   - `sharpe_ratio` - Risk-adjusted return metric
    ///   - `sortino_ratio` - Downside risk-adjusted return
    ///   - `calmar_ratio` - Return vs maximum drawdown
    ///   - `max_drawdown` - Maximum peak-to-trough decline
    ///   - `volatility` - Standard deviation of returns
    ///   - `win_rate` - Percentage of profitable trades
    ///   - `total_trades` - Number of completed trades
    /// 
    /// # Errors
    /// * Returns error if no backtest has been run yet
    pub fn calculate_performance_metrics(&self, py: Python) -> PyResult<PyObject> {
        if let Some(ref results) = self.last_results {
            // Call the engine's calculate_performance_metrics method
            let metrics = self.inner.calculate_performance_metrics(results);

            // Convert to Python dict
            let metrics_dict = PyDict::new(py);
            metrics_dict.set_item("total_pnl", metrics.total_pnl)?;
            metrics_dict.set_item("total_return_pct", metrics.total_return_pct)?;
            metrics_dict.set_item("sharpe_ratio", metrics.sharpe_ratio)?;
            metrics_dict.set_item("sortino_ratio", metrics.sortino_ratio)?;
            metrics_dict.set_item("calmar_ratio", metrics.calmar_ratio)?;
            metrics_dict.set_item("max_drawdown", metrics.max_drawdown)?;
            metrics_dict.set_item("max_drawdown_pct", metrics.max_drawdown_pct)?;
            metrics_dict.set_item("volatility", metrics.volatility)?;
            metrics_dict.set_item("downside_volatility", metrics.downside_volatility)?;
            metrics_dict.set_item("cvar_5", metrics.cvar_5)?;
            metrics_dict.set_item("win_rate", metrics.win_rate)?;
            metrics_dict.set_item("total_trades", metrics.total_trades)?;
            metrics_dict.set_item("avg_trade_pnl", metrics.avg_trade_pnl)?;
            metrics_dict.set_item("profit_factor", metrics.profit_factor)?;
            
            // Add total costs for backward compatibility
            metrics_dict.set_item("total_funding_cost", metrics.total_funding_cost)?;
            metrics_dict.set_item("total_transaction_cost", metrics.total_transaction_cost)?;
            
            // Add per-instrument cost breakdowns from current engine status
            // Funding costs per instrument
            let funding_by_instrument = PyDict::new(py);
            for (instrument, cost) in &self.inner.status.cur_cum_funding_cost {
                funding_by_instrument.set_item(instrument, *cost)?;
            }
            metrics_dict.set_item("funding_cost_by_instrument", funding_by_instrument)?;
            
            // Transaction costs per instrument (calculated from trades)
            let transaction_by_instrument = PyDict::new(py);
            for (instrument, pos_manager) in &self.inner.status.cur_positions {
                let mut instrument_tx_cost = 0.0;
                
                // Calculate from closed trades
                for trade in &pos_manager.closed_positions {
                    let notional = trade.entry_price * trade.size;
                    instrument_tx_cost += notional * TRANSACTION_FEE; // TRANSACTION_FEE constant
                }
                
                // Calculate from open trades (they also paid fees when opened)
                if let Some(ref long_pos) = pos_manager.long_position {
                    if long_pos.trade_status == crate::utils::types::TradeStatus::Open {
                        let notional = long_pos.entry_price * long_pos.size;
                        instrument_tx_cost += notional * TRANSACTION_FEE; // TRANSACTION_FEE constant
                    }
                }
                if let Some(ref short_pos) = pos_manager.short_position {
                    if short_pos.trade_status == crate::utils::types::TradeStatus::Open {
                        let notional = short_pos.entry_price * short_pos.size;
                        instrument_tx_cost += notional * TRANSACTION_FEE; // TRANSACTION_FEE constant
                    }
                }
                
                transaction_by_instrument.set_item(instrument, instrument_tx_cost)?;
            }
            metrics_dict.set_item("transaction_cost_by_instrument", transaction_by_instrument)?;

            Ok(metrics_dict.into())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No backtest results available. Please run_strategy first.",
            ))
        }
    }

    pub fn get_cum_pnl_by_instrument(&self, py: Python) -> PyResult<PyObject> {
        if let Some(ref results) = self.last_results {
            // Call the engine's get_cum_pnl_by_instrument method
            let pnl_by_instrument = self.inner.get_cum_pnl_by_instrument(results);

            // Convert to Python dict
            let pnl_dict = PyDict::new(py);

            for (instrument, pnl_series) in pnl_by_instrument {
                let instrument_list = PyList::empty(py);
                for (timestamp, pnl) in pnl_series {
                    let tuple = PyTuple::new(py, &[timestamp.timestamp() as f64, pnl]);
                    instrument_list.append(tuple)?;
                }
                pnl_dict.set_item(instrument, instrument_list)?;
            }

            Ok(pnl_dict.into())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No backtest results available. Please run_strategy first.",
            ))
        }
    }

    pub fn get_trade_history(&self, py: Python) -> PyResult<PyObject> {
        if let Some(ref results) = self.last_results {
            // Call the engine's get_trade_history method
            let trade_history = self.inner.get_trade_history(results);

            // Convert to Python dict of DataFrames
            let history_dict = PyDict::new(py);

            for (instrument, trades) in trade_history {
                // Collect trades for this instrument into vectors
                let mut open_times = Vec::new();
                let mut close_times = Vec::new();
                let mut directions = Vec::new();
                let mut entry_prices = Vec::new();
                let mut leverages = Vec::new();
                let mut pnls = Vec::new();
                let mut durations = Vec::new();

                for trade in trades {
                    open_times.push(trade.open_time.format("%Y-%m-%d %H:%M").to_string());  
                    close_times.push(
                        match trade.close_time {
                            Some(close_dt) => close_dt.format("%Y-%m-%d %H:%M").to_string(),
                            None => "Open".to_string(),  // For open positions
                        }
                    );
                    directions.push(format!("{:?}", trade.direction));
                    entry_prices.push(trade.entry_price);
                    leverages.push(trade.leverage as i32);
                    pnls.push(trade.pnl);
                    durations.push(trade.duration_hours);
                }

                // Create Polars DataFrame for this instrument
                let df = df! {
                    "Open Time" => open_times,
                    "Close Time" => close_times,
                    "Direction" => directions,
                    "Entry Price" => entry_prices,
                    "Leverage" => leverages,
                    "PnL" => pnls,
                    "Duration in Hours" => durations,
                }.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create DataFrame for {}: {}", instrument, e)))?;
                
                // Convert to PyDataFrame and add to dictionary
                let py_df = pyo3_polars::PyDataFrame(df);
                history_dict.set_item(instrument, py_df.into_py(py))?;
            }

            Ok(history_dict.into())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No backtest results available. Please run_strategy first.",
            ))
        }
    }

    pub fn get_portfolio_values(&self, py: Python) -> PyResult<PyObject> {
        if let Some(ref results) = self.last_results {
            // Call the engine's get_portfolio_values method
            let portfolio_values = self.inner.get_portfolio_values(results);

            // Convert to Python list of tuples
            let values_list = PyList::empty(py);
            for (timestamp, value) in portfolio_values {
                let tuple = PyTuple::new(py, &[timestamp.timestamp() as f64, value]);
                values_list.append(tuple)?;
            }

            Ok(values_list.into())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No backtest results available. Please run_strategy first.",
            ))
        }
    }

    /// Enable profiling to measure performance bottlenecks
    pub fn enable_profiling(&mut self) {
        self.inner.enable_profiling();
    }

    /// Disable profiling
    pub fn disable_profiling(&mut self) {
        self.inner.disable_profiling();
    }

    /// Print detailed profiling summary to console
    pub fn print_profiling_summary(&self) {
        self.inner.print_profiling_summary();
    }

    /// Get profiling data as Python dictionary
    pub fn get_profiling_data(&self, py: Python) -> PyResult<PyObject> {
        let summary = self.inner.profiler.get_summary();
        let result_dict = PyDict::new(py);

        // Add operation statistics
        let operations_dict = PyDict::new(py);
        for (operation, stats) in summary.operation_stats {
            let stats_dict = PyDict::new(py);
            stats_dict.set_item("total_duration_ms", stats.total_duration.as_millis() as f64)?;
            stats_dict.set_item("avg_duration_ms", stats.avg_duration.as_millis() as f64)?;
            stats_dict.set_item("max_duration_ms", stats.max_duration.as_millis() as f64)?;
            stats_dict.set_item("min_duration_ms", stats.min_duration.as_millis() as f64)?;
            stats_dict.set_item("call_count", stats.call_count)?;
            operations_dict.set_item(operation, stats_dict)?;
        }
        result_dict.set_item("operations", operations_dict)?;

        // Add counters
        let counters_dict = PyDict::new(py);
        for (counter, value) in summary.counters {
            counters_dict.set_item(counter, value)?;
        }
        result_dict.set_item("counters", counters_dict)?;

        Ok(result_dict.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_dataframe_with_features() -> DataFrame {
        // Create a test DataFrame with both basic OHLCV and custom features
        let timestamps = vec![
            1640995200000i64, // 2022-01-01 00:00:00
            1641081600000i64, // 2022-01-02 00:00:00
            1641168000000i64, // 2022-01-03 00:00:00
        ];
        
        df! {
            "Open Time" => timestamps.clone(),
            "Open" => vec![50000.0, 50100.0, 50200.0],
            "High" => vec![51000.0, 51100.0, 51200.0],
            "Low" => vec![49000.0, 49100.0, 49200.0],
            "Close" => vec![50500.0, 50600.0, 50700.0],
            "Volume" => vec![1000.0, 1100.0, 1200.0],
            // Custom features (like your sophisticated ones)
            "MomScore_180_42_3" => vec![0.025, 0.032, 0.018],
            "Beta" => vec![1.2, 1.15, 1.25],
            "RSI_21" => vec![65.0, 72.0, 58.0],
            "ATRRatio_21" => vec![0.02, 0.025, 0.018],
        }.unwrap()
    }

    fn create_test_market_data() -> HashMap<String, DataFrame> {
        let mut market_data = HashMap::new();
        market_data.insert("BTCUSDT".to_string(), create_test_dataframe_with_features());
        market_data.insert("ETHUSDT".to_string(), create_test_dataframe_with_features());
        market_data
    }

    #[test]
    fn test_pybacktest_creation_with_features() {
        let market_data = create_test_market_data();
        let py_market_data: HashMap<String, PyDataFrame> = market_data
            .into_iter()
            .map(|(k, v)| (k, PyDataFrame(v)))
            .collect();

        // Should create successfully with feature-rich data
        let result = PyBacktest::new(py_market_data, 1000000.0, "4h", Some(3), None);
        assert!(result.is_ok());
        
        let backtest = result.unwrap();
        assert_eq!(backtest.inner.market_data.len(), 2);
        assert!(backtest.inner.market_data.contains_key("BTCUSDT"));
        assert!(backtest.inner.market_data.contains_key("ETHUSDT"));
    }

    #[test] 
    fn test_get_current_available_data_preserves_features() {
        let market_data = create_test_market_data();
        let py_market_data: HashMap<String, PyDataFrame> = market_data
            .into_iter()
            .map(|(k, v)| (k, PyDataFrame(v)))
            .collect();

        let mut backtest = PyBacktest::new(py_market_data, 1000000.0, "4h", Some(3), None).unwrap();
        backtest.inner.current_time_index = 1; // Set to second row

        // Test that get_current_available_data preserves custom features
        let available_data = backtest.inner.get_current_available_data(Some(2)).unwrap();
        
        assert!(available_data.contains_key("BTCUSDT"));
        let btc_data = &available_data["BTCUSDT"];
        
        // Verify all columns are present
        let column_names = btc_data.get_column_names();
        assert!(column_names.contains(&"Open Time"));
        assert!(column_names.contains(&"Close"));
        assert!(column_names.contains(&"MomScore_180_42_3"));
        assert!(column_names.contains(&"Beta"));
        assert!(column_names.contains(&"RSI_21"));
        assert!(column_names.contains(&"ATRRatio_21"));
        
        // Verify data integrity for custom features
        let mom_score_col = btc_data.column("MomScore_180_42_3").unwrap();
        let beta_col = btc_data.column("Beta").unwrap();
        
        assert_eq!(mom_score_col.len(), 2); // Should have 2 rows (lookback=2)
        assert_eq!(beta_col.len(), 2);
    }

    #[test]
    fn test_configurable_lookback_periods() {
        let market_data = create_test_market_data();
        let py_market_data: HashMap<String, PyDataFrame> = market_data
            .into_iter()
            .map(|(k, v)| (k, PyDataFrame(v)))
            .collect();

        let mut backtest = PyBacktest::new(py_market_data, 1000000.0, "4h", Some(3), None).unwrap();
        backtest.inner.current_time_index = 2; // Set to third row (index 2)

        // Test different lookback periods
        let lookback_1 = backtest.inner.get_current_available_data(Some(1)).unwrap();
        let lookback_2 = backtest.inner.get_current_available_data(Some(2)).unwrap();
        let lookback_3 = backtest.inner.get_current_available_data(Some(3)).unwrap();

        // Verify correct data lengths
        assert_eq!(lookback_1["BTCUSDT"].height(), 1);
        assert_eq!(lookback_2["BTCUSDT"].height(), 2);
        assert_eq!(lookback_3["BTCUSDT"].height(), 3);

        // Verify features are preserved across different lookback periods
        for data in [&lookback_1, &lookback_2, &lookback_3] {
            let columns = data["BTCUSDT"].get_column_names();
            assert!(columns.contains(&"MomScore_180_42_3"));
            assert!(columns.contains(&"Beta"));
        }
    }

    #[test]
    fn test_mixed_data_types_conversion() {
        // Test DataFrame with various data types
        let df = df! {
            "Open Time" => vec![1640995200000i64, 1641081600000i64],
            "Close" => vec![50000.0, 50100.0],
            "Volume" => vec![1000i32, 1100i32],
            "Symbol" => vec!["BTC", "BTC"],
            "Active" => vec![true, false],
            "MomScore_180_42_3" => vec![0.025, 0.032],
            "Beta" => vec![Some(1.2), None], // Test null values
        }.unwrap();

        let mut market_data = HashMap::new();
        market_data.insert("BTCUSDT".to_string(), df);
        
        let py_market_data: HashMap<String, PyDataFrame> = market_data
            .into_iter()
            .map(|(k, v)| (k, PyDataFrame(v)))
            .collect();

        // Should handle mixed data types without errors
        let result = PyBacktest::new(py_market_data, 1000000.0, "4h", Some(3), None);
        assert!(result.is_ok());
        
        let mut backtest = result.unwrap();
        backtest.inner.current_time_index = 1;
        
        // Should be able to get data with mixed types
        let available_data = backtest.inner.get_current_available_data(Some(2));
        assert!(available_data.is_ok());
        
        let data = available_data.unwrap();
        let btc_data = &data["BTCUSDT"];
        
        // Verify all columns preserved
        let columns = btc_data.get_column_names();
        assert!(columns.contains(&"Close"));
        assert!(columns.contains(&"Volume"));
        assert!(columns.contains(&"Symbol"));
        assert!(columns.contains(&"Active"));
        assert!(columns.contains(&"MomScore_180_42_3"));
        assert!(columns.contains(&"Beta"));
    }

    #[test]
    fn test_feature_extraction_edge_cases() {
        // Test edge cases for feature extraction
        let market_data = create_test_market_data();
        let py_market_data: HashMap<String, PyDataFrame> = market_data
            .into_iter()
            .map(|(k, v)| (k, PyDataFrame(v)))
            .collect();

        let mut backtest = PyBacktest::new(py_market_data, 1000000.0, "4h", Some(3), None).unwrap();
        
        // Test lookback larger than available data
        backtest.inner.current_time_index = 1;
        let large_lookback = backtest.inner.get_current_available_data(Some(100));
        assert!(large_lookback.is_ok()); // Should handle gracefully
        
        // Test lookback of 0
        let zero_lookback = backtest.inner.get_current_available_data(Some(0));
        assert!(zero_lookback.is_ok());
        
        // Test None lookback (should return all data)
        let all_data = backtest.inner.get_current_available_data(None);
        assert!(all_data.is_ok());
        let data = all_data.unwrap();
        assert_eq!(data["BTCUSDT"].height(), 2); // Should return from start to current
    }

    #[test]
    fn test_default_lookback_parameter() {
        // Test that default lookback works correctly
        let market_data = create_test_market_data();
        let py_market_data: HashMap<String, PyDataFrame> = market_data
            .into_iter()
            .map(|(k, v)| (k, PyDataFrame(v)))
            .collect();

        let backtest = PyBacktest::new(py_market_data, 1000000.0, "4h", Some(3), None).unwrap();
        
        // Verify that the default lookback (50) is reasonable
        // This is more of a design validation test
        assert_eq!(backtest.inner.tradables.len(), 2);
        assert!(backtest.inner.market_data.contains_key("BTCUSDT"));
        assert!(backtest.inner.market_data.contains_key("ETHUSDT"));
    }
}
