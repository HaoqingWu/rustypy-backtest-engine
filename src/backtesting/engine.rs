use crate::backtesting::order::{Order, OrderTicket};
use crate::backtesting::status::BacktestStatus;
use crate::utils::types::{Direction, OrderType, TradeStatus};
use crate::utils::profiler::BacktestProfiler;
use crate::{log_debug, log_error, log_info, log_warn, profile_block};
use chrono::{DateTime, Timelike, Utc};
use polars::prelude::*;
use std::collections::HashMap;

const TRANSACTION_FEE: f64 = 0.0002; // 2 BPS
const DEFAULT_FUNDING_RATE: f64 = 0.0001;
const CVAR_PERCENTILE: f64 = 0.05; // 5% CVaR
const CRYPTO_TRADING_DAYS_PER_YEAR: f64 = 365.0; // 24/7 crypto markets
const FUNDING_HOURS: [u32; 3] = [0, 8, 16]; // Funding payment hours
const MINUTES_PER_DAY: f64 = 24.0 * 60.0; // 1440 minutes per day

/// Record of a completed trade for reporting and analysis
/// 
/// This struct captures all the essential information about a trade that was
/// executed during backtesting, used for trade history export and analysis.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// The trading instrument symbol (e.g., "BTCUSDT", "ETHUSDT")
    pub instrument: String,
    
    /// When the trade was opened (position entry time)
    pub open_time: DateTime<Utc>,
    
    /// When the trade was closed (position exit time)
    /// None if the trade is still open at the end of the backtest
    pub close_time: Option<DateTime<Utc>>,
    
    /// Direction of the trade (Long or Short)
    pub direction: Direction,
    
    /// Average entry price for the position
    /// If multiple orders contributed to this position, this is the volume-weighted average
    pub entry_price: f64,
    
    /// Average exit price for the position (if closed)
    /// If multiple orders closed this position, this is the volume-weighted average
    pub exit_price: f64,
    
    /// Size of the position in base asset units
    /// For BTCUSDT, this would be the number of BTC traded
    pub size: f64,
    
    /// Leverage used for this trade (1-100)
    /// Affects margin requirements and potential returns
    pub leverage: u8,
    
    /// Final profit/loss for this trade in base currency (USD)
    /// Includes transaction costs and funding costs if applicable
    /// Positive = profit, negative = loss
    pub pnl: f64,
    
    /// Duration of the trade in hours
    /// Calculated from open_time to close_time (or current time if still open)
    pub duration_hours: f64,
}

/// Comprehensive performance metrics for backtesting results
/// 
/// This struct contains all the key performance indicators calculated after a backtest run.
/// All monetary values are in the base currency (e.g., USD for BTCUSDT trading).
/// All ratios and percentages are calculated using proper statistical methods.
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total profit/loss in base currency (final_portfolio_value - initial_cash)
    /// This is AFTER all funding costs and transaction fees have been deducted
    pub total_pnl: f64,
    
    /// Total return as a percentage (total_pnl / initial_cash)
    /// Example: 0.15 means 15% return
    pub total_return_pct: f64,
    
    /// Sharpe ratio - risk-adjusted return metric (annualized)
    /// Calculated as: (mean_daily_return / daily_volatility) * sqrt(365)
    /// Higher values indicate better risk-adjusted performance
    pub sharpe_ratio: f64,
    
    /// Sortino ratio - downside risk-adjusted return metric (annualized)
    /// Only considers negative returns in volatility calculation
    /// Generally higher than Sharpe ratio as it ignores upside volatility
    pub sortino_ratio: f64,
    
    /// Calmar ratio - return vs maximum drawdown
    /// Calculated as: annualized_return / max_drawdown_percentage
    /// Higher values indicate better return per unit of drawdown risk
    pub calmar_ratio: f64,
    
    /// Maximum drawdown in absolute currency terms (always negative)
    /// Represents the largest peak-to-trough decline in portfolio value
    pub max_drawdown: f64,
    
    /// Maximum drawdown as a percentage of peak value (always negative)
    /// Example: -0.25 means the portfolio fell 25% from its peak
    pub max_drawdown_pct: f64,
    
    /// Annualized volatility (standard deviation of daily returns)
    /// Represents the year-over-year variability of returns
    pub volatility: f64,
    
    /// Annualized downside volatility (standard deviation of negative daily returns only)
    /// Used in Sortino ratio calculation - measures only "bad" volatility
    pub downside_volatility: f64,
    
    /// Conditional Value at Risk at 5% level (CVaR-5%)
    /// Average of the worst 5% of daily returns
    /// Measures tail risk - what you can expect to lose in bad scenarios
    pub cvar_5: f64,
    
    /// Win rate - percentage of profitable trades
    /// Example: 0.6 means 60% of trades were profitable
    pub win_rate: f64,
    
    /// Total number of completed trades (both long and short)
    /// Only counts closed positions, not currently open trades
    pub total_trades: u32,
    
    /// Average profit/loss per trade
    /// Calculated as: total_pnl_from_trades / total_trades
    pub avg_trade_pnl: f64,
    
    /// Profit factor - ratio of gross profits to gross losses
    /// Values > 1.0 indicate overall profitability
    /// Example: 1.5 means total profits are 1.5x total losses
    pub profit_factor: f64,
    
    /// Total funding costs paid across all instruments and time periods
    /// Always positive (cost to the trader)
    /// Sum of all cur_cum_funding_cost values
    pub total_funding_cost: f64,
    
    /// Total transaction costs (fees) paid for all trades
    /// Always positive (cost to the trader)
    /// Calculated from all executed trades at TRANSACTION_FEE rate (0.02%)
    pub total_transaction_cost: f64,
}

/// Main backtesting engine that executes trading strategies against historical market data
/// 
/// This struct contains all the state needed to run a backtest, including market data,
/// current positions, cash balances, and configuration parameters. It operates by
/// stepping through historical data time-by-time and executing strategy decisions.
#[derive(Debug)]
pub struct Backtest {
    /// List of tradable instruments (e.g., ["BTCUSDT", "ETHUSDT"])
    /// These must match the keys in market_data HashMap
    pub tradables: Vec<String>,
    
    /// Starting cash amount for the backtest in base currency
    /// This is the initial capital available for trading
    pub initial_cash: f64,
    
    /// Current state of the backtest including positions, cash, and PnL
    /// Updated at each time step during strategy execution
    pub status: BacktestStatus,
    
    /// Historical market data for all instruments
    /// Key: instrument symbol (e.g., "BTCUSDT")
    /// Value: Polars DataFrame with OHLCV data and timestamps
    /// Must include columns: "Open Time", "Open", "High", "Low", "Close", "Volume", "fundingRate"
    pub market_data: HashMap<String, DataFrame>,
    
    /// Sorted list of all unique timestamps from market data
    /// Used to iterate through time during backtesting
    /// All instruments must have data for these timestamps
    pub available_timestamps: Vec<DateTime<Utc>>,
    
    /// Current position in the available_timestamps vector
    /// Represents "where we are" in the backtest timeline
    /// Incremented at each time step during strategy execution
    pub current_time_index: usize,
    
    /// Data frequency in minutes (e.g., 240.0 for 4-hour candles)
    /// Used for performance metric calculations and funding rate applications
    /// Common values: 1.0 (1m), 5.0 (5m), 60.0 (1h), 240.0 (4h), 1440.0 (1d)
    pub data_frequency_minutes: f64,
    
    /// Performance profiler for measuring execution times
    /// Tracks time spent in different parts of the backtesting process
    /// Useful for optimizing performance bottlenecks
    pub profiler: BacktestProfiler,
    
}

impl Backtest {
    pub fn new(market_data: HashMap<String, DataFrame>, initial_cash: f64) -> Result<Self, String> {
        Self::new_with_frequency(market_data, initial_cash, 240.0) // Default to 4H for backward compatibility
    }

    pub fn new_with_frequency(market_data: HashMap<String, DataFrame>, initial_cash: f64, data_frequency_minutes: f64) -> Result<Self, String> {
        if market_data.is_empty() {
            return Err("Market data cannot be empty".to_string());
        }

        let tradables: Vec<String> = market_data.keys().cloned().collect();

        // Extract timestamps from the first DataFrame (should be BTCUSDT by convention)
        let available_timestamps = if let Some(first_df) = market_data.values().next() {
            let timestamp_series = first_df
                .column("Open Time")
                .map_err(|e| format!("No Open Time column found: {}", e))?;

            // Convert the timestamp series to Vec<DateTime<Utc>>
            timestamp_series
                .datetime()
                .map_err(|e| format!("Failed to parse datetime column: {}", e))?
                .into_iter()
                .map(|opt_dt| {
                    opt_dt
                        .map(|dt| DateTime::from_timestamp_millis(dt).unwrap_or_default())
                        .unwrap_or_default()
                })
                .collect::<Vec<DateTime<Utc>>>()
        } else {
            return Err("No market data provided".to_string());
        };

        // Validate all DataFrames have the same length
        let expected_len = available_timestamps.len();
        for (symbol, df) in &market_data {
            if df.height() != expected_len {
                return Err(format!(
                    "Mismatch of market data lengths for instrument {}. Expected: {}, Got: {}",
                    symbol,
                    expected_len,
                    df.height()
                ));
            }
        }

        let status = BacktestStatus::new(tradables.clone(), initial_cash);

        Ok(Self {
            tradables,
            initial_cash,
            status,
            market_data: market_data,
            available_timestamps,
            current_time_index: 0,
            data_frequency_minutes,
            profiler: BacktestProfiler::new(false), // Disabled by default
        })
    }

    /// Enable profiling for performance analysis
    pub fn enable_profiling(&mut self) {
        self.profiler.enabled = true;
    }

    /// Disable profiling
    pub fn disable_profiling(&mut self) {
        self.profiler.enabled = false;
    }

    /// Get profiling summary - this will print detailed performance breakdown
    pub fn print_profiling_summary(&self) {
        self.profiler.print_summary();
    }

    pub fn get_cur_time(&self) -> Option<DateTime<Utc>> {
        self.available_timestamps
            .get(self.current_time_index)
            .cloned()
    }

    pub fn get_next_time(&self) -> Option<DateTime<Utc>> {
        self.available_timestamps
            .get(self.current_time_index + 1)
            .cloned()
    }

    fn increment_time(&mut self) {
        if self.current_time_index + 1 < self.available_timestamps.len() {
            self.current_time_index += 1;
        }
    }

    pub fn get_next_open_price(&self, symbol: &str) -> Option<f64> {
        self.get_price_at_index(symbol, self.current_time_index + 1, "Open")
    }

    pub fn get_next_high_price(&self, symbol: &str) -> Option<f64> {
        self.get_price_at_index(symbol, self.current_time_index + 1, "High")
    }

    pub fn get_next_low_price(&self, symbol: &str) -> Option<f64> {
        self.get_price_at_index(symbol, self.current_time_index + 1, "Low")
    }

    pub fn get_next_close_price(&self, symbol: &str) -> Option<f64> {
        self.get_price_at_index(symbol, self.current_time_index + 1, "Close")
    }

    pub fn get_next_volume(&self, symbol: &str) -> Option<f64> {
        self.get_price_at_index(symbol, self.current_time_index + 1, "Volume")
    }

    pub fn get_next_funding_rate(&self, symbol: &str) -> f64 {
        self.get_price_at_index(symbol, self.current_time_index + 1, "fundingRate")
            .unwrap_or(DEFAULT_FUNDING_RATE)
    }

    fn get_price_at_index(&self, symbol: &str, index: usize, column: &str) -> Option<f64> {
        let df = self.market_data.get(symbol)?;

        if index >= df.height() {
            return None;
        }

        let series = df.column(column).ok()?;

        match series.dtype() {
            DataType::Float64 => Some(series.f64().ok()?.get(index)?),
            DataType::Float32 => Some(series.f32().ok()?.get(index)? as f64),
            _ => None,
        }
    }

    pub fn get_current_available_data(
        &self,
        look_back: Option<usize>,
    ) -> Result<HashMap<String, DataFrame>, String> {
        let mut result_data = HashMap::new();
        let current_idx = self.current_time_index;
        let start_idx = if let Some(lb) = look_back {
            current_idx.saturating_sub(lb.saturating_sub(1))
        } else {
            0
        };

        for (symbol, df) in &self.market_data {
            let slice_length = (current_idx - start_idx + 1) as usize;
            if slice_length > df.height() {
                return Err(format!("Invalid slice length {} for symbol {}", slice_length, symbol));
            }
            let sliced_df = df.slice(start_idx as i64, slice_length);
            result_data.insert(symbol.clone(), sliced_df);
        }
        Ok(result_data)
    }

    fn execute_orders(&mut self, cur_time: DateTime<Utc>) {
        let mut orders_to_remove_per_imnt: HashMap<String, Vec<Order>> = HashMap::new();

        // Extract orders first to avoid borrowing conflicts
        let order_ticket_clone = if let Some(order_ticket) = &self.status.cur_order_ticket {
            order_ticket.orders.clone()
        } else {
            return;
        };

        for (imnt, orders) in order_ticket_clone {
            let (Some(next_high_price), Some(next_low_price)) = (
                self.get_next_high_price(&imnt),
                self.get_next_low_price(&imnt),
            ) else {
                continue;
            };

            for order in orders.iter() {
                if order.size <= 0.0 {
                    log_warn!("Size of the order {:?} is negative or zero.", order);
                    continue;
                }

                let should_remove = match order.order_type {
                    OrderType::Open => self.execute_open_order(
                        &order, 
                        &imnt, 
                        next_high_price, 
                        next_low_price, 
                        cur_time
                    ),
                    OrderType::Close => self.execute_close_order(
                        &order, 
                        &imnt, 
                        next_high_price, 
                        next_low_price, 
                        cur_time
                    ),
                };

                if should_remove {
                    orders_to_remove_per_imnt
                        .entry(imnt.clone())
                        .or_default()
                        .push(order.clone());
                }
            }
        }

        // Remove executed orders
        self.remove_executed_orders(orders_to_remove_per_imnt);
    }

    fn execute_open_order(
        &mut self,
        order: &Order,
        imnt: &str,
        next_high_price: f64,
        next_low_price: f64,
        cur_time: DateTime<Utc>,
    ) -> bool {
        // Check if execution price is between Low and High of next candle
        if order.price <= next_high_price && order.price >= next_low_price {
            let notional = order.notional();
            let margin = order.margin();
            let transaction_fee = notional * TRANSACTION_FEE;

            // Check if enough cash to execute
            if self.status.cur_cash >= (margin + transaction_fee) {
                match order.to_trade(cur_time) {
                    Ok(trade) => {
                        if let Some(pos_manager) = self.status.cur_positions.get_mut(imnt) {
                            pos_manager.process_new_trade(trade);
                        }

                        self.status.cur_cash -= margin + transaction_fee;
                        log_debug!("Order {:?} executed.", order.info());
                        return true;
                    }
                    Err(e) => {
                        log_error!("Failed to create trade from order: {}", e);
                        return false;
                    }
                }
            } else {
                log_info!("Not enough cash to execute the trade {:?}.", order);
            }
        }
        false
    }

    fn execute_close_order(
        &mut self,
        order: &Order,
        imnt: &str,
        next_high_price: f64,
        next_low_price: f64,
        cur_time: DateTime<Utc>,
    ) -> bool {
        // First, find the position to close without holding a mutable reference
        let position_to_close = if let Some(position_manager) = self.status.cur_positions.get(imnt) {
            self.find_position_to_close(position_manager, order)
        } else {
            None
        };
        
        if let Some((last_trade, is_long)) = position_to_close {
            return self.process_close_order(
                order, 
                imnt, 
                &last_trade, 
                is_long, 
                next_high_price, 
                next_low_price, 
                cur_time
            );
        } else {
            log_warn!("No position on {} to close!", imnt);
            return true; // Remove the order
        }
    }

    fn find_position_to_close(
        &self,
        position_manager: &crate::backtesting::position::PositionManager,
        order: &Order,
    ) -> Option<(crate::backtesting::trade::Trade, bool)> {
        // Check long position
        if let Some(long_pos) = &position_manager.long_position {
            if long_pos.trade_status == TradeStatus::Open && order.direction == Direction::Short {
                return Some((long_pos.clone(), true));
            }
        }
        
        // Check short position
        if let Some(short_pos) = &position_manager.short_position {
            if short_pos.trade_status == TradeStatus::Open && order.direction == Direction::Long {
                return Some((short_pos.clone(), false));
            }
        }
        
        None
    }

    fn process_close_order(
        &mut self,
        order: &Order,
        imnt: &str,
        last_trade: &crate::backtesting::trade::Trade,
        is_long: bool,
        next_high_price: f64,
        next_low_price: f64,
        cur_time: DateTime<Utc>,
    ) -> bool {
        // Validate close order
        if order.leverage != last_trade.leverage {
            log_warn!("Cannot close a position with different leverage.");
            return true; // Remove invalid order
        }

        let mut close_size = order.size;
        if close_size > last_trade.size {
            log_info!("{} is larger than the current position, it'll get truncated to current position.", close_size);
            close_size = last_trade.size;
        }

        // Check if execution price is valid
        if order.price <= next_high_price && order.price >= next_low_price {
            let initial_trade_margin = last_trade.margin();
            let closed_pnl = if is_long {
                (order.price - last_trade.entry_price) * close_size
            } else {
                (last_trade.entry_price - order.price) * close_size
            };
            let transaction_fee = order.price * close_size * TRANSACTION_FEE;

            // Update the position
            if let Some(position_manager) = self.status.cur_positions.get_mut(imnt) {
                if is_long {
                    if let Some(long_pos) = &mut position_manager.long_position {
                        match long_pos.close_position(order.price, close_size, cur_time) {
                            Ok(()) => {
                                // Position closed successfully
                                let new_margin = long_pos.margin();
                                self.status.cur_cash += (initial_trade_margin - new_margin) - transaction_fee + closed_pnl;
                                if long_pos.trade_status == TradeStatus::Closed {
                                    // Move the closed trade to the closed_positions vector
                                    let mut closed_trade = long_pos.clone();
                                    closed_trade.closed_pnl = closed_pnl - transaction_fee; // Set final PnL deducting fees
                                    position_manager.cum_closed_pnl += closed_trade.closed_pnl;
                                    position_manager.closed_positions.push(closed_trade);
                                    position_manager.long_position = None;
                                }
                            }
                            Err(e) => {
                                log_error!("Failed to close {} long position: {}", imnt, e);
                                return false; // don't remove order if closing failed
                            }
                        }
                    }
                } else {
                    if let Some(short_pos) = &mut position_manager.short_position {
                        match short_pos.close_position(order.price, close_size, cur_time) {
                            Ok(()) => {
                                // Position closed successfully
                                let new_margin = short_pos.margin();
                                self.status.cur_cash += (initial_trade_margin - new_margin) - transaction_fee + closed_pnl;

                                if short_pos.trade_status == TradeStatus::Closed {
                                    // Move the closed trade to the closed_positions vector
                                    let mut closed_trade = short_pos.clone();
                                    closed_trade.closed_pnl = closed_pnl - transaction_fee; // Set final PnL including fees
                                    position_manager.cum_closed_pnl += closed_trade.closed_pnl;
                                    position_manager.closed_positions.push(closed_trade);
                                    position_manager.short_position = None;
                                }
                            }
                            Err(e) => {
                                log_error!("Failed to close {} short position: {}", imnt, e);
                                return false; // Don't remove order if closing failed
                            }
                        }
                    }
                }
            }
            return true; // Remove executed order
        }
        false // Keep order if not executed
    }

    fn remove_executed_orders(&mut self, orders_to_remove_per_imnt: HashMap<String, Vec<Order>>) {
        if let Some(order_ticket) = self.status.cur_order_ticket.as_mut() {
            for (_imnt, orders_to_remove) in orders_to_remove_per_imnt {
                for order_to_remove in orders_to_remove {
                    order_ticket.remove_order(&order_to_remove);
                }
            }
        }
    }

    pub fn run_strategy<F>(
        &mut self,
        start_time_opt: Option<DateTime<Utc>>,
        end_time_opt: Option<DateTime<Utc>>,
        mut strategy: F,
    ) -> Result<HashMap<DateTime<Utc>, BacktestStatus>, String>
    where
        F: FnMut(&Self, &BacktestStatus) -> OrderTicket,
    {
        let total_timer = self.profiler.start_timer("total_backtest_runtime");
        let start_idx = match start_time_opt {
            Some(st) => self
                .available_timestamps
                .iter()
                .position(|&t| t >= st)
                .unwrap_or(0),
            None => 0,
        };
        let end_idx = match end_time_opt {
            Some(et) => self
                .available_timestamps
                .iter()
                .position(|&t| t > et)
                .map_or(self.available_timestamps.len() - 1, |p| {
                    if p > 0 {
                        p - 1
                    } else {
                        0
                    }
                }),
            None => self.available_timestamps.len() - 1,
        };

        if start_idx > end_idx || self.available_timestamps.is_empty() {
            return Ok(HashMap::new());
        }

        self.current_time_index = start_idx;

        // Initialize cur_price_vector in status
        for imnt in &self.tradables {
            if let Some(close_price) =
                self.get_price_at_index(imnt, self.current_time_index, "Close")
            {
                self.status
                    .cur_price_vector
                    .insert(imnt.clone(), close_price);
            }
        }

        let mut status_history = HashMap::new();

        while self.current_time_index <= end_idx {
            let cur_time = self.available_timestamps[self.current_time_index];
            self.profiler.increment_counter("backtest_iterations");

            // 1. Generate orders from strategy
            let new_order_ticket = profile_block!(&mut self.profiler, "strategy_execution", {
                strategy(self, &self.status)
            });

            // 2. Update master order ticket
            profile_block!(&mut self.profiler, "order_ticket_management", {
                if let Some(master_ticket) = self.status.cur_order_ticket.as_mut() {
                    master_ticket.update_orders();
                    master_ticket.aggregate_orders(new_order_ticket);
                } else {
                    self.status.cur_order_ticket = Some(new_order_ticket);
                    if let Some(master_ticket) = self.status.cur_order_ticket.as_mut() {
                        master_ticket.update_orders();
                    }
                }
            });

            // 3. Execute orders
            if self.current_time_index + 1 < self.available_timestamps.len() {
                profile_block!(&mut self.profiler, "order_execution", {
                    self.execute_orders(cur_time);
                });
            }

            // 4. Pay funding cost (at specific hours and minute 0)
            if FUNDING_HOURS.contains(&cur_time.hour()) && cur_time.minute() == 0 {
                profile_block!(&mut self.profiler, "funding_cost_update", {
                    self.update_funding_cost();
                });
            }

            // 5. Update status
            profile_block!(&mut self.profiler, "status_update", {
                self.update_status_after_iteration();
            });

            // 6. Store status snapshot
            profile_block!(&mut self.profiler, "status_snapshot", {
                status_history.insert(cur_time, self.status.clone());
            });

            // 7. Check for liquidation
            if profile_block!(&mut self.profiler, "liquidation_check", {
                self.check_liquidation()
            }) {
                total_timer.finish(&mut self.profiler);
                return Err(format!("Portfolio liquidated at {}", cur_time));
            }

            // 8. Increment time
            if self.current_time_index == end_idx {
                break;
            }
            self.increment_time();
        }
        
        total_timer.finish(&mut self.profiler);
        Ok(status_history)
    }

    fn check_liquidation(&self) -> bool {
        let mut total_open_pnl = 0.0;
        for position_manager in self.status.cur_positions.values() {
            if let Some(long_trade) = &position_manager.long_position {
                if long_trade.trade_status == TradeStatus::Open {
                    total_open_pnl += long_trade.open_pnl;
                }
            }
            if let Some(short_trade) = &position_manager.short_position {
                if short_trade.trade_status == TradeStatus::Open {
                    total_open_pnl += short_trade.open_pnl;
                }
            }
        }
        self.status.cur_cash + self.status.cur_total_margin + total_open_pnl < 0.0
    }

    fn update_funding_cost(&mut self) {
        // Use iterator directly without collecting
        for imnt in &self.tradables {
            let cur_price = if let Some(&price) = self.status.cur_price_vector.get(imnt) {
                price
            } else {
                continue; // Skip instruments without price data
            };
            let funding_rate = self.get_next_funding_rate(imnt);

            let cur_cash = self.status.cur_cash;

            if let Some(position_manager) = self.status.cur_positions.get_mut(imnt) {
                // Handle long position
                if let Some(long_pos) = &mut position_manager.long_position {
                    if long_pos.trade_status == TradeStatus::Open {
                        let notional = long_pos.entry_price * long_pos.size;

                        if funding_rate > 0.0 {
                            // Pay funding cost
                            if self.status.cur_cash > (notional * funding_rate).abs() {
                                self.status.cur_cash -= notional * funding_rate;
                            } else {
                                // Not enough cash, reduce position size
                                long_pos.size -= (notional * funding_rate) / cur_price;
                            }
                        } else {
                            // Receive funding payment
                            self.status.cur_cash -= notional * funding_rate; // funding_rate is negative, so this adds
                        }
                    }
                }

                // Handle short position
                if let Some(short_pos) = &mut position_manager.short_position {
                    if short_pos.trade_status == TradeStatus::Open {
                        let notional = short_pos.entry_price * short_pos.size;

                        if funding_rate < 0.0 {
                            // Pay funding cost (funding rate is negative for shorts)
                            if self.status.cur_cash > (notional * funding_rate).abs() {
                                self.status.cur_cash += notional * funding_rate;
                            // funding_rate is negative
                            } else {
                                // Not enough cash, reduce position size
                                short_pos.size += (notional * funding_rate) / cur_price;
                            }
                        } else {
                            // Receive funding payment
                            self.status.cur_cash += notional * funding_rate;
                        }
                    }
                }
            }

            // Update cumulative funding cost
            let funding_cost_paid = cur_cash - self.status.cur_cash;
            *self.status.cur_cum_funding_cost.entry(imnt.clone()).or_insert(0.0) += funding_cost_paid;
        }
    }

    fn update_status_after_iteration(&mut self) {
        // Update current prices
        for imnt in &self.tradables {
            if let Some(close_price) =
                self.get_price_at_index(imnt, self.current_time_index, "Close")
            {
                self.status
                    .cur_price_vector
                    .insert(imnt.clone(), close_price);
            }
        }

        // Update PnL and portfolio value
        let mut total_open_pnl_portfolio = 0.0;

        for (imnt, pos_manager) in self.status.cur_positions.iter_mut() {
            let current_price = *self.status.cur_price_vector.get(imnt).unwrap_or(&0.0);
            pos_manager.update_position_unrealized_pnl(current_price);

            let mut instrument_open_pnl = 0.0;
            if let Some(trade) = &pos_manager.long_position {
                if trade.trade_status == TradeStatus::Open {
                    instrument_open_pnl += trade.open_pnl;
                }
            }
            if let Some(trade) = &pos_manager.short_position {
                if trade.trade_status == TradeStatus::Open {
                    instrument_open_pnl += trade.open_pnl;
                }
            }
            self.status
                .cur_open_pnl_vector
                .insert(imnt.clone(), instrument_open_pnl);
            self.status.cur_cum_pnl.insert(
                imnt.clone(),
                instrument_open_pnl + pos_manager.cum_closed_pnl,
            );
            total_open_pnl_portfolio += instrument_open_pnl;
        }

        self.status.update_margin_and_notional();
        self.status.cur_portfolio_mtm_value =
            self.status.cur_cash + self.status.cur_total_margin + total_open_pnl_portfolio;

        // Update price vector for next iteration
        for imnt in &self.tradables {
            if let Some(next_close_price) = self.get_next_close_price(imnt) {
                self.status
                    .cur_price_vector
                    .insert(imnt.clone(), next_close_price);
            } else if self.current_time_index + 1 >= self.available_timestamps.len() {
                if let Some(last_close) =
                    self.get_price_at_index(imnt, self.current_time_index, "Close")
                {
                    self.status
                        .cur_price_vector
                        .insert(imnt.clone(), last_close);
                }
            }
        }
    }

    /// Convert portfolio values to daily frequency regardless of input timeframe
    fn get_daily_portfolio_values(
        &self,
        results: &HashMap<DateTime<Utc>, BacktestStatus>,
    ) -> Vec<(DateTime<Utc>, f64)> {
        if results.is_empty() {
            return Vec::new();
        }

        // Get all portfolio values sorted by time
        let mut all_values: Vec<(DateTime<Utc>, f64)> = results
            .iter()
            .map(|(datetime, status)| (*datetime, status.cur_portfolio_mtm_value))
            .collect();
        all_values.sort_by_key(|(datetime, _)| *datetime);

        // If data is already daily or less frequent, return as-is
        if self.data_frequency_minutes >= MINUTES_PER_DAY {
            return all_values;
        }

        // Group by date and take the last value for each day (end-of-day portfolio value)
        let mut daily_values: HashMap<chrono::NaiveDate, (DateTime<Utc>, f64)> = HashMap::new();
        
        for (datetime, value) in all_values {
            let date = datetime.date_naive();
            // Always keep the latest value for each day
            daily_values.insert(date, (datetime, value));
        }

        // Convert back to sorted vector
        let mut result: Vec<(DateTime<Utc>, f64)> = daily_values.into_values().collect();
        result.sort_by_key(|(datetime, _)| *datetime);
        result
    }

    /// Calculate daily returns from daily portfolio values
    fn calculate_daily_returns(&self, daily_values: &[(DateTime<Utc>, f64)]) -> Vec<f64> {
        if daily_values.len() < 2 {
            return Vec::new();
        }

        daily_values
            .windows(2)
            .map(|w| (w[1].1 - w[0].1) / w[0].1)
            .collect()
    }

    /// Get the annualization factor for crypto markets (24/7 trading)
    fn get_crypto_annualization_factor() -> f64 {
        CRYPTO_TRADING_DAYS_PER_YEAR.sqrt()
    }

    /// Calculate comprehensive performance metrics from backtest results
    /// Uses daily portfolio values and crypto-appropriate annualization (24/7 trading)
    pub fn calculate_performance_metrics(
        &self,
        results: &HashMap<DateTime<Utc>, BacktestStatus>,
    ) -> PerformanceMetrics {
        if results.is_empty() {
            return PerformanceMetrics::default();
        }

        // Convert to daily portfolio values for proper risk metrics calculation
        let daily_values = self.get_daily_portfolio_values(results);
        
        if daily_values.len() < 2 {
            return PerformanceMetrics::default();
        }

        // Calculate daily returns (this is the key fix)
        let daily_returns = self.calculate_daily_returns(&daily_values);
        
        if daily_returns.is_empty() {
            return PerformanceMetrics::default();
        }

        // Total P&L calculations using the final portfolio value
        let total_pnl = daily_values[daily_values.len() - 1].1 - self.initial_cash;
        let total_return_pct = total_pnl / self.initial_cash;

        // Basic statistics on DAILY returns
        let mean_daily_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance = daily_returns
            .iter()
            .map(|r| (r - mean_daily_return).powi(2))
            .sum::<f64>()
            / (daily_returns.len() - 1) as f64;
        let daily_volatility = variance.sqrt();

        // Downside deviation (for Sortino ratio) - only negative daily returns
        let negative_daily_returns: Vec<f64> = daily_returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        let downside_variance = if !negative_daily_returns.is_empty() {
            negative_daily_returns.iter().map(|r| r.powi(2)).sum::<f64>() / negative_daily_returns.len() as f64
        } else {
            0.0
        };
        let daily_downside_volatility = downside_variance.sqrt();

        // Risk-adjusted returns (annualized using crypto factor: sqrt(365))
        let crypto_annualization_factor = Self::get_crypto_annualization_factor();
        
        let sharpe_ratio = if daily_volatility > 0.0 {
            mean_daily_return / daily_volatility * crypto_annualization_factor
        } else {
            0.0
        };

        let sortino_ratio = if daily_downside_volatility > 0.0 {
            mean_daily_return / daily_downside_volatility * crypto_annualization_factor
        } else {
            0.0
        };

        // Maximum drawdown calculated on full intraday portfolio values for accuracy
        let intraday_values = self.get_portfolio_values(results);
        let portfolio_values: Vec<f64> = intraday_values.iter().map(|(_, value)| *value).collect();
        let mut peak = portfolio_values[0];
        let mut max_dd = 0.0;
        let mut max_dd_pct = 0.0;

        for &value in &portfolio_values {
            if value > peak {
                peak = value;
            }
            let dd = peak - value;
            let dd_pct = dd / peak;

            if dd > max_dd {
                max_dd = dd;
            }
            if dd_pct > max_dd_pct {
                max_dd_pct = dd_pct;
            }
        }

        // Calmar ratio (annualized return / max drawdown)
        let annualized_return = mean_daily_return * CRYPTO_TRADING_DAYS_PER_YEAR;
        let calmar_ratio = if max_dd_pct > 0.0 {
            annualized_return / max_dd_pct
        } else {
            0.0
        };

        // CVaR (worst daily returns based on percentile)
        let mut sorted_daily_returns = daily_returns.clone();
        sorted_daily_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cvar_index = (daily_returns.len() as f64 * CVAR_PERCENTILE).ceil() as usize;
        let cvar_5 = if cvar_index > 0 && cvar_index <= sorted_daily_returns.len() {
            sorted_daily_returns[..cvar_index].iter().sum::<f64>() / cvar_index as f64
        } else {
            0.0
        };

        // Trade statistics - use trade history to get accurate count
        let mut total_trades = 0u32;
        let mut winning_trades = 0u32;
        let mut total_trade_pnl = 0.0;
        let mut positive_pnl = 0.0;
        let mut negative_pnl = 0.0;

        // Use trade history to get accurate trade count (only closed trades)
        let trade_history = self.get_trade_history(results);
        for (_instrument, trades) in trade_history {
            for trade_record in trades {
                total_trades += 1;
                total_trade_pnl += trade_record.pnl;

                if trade_record.pnl > 0.0 {
                    winning_trades += 1;
                    positive_pnl += trade_record.pnl;
                } else {
                    negative_pnl += trade_record.pnl.abs();
                }
            }
        }

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let avg_trade_pnl = if total_trades > 0 {
            total_trade_pnl / total_trades as f64
        } else {
            0.0
        };

        let profit_factor = if negative_pnl > 0.0 {
            positive_pnl / negative_pnl
        } else if positive_pnl > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Calculate total funding and transaction costs from final status
        // Use the current status since it contains the final state
        let total_funding_cost: f64 = self.status.cur_cum_funding_cost.values().sum();
        
        // For transaction costs, we need to calculate from all completed trades
        // Since transaction costs are applied immediately when trades are executed
        let total_transaction_cost: f64 = {
            // Calculate transaction costs from trade history
            let mut total_tx_cost = 0.0;
            for pos_manager in self.status.cur_positions.values() {
                // Add transaction costs from closed trades
                for trade in &pos_manager.closed_positions {
                    let notional = trade.entry_price * trade.size;
                    total_tx_cost += notional * TRANSACTION_FEE;
                }
                
                // Add transaction costs from open trades (they also paid fees when opened)
                if let Some(ref long_pos) = pos_manager.long_position {
                    if long_pos.trade_status == TradeStatus::Open {
                        let notional = long_pos.entry_price * long_pos.size;
                        total_tx_cost += notional * TRANSACTION_FEE;
                    }
                }
                if let Some(ref short_pos) = pos_manager.short_position {
                    if short_pos.trade_status == TradeStatus::Open {
                        let notional = short_pos.entry_price * short_pos.size;
                        total_tx_cost += notional * TRANSACTION_FEE;
                    }
                }
            }
            total_tx_cost
        };

        PerformanceMetrics {
            total_pnl,
            total_return_pct,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown: -max_dd, // Negative for conventional representation
            max_drawdown_pct: -max_dd_pct,
            volatility: daily_volatility * crypto_annualization_factor, // Annualized daily volatility
            downside_volatility: daily_downside_volatility * crypto_annualization_factor, // Annualized daily downside volatility
            cvar_5,
            win_rate,
            total_trades,
            avg_trade_pnl,
            profit_factor,
            total_funding_cost,
            total_transaction_cost,
        }
    }

    /// Get portfolio values time series for plotting
    pub fn get_portfolio_values(
        &self,
        results: &HashMap<DateTime<Utc>, BacktestStatus>,
    ) -> Vec<(DateTime<Utc>, f64)> {
        let mut values: Vec<(DateTime<Utc>, f64)> = results
            .iter()
            .map(|(datetime, status)| (*datetime, status.cur_portfolio_mtm_value))
            .collect();

        values.sort_by_key(|(datetime, _)| *datetime);
        values
    }

    /// Get daily PnL by instrument
    pub fn get_cum_pnl_by_instrument(
        &self,
        results: &HashMap<DateTime<Utc>, BacktestStatus>,
    ) -> HashMap<String, Vec<(DateTime<Utc>, f64)>> {
        let mut pnl_by_instrument: HashMap<String, Vec<(DateTime<Utc>, f64)>> = HashMap::new();

        for imnt in &self.tradables {
            let mut instrument_pnl: Vec<(DateTime<Utc>, f64)> = results
                .iter()
                .map(|(datetime, status)| {
                    let pnl = status.cur_cum_pnl.get(imnt).unwrap_or(&0.0);
                    (*datetime, *pnl)
                })
                .collect();

            instrument_pnl.sort_by_key(|(datetime, _)| *datetime);
            pnl_by_instrument.insert(imnt.clone(), instrument_pnl);
        }

        pnl_by_instrument
    }

    /// Get trade history for analysis
    /// 
    /// # Arguments
    /// * `results` - Backtest results containing position history
    /// 
    /// # Returns
    /// HashMap mapping instrument names to vectors of trade records
    pub fn get_trade_history(
        &self,
        results: &HashMap<DateTime<Utc>, BacktestStatus>,
    ) -> HashMap<String, Vec<TradeRecord>> {
        let mut trade_history: HashMap<String, Vec<TradeRecord>> = HashMap::new();

        for imnt in &self.tradables {
            trade_history.insert(imnt.clone(), Vec::new());
        }

        for status in results.values() {
            for (imnt, pos_manager) in &status.cur_positions {
                for position in &pos_manager.closed_positions {
                    // Calculate exit price from PnL if available
                    let exit_price = if position.closed_pnl != 0.0 && position.size > 0.0 {
                        match position.direction {
                            Direction::Long => position.entry_price + (position.closed_pnl / position.size),
                            Direction::Short => position.entry_price - (position.closed_pnl / position.size),
                        }
                    } else {
                        position.entry_price // Fallback to entry price
                    };

                    // Calculate duration in hours
                    let duration_hours = if let Some(close_time) = position.close_time {
                        (close_time.timestamp() - position.filled_time.timestamp()) as f64 / 3600.0
                    } else {
                        0.0
                    };

                    let trade_record = TradeRecord {
                        instrument: imnt.clone(),
                        open_time: position.filled_time,
                        close_time: position.close_time,
                        direction: position.direction.clone(),
                        entry_price: position.entry_price,
                        exit_price,
                        size: position.size,
                        leverage: position.leverage,
                        pnl: position.closed_pnl,
                        duration_hours,
                    };

                    if let Some(trades) = trade_history.get_mut(imnt) {
                        if !trades.iter().any(|t| {
                            t.open_time == trade_record.open_time && t.pnl == trade_record.pnl
                        }) {
                            trades.push(trade_record);
                        }
                    }
                }
            }
        }

        // Add open positions to trade history (from current status)
        for (imnt, pos_manager) in &self.status.cur_positions {
            // Add open long position
            if let Some(long_pos) = &pos_manager.long_position {
                if long_pos.trade_status == TradeStatus::Open {
                    let trade_record = TradeRecord {
                        instrument: imnt.clone(),
                        open_time: long_pos.filled_time,
                        close_time: None, // Same as open time for open positions
                        direction: long_pos.direction.clone(),
                        entry_price: long_pos.entry_price,
                        exit_price: long_pos.entry_price, // Current price for open positions (placeholder)
                        size: long_pos.size,
                        leverage: long_pos.leverage,
                        pnl: long_pos.open_pnl, // Use unrealized PnL
                        duration_hours: 0.0, // Open positions don't have duration yet
                    };

                    if let Some(trades) = trade_history.get_mut(imnt) {
                        trades.push(trade_record);
                    }
                }
            }

            // Add open short position
            if let Some(short_pos) = &pos_manager.short_position {
                if short_pos.trade_status == TradeStatus::Open {
                    let trade_record = TradeRecord {
                        instrument: imnt.clone(),
                        open_time: short_pos.filled_time,
                        close_time: None, // Same as open time for open positions
                        direction: short_pos.direction.clone(),
                        entry_price: short_pos.entry_price,
                        exit_price: short_pos.entry_price, // Current price for open positions (placeholder)
                        size: short_pos.size,
                        leverage: short_pos.leverage,
                        pnl: short_pos.open_pnl, // Use unrealized PnL
                        duration_hours: 0.0, // Open positions don't have duration yet
                    };

                    if let Some(trades) = trade_history.get_mut(imnt) {
                        trades.push(trade_record);
                    }
                }
            }
        }

        trade_history
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtesting::order::{Order, OrderTicket};
    use crate::utils::types::{Direction, OrderType};
    use chrono::{DateTime, Utc};
    use std::collections::HashMap;

    fn create_test_market_data() -> HashMap<String, DataFrame> {
        let mut market_data = HashMap::new();

        // Create datetime timestamps as milliseconds since epoch (like CSV data would have)
        let timestamps = vec![
            1609459200000i64, // 2021-01-01 00:00:00
            1609473600000i64, // 2021-01-01 04:00:00
            1609488000000i64, // 2021-01-01 08:00:00
            1609502400000i64, // 2021-01-01 12:00:00
            1609516800000i64, // 2021-01-01 16:00:00
        ];

        let datetime_series = Series::new("Open Time", &timestamps)
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();

        // Create test data for BTCUSDT with actual column names
        let btc_data = df! {
            "Open Time" => datetime_series.clone(),
            "Open" => [29000.0, 29100.0, 29200.0, 29300.0, 29400.0],
            "High" => [29500.0, 29600.0, 29700.0, 29800.0, 29900.0],
            "Low" => [28800.0, 28900.0, 29000.0, 29100.0, 29200.0],
            "Close" => [29100.0, 29200.0, 29300.0, 29400.0, 29500.0],
            "Volume" => [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            "Number of Trades" => [500, 550, 600, 650, 700],
            "Taker Buy Base" => [600.0, 660.0, 720.0, 780.0, 840.0],
            "Taker_Avg_Cost" => [29050.0, 29150.0, 29250.0, 29350.0, 29450.0],
            "fundingRate" => [0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        }
        .unwrap();
        market_data.insert("BTCUSDT".to_string(), btc_data);

        // Create test data for ETHUSDT with actual column names
        let eth_data = df! {
            "Open Time" => datetime_series,
            "Open" => [730.0, 735.0, 740.0, 745.0, 750.0],
            "High" => [740.0, 745.0, 750.0, 755.0, 760.0],
            "Low" => [720.0, 725.0, 730.0, 735.0, 740.0],
            "Close" => [735.0, 740.0, 745.0, 750.0, 755.0],
            "Volume" => [2000.0, 2100.0, 2200.0, 2300.0, 2400.0],
            "Number of Trades" => [800, 850, 900, 950, 1000],
            "Taker Buy Base" => [1200.0, 1260.0, 1320.0, 1380.0, 1440.0],
            "Taker_Avg_Cost" => [732.5, 737.5, 742.5, 747.5, 752.5],
            "fundingRate" => [0.0001, -0.0001, 0.0001, 0.0001, -0.0001],
        }
        .unwrap();
        market_data.insert("ETHUSDT".to_string(), eth_data);

        market_data
    }

    fn create_test_backtest() -> Backtest {
        let market_data = create_test_market_data();
        Backtest::new(market_data, 1000000.0).unwrap()
    }

    #[test]
    fn test_backtest_creation() {
        let market_data = create_test_market_data();
        let initial_cash = 1000000.0;
        let backtest = Backtest::new(market_data, initial_cash);

        assert!(backtest.is_ok());
        let backtest = backtest.unwrap();
        assert_eq!(backtest.initial_cash, initial_cash);
        assert_eq!(backtest.tradables.len(), 2);
        assert!(backtest.tradables.contains(&"BTCUSDT".to_string()));
        assert!(backtest.tradables.contains(&"ETHUSDT".to_string()));
        assert_eq!(backtest.current_time_index, 0);
        assert_eq!(backtest.available_timestamps.len(), 5);
    }

    #[test]
    fn test_backtest_empty_data() {
        let empty_data = HashMap::new();
        let result = Backtest::new(empty_data, 1000000.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Market data cannot be empty");
    }

    #[test]
    fn test_backtest_mismatched_lengths() {
        let mut market_data = HashMap::new();

        // Create datetime series for BTC (2 entries)
        let btc_timestamps = vec![1609459200000i64, 1609473600000i64];
        let btc_datetime_series = Series::new("Open Time", &btc_timestamps)
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();

        let btc_data = df! {
            "Open Time" => btc_datetime_series,
            "Open" => [29000.0, 29100.0],
            "High" => [29500.0, 29600.0],
            "Low" => [28800.0, 28900.0],
            "Close" => [29100.0, 29200.0],
            "Volume" => [1000.0, 1100.0],
            "Number of Trades" => [500, 550],
            "Taker Buy Base" => [600.0, 660.0],
            "Taker_Avg_Cost" => [29050.0, 29150.0],
            "fundingRate" => [0.0001, 0.0001],
        }
        .unwrap();

        // Create datetime series for ETH (1 entry - mismatched length)
        let eth_timestamps = vec![1609459200000i64];
        let eth_datetime_series = Series::new("Open Time", &eth_timestamps)
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();

        let eth_data = df! {
            "Open Time" => eth_datetime_series,
            "Open" => [730.0],
            "High" => [740.0],
            "Low" => [720.0],
            "Close" => [735.0],
            "Volume" => [2000.0],
            "Number of Trades" => [800],
            "Taker Buy Base" => [1200.0],
            "Taker_Avg_Cost" => [732.5],
            "fundingRate" => [0.0001],
        }
        .unwrap();

        market_data.insert("BTCUSDT".to_string(), btc_data);
        market_data.insert("ETHUSDT".to_string(), eth_data);

        let result = Backtest::new(market_data, 1000000.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Mismatch of market data lengths"));
    }

    #[test]
    fn test_get_cur_time() {
        let backtest = create_test_backtest();
        let cur_time = backtest.get_cur_time();
        assert!(cur_time.is_some());
        // Should be the first timestamp: 2021-01-01 00:00:00
        assert_eq!(cur_time.unwrap().timestamp_millis(), 1609459200000i64);
    }

    #[test]
    fn test_get_next_time() {
        let backtest = create_test_backtest();
        let next_time = backtest.get_next_time();
        assert!(next_time.is_some());
        // Should be the second timestamp: 2021-01-01 04:00:00
        assert_eq!(next_time.unwrap().timestamp_millis(), 1609473600000i64);
    }

    #[test]
    fn test_increment_time() {
        let mut backtest = create_test_backtest();
        let initial_index = backtest.current_time_index;
        backtest.increment_time();
        assert_eq!(backtest.current_time_index, initial_index + 1);
    }

    #[test]
    fn test_get_price_methods() {
        let backtest = create_test_backtest();

        // Test getting prices for BTCUSDT using correct column names
        let next_open = backtest.get_next_open_price("BTCUSDT");
        assert!(next_open.is_some());
        assert_eq!(next_open.unwrap(), 29100.0);

        let next_high = backtest.get_next_high_price("BTCUSDT");
        assert!(next_high.is_some());
        assert_eq!(next_high.unwrap(), 29600.0);

        let next_low = backtest.get_next_low_price("BTCUSDT");
        assert!(next_low.is_some());
        assert_eq!(next_low.unwrap(), 28900.0);

        let next_close = backtest.get_next_close_price("BTCUSDT");
        assert!(next_close.is_some());
        assert_eq!(next_close.unwrap(), 29200.0);

        let next_volume = backtest.get_next_volume("BTCUSDT");
        assert!(next_volume.is_some());
        assert_eq!(next_volume.unwrap(), 1100.0);
    }

    #[test]
    fn test_get_funding_rate() {
        let backtest = create_test_backtest();

        let funding_rate = backtest.get_next_funding_rate("BTCUSDT");
        assert_eq!(funding_rate, 0.0001);

        // Test with non-existent symbol should return default
        let default_rate = backtest.get_next_funding_rate("NONEXISTENT");
        assert_eq!(default_rate, DEFAULT_FUNDING_RATE);
    }

    #[test]
    fn test_get_price_at_index() {
        let backtest = create_test_backtest();

        // Test valid index and column with correct column names
        let price = backtest.get_price_at_index("BTCUSDT", 1, "Open");
        assert!(price.is_some());
        assert_eq!(price.unwrap(), 29100.0);

        // Test invalid index (out of bounds)
        let price = backtest.get_price_at_index("BTCUSDT", 10, "Open");
        assert!(price.is_none());

        // Test invalid symbol
        let price = backtest.get_price_at_index("NONEXISTENT", 1, "Open");
        assert!(price.is_none());

        // Test invalid column
        let price = backtest.get_price_at_index("BTCUSDT", 1, "nonexistent");
        assert!(price.is_none());
    }

    #[test]
    fn test_get_current_available_data() {
        let mut backtest = create_test_backtest();
        backtest.current_time_index = 2; // Set to index 2

        // Test without lookback
        let data = backtest.get_current_available_data(None).unwrap();
        assert_eq!(data.len(), 2); // Should have BTCUSDT and ETHUSDT
        assert!(data.contains_key("BTCUSDT"));
        assert!(data.contains_key("ETHUSDT"));

        // Test with lookback
        let data_with_lookback = backtest.get_current_available_data(Some(2)).unwrap();
        assert_eq!(data_with_lookback.len(), 2);
        // Should contain data from index 1 to 2 (2 rows)
        assert_eq!(data_with_lookback["BTCUSDT"].height(), 2);
    }

    #[test]
    fn test_check_liquidation() {
        let mut backtest = create_test_backtest();

        // Initially should not be liquidated
        assert!(!backtest.check_liquidation());

        // Set negative cash and large negative PnL to trigger liquidation
        backtest.status.cur_cash = -100000.0;
        backtest.status.cur_total_margin = 50000.0;

        // Add a position with large negative PnL
        let mut position_manager =
            crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        let mut trade = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            Utc::now(),
            30000.0,
            Direction::Long,
            1.0,
            5,
        )
        .unwrap();
        trade.open_pnl = -60000.0; // Large negative PnL
        position_manager.long_position = Some(trade);
        backtest
            .status
            .cur_positions
            .insert("BTCUSDT".to_string(), position_manager);

        // Should now be liquidated
        assert!(backtest.check_liquidation());
    }

    #[test]
    fn test_update_funding_cost() {
        let mut backtest = create_test_backtest();

        // Add a long position
        let mut position_manager =
            crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        let trade = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            Utc::now(),
            29000.0,
            Direction::Long,
            1.0,
            5,
        )
        .unwrap();
        position_manager.long_position = Some(trade);
        backtest
            .status
            .cur_positions
            .insert("BTCUSDT".to_string(), position_manager);
        backtest
            .status
            .cur_price_vector
            .insert("BTCUSDT".to_string(), 29100.0);

        let initial_cash = backtest.status.cur_cash;
        backtest.update_funding_cost();

        // With positive funding rate (0.0001), long position should pay funding
        let expected_funding_cost = 29000.0 * 1.0 * 0.0001;
        let expected_cash = initial_cash - expected_funding_cost;
        assert!((backtest.status.cur_cash - expected_cash).abs() < 1e-10);

        // Check that cumulative funding cost is tracked
        let funding_cost = backtest.status.cur_cum_funding_cost.get("BTCUSDT").unwrap();
        assert!((*funding_cost - expected_funding_cost).abs() < 1e-10);
    }

    #[test]
    fn test_execute_orders_open_position() {
        let mut backtest = create_test_backtest();

        // Create an order to open a position
        let order = Order::new(
            "BTCUSDT".to_string(),
            29300.0, // Within High/Low range of next candle
            Direction::Long,
            OrderType::Open,
            1.0,
            5,
            Utc::now(),
            Some(1),
        ).unwrap();

        let mut order_ticket = OrderTicket::new();
        order_ticket.add_order(order);
        backtest.status.cur_order_ticket = Some(order_ticket);

        let initial_cash = backtest.status.cur_cash;
        backtest.execute_orders(Utc::now());

        // Check that position was created
        let position_manager = backtest.status.cur_positions.get("BTCUSDT").unwrap();
        assert!(position_manager.long_position.is_some());

        let trade = position_manager.long_position.as_ref().unwrap();
        assert_eq!(trade.entry_price, 29300.0);
        assert_eq!(trade.size, 1.0);
        assert_eq!(trade.leverage, 5);

        // Check that cash was reduced (margin + transaction fee)
        let expected_margin = 29300.0 / 5.0;
        let expected_fee = 29300.0 * TRANSACTION_FEE;
        let expected_cash = initial_cash - expected_margin - expected_fee;
        assert_eq!(backtest.status.cur_cash, expected_cash);
    }

    #[test]
    fn test_execute_orders_insufficient_cash() {
        let mut backtest = create_test_backtest();
        backtest.status.cur_cash = 100.0; // Very low cash

        // Create an expensive order
        let order = Order::new(
            "BTCUSDT".to_string(),
            29300.0,
            Direction::Long,
            OrderType::Open,
            10.0, // Large size
            1,    // Low leverage means high margin requirement
            Utc::now(),
            Some(1),
        ).unwrap();

        let mut order_ticket = OrderTicket::new();
        order_ticket.add_order(order);
        backtest.status.cur_order_ticket = Some(order_ticket);

        backtest.execute_orders(Utc::now());

        // Check that no position was created
        let position_manager = backtest.status.cur_positions.get("BTCUSDT").unwrap();
        assert!(position_manager.long_position.is_none());

        // Cash should remain unchanged
        assert_eq!(backtest.status.cur_cash, 100.0);
    }

    #[test]
    fn test_execute_orders_close_position() {
        let mut backtest = create_test_backtest();

        // First, create a long position
        let mut position_manager =
            crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        let trade = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            Utc::now(),
            29000.0,
            Direction::Long,
            1.0,
            5,
        ).unwrap();
        
        position_manager.long_position = Some(trade);
        backtest
            .status
            .cur_positions
            .insert("BTCUSDT".to_string(), position_manager);

        // Create a close order (SHORT to close LONG)
        let close_order = Order::new(
            "BTCUSDT".to_string(),
            29300.0, // Higher price for profit
            Direction::Short,
            OrderType::Close,
            1.0,
            5,
            Utc::now(),
            Some(1),
        ).unwrap();

        let mut order_ticket = OrderTicket::new();
        order_ticket.add_order(close_order);
        backtest.status.cur_order_ticket = Some(order_ticket);

        let initial_cash = backtest.status.cur_cash;
        backtest.execute_orders(Utc::now());

        // Check that position was closed
        let position_manager = backtest.status.cur_positions.get("BTCUSDT").unwrap();
        assert!(position_manager.long_position.is_none());

        // Check that cash increased due to profit and margin release
        let expected_pnl = (29300.0 - 29000.0) * 1.0; // 300.0 profit
        let expected_margin_release = 29000.0 / 5.0; // 5800.0 margin released
        let expected_fee = 29300.0 * 1.0 * TRANSACTION_FEE;
        let expected_cash = initial_cash + expected_margin_release + expected_pnl - expected_fee;

        assert!((backtest.status.cur_cash - expected_cash).abs() < 0.01);
    }

    #[test]
    fn test_update_status_after_iteration() {
        let mut backtest = create_test_backtest();

        // Add a position to test PnL updates
        let mut position_manager =
            crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        let trade = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            Utc::now(),
            29000.0,
            Direction::Long,
            1.0,
            5,
        )
        .unwrap();
        position_manager.long_position = Some(trade);
        backtest
            .status
            .cur_positions
            .insert("BTCUSDT".to_string(), position_manager);

        backtest.update_status_after_iteration();

        // Check that prices were updated
        assert!(backtest.status.cur_price_vector.contains_key("BTCUSDT"));
        assert!(backtest.status.cur_price_vector.contains_key("ETHUSDT"));

        // Check that PnL vectors were updated
        assert!(backtest.status.cur_open_pnl_vector.contains_key("BTCUSDT"));
        assert!(backtest.status.cur_cum_pnl.contains_key("BTCUSDT"));

        // Check that margin and notional were updated
        assert!(backtest.status.cur_total_margin > 0.0);
        assert!(backtest.status.cur_total_notional > 0.0);

        // Check that portfolio MtM value was updated
        assert!(backtest.status.cur_portfolio_mtm_value > 0.0);
    }

    // Test a simple strategy execution
    fn simple_buy_strategy(backtest: &Backtest, _status: &BacktestStatus) -> OrderTicket {
        let mut order_ticket = OrderTicket::new();

        // Only buy on first iteration
        if backtest.current_time_index == 0 {
            if let Some(next_open_price) = backtest.get_next_open_price("BTCUSDT") {
                if let Ok(order) = Order::new(
                    "BTCUSDT".to_string(),
                    next_open_price,
                    Direction::Long,
                    OrderType::Open,
                    0.1,
                    5,
                    Utc::now(),
                    Some(2),
                ) {
                    order_ticket.add_order(order);
                }
            }
        }

        order_ticket
    }

    #[test]
    fn test_run_strategy() {
        let mut backtest = create_test_backtest();

        // Set start and end times based on our test data
        let start_time = DateTime::from_timestamp(1609459200, 0).unwrap(); // 2021-01-01 00:00:00
        let end_time = DateTime::from_timestamp(1609488000, 0).unwrap(); // 2021-01-01 08:00:00

        let result = backtest.run_strategy(Some(start_time), Some(end_time), simple_buy_strategy);

        assert!(result.is_ok());
        let status_history = result.unwrap();

        // Should have status snapshots for each time step
        assert!(status_history.len() >= 2);

        // Check that the strategy executed and created a position
        let final_status = status_history.values().last().unwrap();
        let btc_position = final_status.cur_positions.get("BTCUSDT").unwrap();
        assert!(btc_position.long_position.is_some());
    }

    // Strategy that creates multiple trades for testing trade history
    fn trade_history_strategy(backtest: &Backtest, _status: &BacktestStatus) -> OrderTicket {
        let mut order_ticket = OrderTicket::new();

        match backtest.current_time_index {
            0 => {
                // Open BTC long position
                if let Some(next_open_price) = backtest.get_next_open_price("BTCUSDT") {
                    if let Ok(order) = Order::new(
                        "BTCUSDT".to_string(),
                        next_open_price,
                        Direction::Long,
                        OrderType::Open,
                        0.1,
                        5,
                        Utc::now(),
                        Some(2),
                    ) {
                        order_ticket.add_order(order);
                    }
                }
            }
            1 => {
                // Open ETH short position
                if let Some(next_open_price) = backtest.get_next_open_price("ETHUSDT") {
                    if let Ok(order) = Order::new(
                        "ETHUSDT".to_string(),
                        next_open_price,
                        Direction::Short,
                        OrderType::Open,
                        1.0,
                        3,
                        Utc::now(),
                        Some(2),
                    ) {
                        order_ticket.add_order(order);
                    }
                }
            }
            2 => {
                // Close BTC position
                if let Some(next_open_price) = backtest.get_next_open_price("BTCUSDT") {
                    if let Ok(order) = Order::new(
                        "BTCUSDT".to_string(),
                        next_open_price,
                        Direction::Short,
                        OrderType::Close,
                        0.1,
                        5,
                        Utc::now(),
                        Some(1),
                    ) {
                        order_ticket.add_order(order);
                    }
                }
            }
            3 => {
                // Close ETH position
                if let Some(next_open_price) = backtest.get_next_open_price("ETHUSDT") {
                    if let Ok(order) = Order::new(
                        "ETHUSDT".to_string(),
                        next_open_price,
                        Direction::Long,
                        OrderType::Close,
                        1.0,
                        3,
                        Utc::now(),
                        Some(1),
                    ) {
                        order_ticket.add_order(order);
                    }
                }
            }
            _ => {}
        }

        order_ticket
    }

    #[test]
    fn test_get_trade_history() {
        let mut backtest = create_test_backtest();

        // Run strategy with multiple trades
        let start_time = DateTime::from_timestamp(1609459200, 0).unwrap(); // 2021-01-01 00:00:00
        let end_time = DateTime::from_timestamp(1609516800, 0).unwrap(); // 2021-01-01 16:00:00

        let result = backtest.run_strategy(Some(start_time), Some(end_time), trade_history_strategy);
        assert!(result.is_ok());
        let status_history = result.unwrap();

        // Test get_trade_history
        let trade_history = backtest.get_trade_history(&status_history);

        println!("🔍 TESTING get_trade_history METHOD");
        println!("Trade history keys: {:?}", trade_history.keys().collect::<Vec<_>>());
        
        // Should have entries for both instruments
        assert!(trade_history.contains_key("BTCUSDT"));
        assert!(trade_history.contains_key("ETHUSDT"));

        // Check BTC trades
        let btc_trades = &trade_history["BTCUSDT"];
        println!("BTC trades found: {}", btc_trades.len());
        for (i, trade) in btc_trades.iter().enumerate() {
            println!("  BTC Trade {}: instrument={}, direction={:?}, entry_price={}, exit_price={}, size={}, pnl={}, open_time={}, close_time={:?}", 
                i + 1, trade.instrument, trade.direction, trade.entry_price, trade.exit_price, trade.size, trade.pnl, trade.open_time, trade.close_time);
        }

        // Check ETH trades
        let eth_trades = &trade_history["ETHUSDT"];
        println!("ETH trades found: {}", eth_trades.len());
        for (i, trade) in eth_trades.iter().enumerate() {
            println!("  ETH Trade {}: instrument={}, direction={:?}, entry_price={}, exit_price={}, size={}, pnl={}, open_time={}, close_time={:?}", 
                i + 1, trade.instrument, trade.direction, trade.entry_price, trade.exit_price, trade.size, trade.pnl, trade.open_time, trade.close_time);
        }

        // Verify trade data integrity
        if !btc_trades.is_empty() {
            let first_btc_trade = &btc_trades[0];
            assert_eq!(first_btc_trade.instrument, "BTCUSDT");
            assert_eq!(first_btc_trade.direction, Direction::Long);
            assert!(first_btc_trade.entry_price > 0.0);
            assert!(first_btc_trade.size > 0.0);
            assert_eq!(first_btc_trade.leverage, 5);
            
            // Check that close_time is after open_time for closed trades
            if first_btc_trade.pnl != 0.0 {
                assert!( first_btc_trade.close_time >= Some(first_btc_trade.open_time));
                assert!(first_btc_trade.duration_hours >= 0.0);
            }
        }

        if !eth_trades.is_empty() {
            let first_eth_trade = &eth_trades[0];
            assert_eq!(first_eth_trade.instrument, "ETHUSDT");
            assert_eq!(first_eth_trade.direction, Direction::Short);
            assert!(first_eth_trade.entry_price > 0.0);
            assert!(first_eth_trade.size > 0.0);
            assert_eq!(first_eth_trade.leverage, 3);
        }

        // Test edge cases
        println!("\n🔍 Testing edge cases:");
        
        // Test with empty results
        let empty_results = HashMap::new();
        let empty_trade_history = backtest.get_trade_history(&empty_results);
        assert_eq!(empty_trade_history.len(), 2); // Should still have instrument keys
        assert_eq!(empty_trade_history["BTCUSDT"].len(), 0);
        assert_eq!(empty_trade_history["ETHUSDT"].len(), 0);
        println!("✅ Empty results handled correctly");

        // Verify no duplicate trades
        let all_btc_trades = &trade_history["BTCUSDT"];
        for i in 0..all_btc_trades.len() {
            for j in (i + 1)..all_btc_trades.len() {
                let trade1 = &all_btc_trades[i];
                let trade2 = &all_btc_trades[j];
                // Trades should not be identical (same open time and PnL)
                assert!(!(trade1.open_time == trade2.open_time && (trade1.pnl - trade2.pnl).abs() < 1e-10));
            }
        }
        println!("✅ No duplicate trades found");

        println!("\n🎉 get_trade_history test completed successfully!");
    }

    #[test]
    fn test_get_trade_history_with_partial_data() {
        let mut backtest = create_test_backtest();
        
        // Create a minimal status with one closed position manually
        let mut status = BacktestStatus::new(backtest.tradables.clone(), backtest.initial_cash);
        
        // Create a position manager with a closed trade
        let mut pos_manager = crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        
        let start_time = DateTime::from_timestamp(1609459200, 0).unwrap();
        let end_time = DateTime::from_timestamp(1609473600, 0).unwrap();
        
        let mut trade = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            start_time,
            29000.0,
            Direction::Long,
            0.5,
            5,
        ).unwrap();
        
        // Simulate closing the trade
        trade.close_time = Some(end_time);
        trade.closed_pnl = 500.0; // $500 profit
        
        pos_manager.closed_positions.push(trade);
        status.cur_positions.insert("BTCUSDT".to_string(), pos_manager);
        
        let mut results = HashMap::new();
        results.insert(end_time, status);
        
        // Test get_trade_history with this manufactured data
        let trade_history = backtest.get_trade_history(&results);
        
        println!("\n🔍 Testing with partial/manufactured data:");
        let btc_trades = &trade_history["BTCUSDT"];
        assert_eq!(btc_trades.len(), 1);
        
        let trade = &btc_trades[0];
        assert_eq!(trade.instrument, "BTCUSDT");
        assert_eq!(trade.direction, Direction::Long);
        assert_eq!(trade.entry_price, 29000.0);
        assert_eq!(trade.size, 0.5);
        assert_eq!(trade.leverage, 5);
        assert_eq!(trade.pnl, 500.0);
        assert_eq!(trade.open_time, start_time);
        assert_eq!(trade.close_time, Some(end_time));
        
        // Verify exit price calculation
        let expected_exit_price = 29000.0 + (500.0 / 0.5); // entry + (pnl / size)
        assert_eq!(trade.exit_price, expected_exit_price);
        
        // Verify duration calculation
        let expected_duration = (end_time.timestamp() - start_time.timestamp()) as f64 / 3600.0;
        assert_eq!(trade.duration_hours, expected_duration);
        
        println!("✅ Partial data test passed - trade record correctly created");
    }

    #[test]
    fn test_get_trade_history_detailed() {
        println!("🔍 DETAILED get_trade_history TEST");
        
        let backtest = create_test_backtest();
        
        // Create a comprehensive results history with various trade scenarios
        let mut results = HashMap::new();
        
        // Create timestamps
        let time1 = DateTime::from_timestamp(1609459200, 0).unwrap(); // 2021-01-01 00:00:00
        let time2 = DateTime::from_timestamp(1609473600, 0).unwrap(); // 2021-01-01 04:00:00  
        let time3 = DateTime::from_timestamp(1609488000, 0).unwrap(); // 2021-01-01 08:00:00
        
        // Scenario 1: Create status with one closed BTC trade
        let mut status1 = BacktestStatus::new(backtest.tradables.clone(), backtest.initial_cash);
        let mut btc_pos_manager = crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        
        let mut btc_trade = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            time1,
            29000.0,
            Direction::Long,
            0.5,
            5,
        ).unwrap();
        btc_trade.close_time = Some(time2);
        btc_trade.closed_pnl = 1000.0; // $1000 profit
        
        btc_pos_manager.closed_positions.push(btc_trade);
        status1.cur_positions.insert("BTCUSDT".to_string(), btc_pos_manager);
        results.insert(time2, status1);
        
        // Scenario 2: Create status with one closed ETH trade  
        let mut status2 = BacktestStatus::new(backtest.tradables.clone(), backtest.initial_cash);
        let mut eth_pos_manager = crate::backtesting::position::PositionManager::new("ETHUSDT".to_string());
        
        let mut eth_trade = crate::backtesting::trade::Trade::new(
            "ETHUSDT".to_string(), 
            time2,
            750.0,
            Direction::Short,
            2.0,
            3,
        ).unwrap();
        eth_trade.close_time = Some(time3);
        eth_trade.closed_pnl = -300.0; // $300 loss
        
        eth_pos_manager.closed_positions.push(eth_trade);
        status2.cur_positions.insert("ETHUSDT".to_string(), eth_pos_manager);
        results.insert(time3, status2);
        
        // Test get_trade_history
        let trade_history = backtest.get_trade_history(&results);
        
        println!("📊 Trade History Results:");
        for (instrument, trades) in &trade_history {
            println!("  {}: {} trades", instrument, trades.len());
            for (i, trade) in trades.iter().enumerate() {
                println!("    Trade {}: dir={:?}, entry=${:.2}, exit=${:.2}, size={:.4}, pnl=${:.2}", 
                    i+1, trade.direction, trade.entry_price, trade.exit_price, trade.size, trade.pnl);
            }
        }
        
        // Assertions
        assert_eq!(trade_history.len(), 2, "Should have entries for both instruments");
        assert!(trade_history.contains_key("BTCUSDT"), "Should contain BTCUSDT");
        assert!(trade_history.contains_key("ETHUSDT"), "Should contain ETHUSDT");
        
        // Check BTC trade
        let btc_trades = &trade_history["BTCUSDT"];
        assert_eq!(btc_trades.len(), 1, "Should have 1 BTC trade");
        
        let btc_trade = &btc_trades[0];
        assert_eq!(btc_trade.instrument, "BTCUSDT");
        assert_eq!(btc_trade.direction, Direction::Long);
        assert_eq!(btc_trade.entry_price, 29000.0);
        assert_eq!(btc_trade.size, 0.5);
        assert_eq!(btc_trade.leverage, 5);
        assert_eq!(btc_trade.pnl, 1000.0);
        assert_eq!(btc_trade.open_time, time1);
        assert_eq!(btc_trade.close_time, Some(time2));
        
        // Check exit price calculation (entry_price + pnl/size for LONG)
        let expected_btc_exit_price = 29000.0 + (1000.0 / 0.5); // 29000 + 2000 = 31000
        assert_eq!(btc_trade.exit_price, expected_btc_exit_price);
        
        // Check duration calculation
        let expected_btc_duration = (time2.timestamp() - time1.timestamp()) as f64 / 3600.0; // 4 hours
        assert_eq!(btc_trade.duration_hours, expected_btc_duration);
        
        // Check ETH trade
        let eth_trades = &trade_history["ETHUSDT"];
        assert_eq!(eth_trades.len(), 1, "Should have 1 ETH trade");
        
        let eth_trade = &eth_trades[0];
        assert_eq!(eth_trade.instrument, "ETHUSDT");
        assert_eq!(eth_trade.direction, Direction::Short);
        assert_eq!(eth_trade.entry_price, 750.0);
        assert_eq!(eth_trade.size, 2.0);
        assert_eq!(eth_trade.leverage, 3);
        assert_eq!(eth_trade.pnl, -300.0);
        assert_eq!(eth_trade.open_time, time2);
        assert_eq!(eth_trade.close_time, Some(time3));
        
        // Check exit price calculation (entry_price - pnl/size for SHORT)
        let expected_eth_exit_price = 750.0 - (-300.0 / 2.0); // 750 - (-150) = 900
        assert_eq!(eth_trade.exit_price, expected_eth_exit_price);
        
        println!("✅ All detailed trade history tests passed!");
    }

    #[test]
    fn test_get_trade_history_edge_cases() {
        println!("🔍 TESTING get_trade_history EDGE CASES");
        
        let backtest = create_test_backtest();
        
        // Test Case 1: Empty results
        let empty_results = HashMap::new();
        let empty_history = backtest.get_trade_history(&empty_results);
        
        assert_eq!(empty_history.len(), 2, "Should have entries for both instruments even with empty results");
        assert_eq!(empty_history["BTCUSDT"].len(), 0, "BTCUSDT should have no trades");
        assert_eq!(empty_history["ETHUSDT"].len(), 0, "ETHUSDT should have no trades");
        println!("✅ Empty results test passed");
        
        // Test Case 2: Results with no closed positions
        let mut results_no_trades = HashMap::new();
        let status_no_trades = BacktestStatus::new(backtest.tradables.clone(), backtest.initial_cash);
        let time = DateTime::from_timestamp(1609459200, 0).unwrap();
        results_no_trades.insert(time, status_no_trades);
        
        let history_no_trades = backtest.get_trade_history(&results_no_trades);
        assert_eq!(history_no_trades["BTCUSDT"].len(), 0, "Should have no BTC trades");
        assert_eq!(history_no_trades["ETHUSDT"].len(), 0, "Should have no ETH trades");
        println!("✅ No trades test passed");
        
        // Test Case 3: Zero PnL trade
        let mut results_zero_pnl = HashMap::new();
        let mut status_zero_pnl = BacktestStatus::new(backtest.tradables.clone(), backtest.initial_cash);
        let mut pos_manager_zero = crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        
        let start_time = DateTime::from_timestamp(1609459200, 0).unwrap();
        let end_time = DateTime::from_timestamp(1609473600, 0).unwrap();
        
        let mut zero_pnl_trade = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            start_time,
            30000.0,
            Direction::Long,
            1.0,
            5,
        ).unwrap();
        zero_pnl_trade.close_time = Some(end_time);
        zero_pnl_trade.closed_pnl = 0.0; // Zero PnL
        
        pos_manager_zero.closed_positions.push(zero_pnl_trade);
        status_zero_pnl.cur_positions.insert("BTCUSDT".to_string(), pos_manager_zero);
        results_zero_pnl.insert(end_time, status_zero_pnl);
        
        let history_zero_pnl = backtest.get_trade_history(&results_zero_pnl);
        let zero_pnl_trades = &history_zero_pnl["BTCUSDT"];
        assert_eq!(zero_pnl_trades.len(), 1, "Should have 1 trade with zero PnL");
        
        let zero_trade = &zero_pnl_trades[0];
        assert_eq!(zero_trade.pnl, 0.0, "PnL should be zero");
        assert_eq!(zero_trade.exit_price, zero_trade.entry_price, "Exit price should equal entry price for zero PnL");
        println!("✅ Zero PnL test passed");
        
        // Test Case 4: Multiple trades same instrument
        let mut results_multiple = HashMap::new();
        let mut status_multiple = BacktestStatus::new(backtest.tradables.clone(), backtest.initial_cash);
        let mut pos_manager_multiple = crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        
        // Create two different trades
        let time_1 = DateTime::from_timestamp(1609459200, 0).unwrap();
        let time_2 = DateTime::from_timestamp(1609473600, 0).unwrap();
        let time_3 = DateTime::from_timestamp(1609488000, 0).unwrap();
        
        let mut trade1 = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            time_1,
            29000.0,
            Direction::Long,
            0.5,
            5,
        ).unwrap();
        trade1.close_time = Some(time_2);
        trade1.closed_pnl = 500.0;
        
        let mut trade2 = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            time_2,
            30000.0,
            Direction::Short,
            1.0,
            3,
        ).unwrap();
        trade2.close_time = Some(time_3);
        trade2.closed_pnl = 800.0;
        
        pos_manager_multiple.closed_positions.push(trade1);
        pos_manager_multiple.closed_positions.push(trade2);
        status_multiple.cur_positions.insert("BTCUSDT".to_string(), pos_manager_multiple);
        results_multiple.insert(time_3, status_multiple);
        
        let history_multiple = backtest.get_trade_history(&results_multiple);
        let multiple_trades = &history_multiple["BTCUSDT"];
        assert_eq!(multiple_trades.len(), 2, "Should have 2 BTC trades");
        
        // Verify both trades are distinct
        let trade_1 = &multiple_trades[0];
        let trade_2 = &multiple_trades[1];
        assert_ne!(trade_1.entry_price, trade_2.entry_price, "Trades should have different entry prices");
        assert_ne!(trade_1.direction, trade_2.direction, "Trades should have different directions");
        assert_ne!(trade_1.pnl, trade_2.pnl, "Trades should have different PnL");
        
        println!("✅ Multiple trades test passed");
        
        println!("🎉 All edge case tests completed successfully!");
    }

    #[test]
    fn test_get_trade_history_validation() {
        println!("🔍 TESTING get_trade_history DATA VALIDATION");
        
        let backtest = create_test_backtest();
        
        // Create a status with various trade scenarios for validation
        let mut results = HashMap::new();
        let mut status = BacktestStatus::new(backtest.tradables.clone(), backtest.initial_cash);
        
        // Create position managers for both instruments
        let mut btc_pos_manager = crate::backtesting::position::PositionManager::new("BTCUSDT".to_string());
        let mut eth_pos_manager = crate::backtesting::position::PositionManager::new("ETHUSDT".to_string());
        
        let start_time = DateTime::from_timestamp(1609459200, 0).unwrap();
        let end_time = DateTime::from_timestamp(1609488000, 0).unwrap(); // 8 hours later
        
        // BTC Long trade with profit
        let mut btc_long_trade = crate::backtesting::trade::Trade::new(
            "BTCUSDT".to_string(),
            start_time,
            29000.0,
            Direction::Long,
            0.25,
            10,
        ).unwrap();
        btc_long_trade.close_time = Some(end_time);
        btc_long_trade.closed_pnl = 725.0; // 0.25 * (32000 - 29000) = 750, minus some fees
        
        // ETH Short trade with loss
        let mut eth_short_trade = crate::backtesting::trade::Trade::new(
            "ETHUSDT".to_string(),
            start_time,
            750.0,
            Direction::Short,
            3.0,
            4,
        ).unwrap();
        eth_short_trade.close_time = Some(end_time);
        eth_short_trade.closed_pnl = -450.0; // Loss
        
        btc_pos_manager.closed_positions.push(btc_long_trade);
        eth_pos_manager.closed_positions.push(eth_short_trade);
        
        status.cur_positions.insert("BTCUSDT".to_string(), btc_pos_manager);
        status.cur_positions.insert("ETHUSDT".to_string(), eth_pos_manager);
        results.insert(end_time, status);
        
        // Get trade history
        let trade_history = backtest.get_trade_history(&results);
        
        println!("📊 Validating trade history data quality...");
        
        // Validation 1: Check all instruments are present
        for instrument in &backtest.tradables {
            assert!(trade_history.contains_key(instrument), 
                "Trade history should contain entry for {}", instrument);
        }
        println!("✅ All instruments present in trade history");
        
        // Validation 2: Check BTC trade data integrity
        let btc_trades = &trade_history["BTCUSDT"];
        assert_eq!(btc_trades.len(), 1, "Should have exactly 1 BTC trade");
        
        let btc_trade = &btc_trades[0];
        
        // Check basic trade fields
        assert_eq!(btc_trade.instrument, "BTCUSDT", "Instrument should match");
        assert_eq!(btc_trade.direction, Direction::Long, "Direction should be Long");
        assert!(btc_trade.entry_price > 0.0, "Entry price should be positive");
        assert!(btc_trade.size > 0.0, "Size should be positive");
        assert!(btc_trade.leverage > 0, "Leverage should be positive");
        
        // Check time relationships
        assert!(btc_trade.close_time >= Some(btc_trade.open_time), "Close time should be >= open time");
        assert!(btc_trade.duration_hours >= 0.0, "Duration should be non-negative");
        
        // Check exit price calculation for Long position
        let expected_btc_exit = btc_trade.entry_price + (btc_trade.pnl / btc_trade.size);
        assert!((btc_trade.exit_price - expected_btc_exit).abs() < 1e-10, 
            "Exit price calculation incorrect for Long position");
        
        println!("✅ BTC trade validation passed");
        
        // Validation 3: Check ETH trade data integrity
        let eth_trades = &trade_history["ETHUSDT"];
        assert_eq!(eth_trades.len(), 1, "Should have exactly 1 ETH trade");
        
        let eth_trade = &eth_trades[0];
        
        // Check basic trade fields
        assert_eq!(eth_trade.instrument, "ETHUSDT", "Instrument should match");
        assert_eq!(eth_trade.direction, Direction::Short, "Direction should be Short");
        assert!(eth_trade.entry_price > 0.0, "Entry price should be positive");
        assert!(eth_trade.size > 0.0, "Size should be positive");
        assert!(eth_trade.leverage > 0, "Leverage should be positive");
        
        // Check exit price calculation for Short position
        let expected_eth_exit = eth_trade.entry_price - (eth_trade.pnl / eth_trade.size);
        assert!((eth_trade.exit_price - expected_eth_exit).abs() < 1e-10, 
            "Exit price calculation incorrect for Short position");
        
        println!("✅ ETH trade validation passed");
        
        // Validation 4: Check duration calculations
        let expected_duration = (end_time.timestamp() - start_time.timestamp()) as f64 / 3600.0;
        assert!((btc_trade.duration_hours - expected_duration).abs() < 1e-10, 
            "BTC trade duration calculation incorrect");
        assert!((eth_trade.duration_hours - expected_duration).abs() < 1e-10, 
            "ETH trade duration calculation incorrect");
        
        println!("✅ Duration calculations validated");
        
        // Validation 5: Check PnL signs make sense
        assert!(btc_trade.pnl > 0.0, "BTC Long trade should be profitable (test data designed this way)");
        assert!(eth_trade.pnl < 0.0, "ETH Short trade should be loss (test data designed this way)");
        
        println!("✅ PnL signs validated");
        
        // Validation 6: Check for no duplicate trades (defensive check)
        let all_btc_open_times: Vec<_> = btc_trades.iter().map(|t| t.open_time).collect();
        let unique_btc_times: std::collections::HashSet<_> = all_btc_open_times.iter().collect();
        assert_eq!(all_btc_open_times.len(), unique_btc_times.len(), 
            "No duplicate BTC trades should exist");
        
        let all_eth_open_times: Vec<_> = eth_trades.iter().map(|t| t.open_time).collect();
        let unique_eth_times: std::collections::HashSet<_> = all_eth_open_times.iter().collect();
        assert_eq!(all_eth_open_times.len(), unique_eth_times.len(), 
            "No duplicate ETH trades should exist");
        
        println!("✅ No duplicate trades found");
        
        println!("🎉 All validation tests completed successfully!");
        println!("✅ get_trade_history method working correctly!");
    }
}
