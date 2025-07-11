// This file defines the BacktestStatus struct, which holds the current state of the backtesting process, including positions, cash, and performance metrics.

use crate::backtesting::order::OrderTicket;
use crate::backtesting::position::PositionManager;
use crate::utils::types::TradeStatus;
use std::collections::HashMap;



#[derive(Debug, Clone)]
pub struct BacktestStatus {
    pub tradables: Vec<String>,
    pub initial_cash: f64, // Initial cash available for trading
    pub cur_positions: HashMap<String, PositionManager>,    
    pub cur_price_vector: HashMap<String, f64>,
    pub cur_open_pnl_vector: HashMap<String, f64>,
    pub cur_cum_pnl: HashMap<String, f64>,
    pub cur_cash: f64,
    pub cur_cum_funding_cost: HashMap<String, f64>,
    pub cur_portfolio_mtm_value: f64,
    pub cur_order_ticket: Option<OrderTicket>,
    pub cur_total_margin: f64,
    pub cur_total_notional: f64,
    pub cur_total_leverage: f64,
}

impl BacktestStatus {
    pub fn new(tradables: Vec<String>, initial_cash: f64) -> Self {
        let mut cur_positions = HashMap::new();
        for imnt in &tradables {
            cur_positions.insert(imnt.clone(), PositionManager::new(imnt.clone()));
        }

        Self {
            tradables,
            initial_cash,
            cur_positions,
            cur_price_vector: HashMap::new(),
            cur_open_pnl_vector: HashMap::new(),
            cur_cum_pnl: HashMap::new(),
            cur_cash: initial_cash,
            cur_cum_funding_cost: HashMap::new(),
            cur_portfolio_mtm_value: initial_cash,
            cur_order_ticket: None,
            cur_total_margin: 0.0,
            cur_total_notional: 0.0,
            cur_total_leverage: 0.0,
        }
    }

    pub fn update_margin_and_notional(&mut self) {
        let mut total_margin = 0.0;
        let mut total_notional = 0.0;

        for (imnt, position_manager) in &self.cur_positions {
            if let Some(long_trade) = &position_manager.long_position {
                if long_trade.trade_status == TradeStatus::Open {
                    total_margin += long_trade.margin();
                    total_notional += self.cur_price_vector[imnt] * long_trade.size;
                }
            }
            if let Some(short_trade) = &position_manager.short_position {
                if short_trade.trade_status == TradeStatus::Open {
                    total_margin += short_trade.margin();
                    total_notional += self.cur_price_vector[imnt] * short_trade.size;
                }
            }
        }

        self.cur_total_margin = total_margin;
        self.cur_total_notional = total_notional;
        self.cur_total_leverage = if total_margin > 0.0 {
            total_notional / total_margin
        } else {
            0.0
        };
    }
}

impl std::fmt::Display for BacktestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Current Positions: {:?}, ", self.cur_positions)?;
        write!(f, "Current Prices: {:?}, ", self.cur_price_vector)?;
        write!(f, "Current Open PnL: {:?}, ", self.cur_open_pnl_vector)?;
        write!(f, "Current Cumulative PnL: {:?}, ", self.cur_cum_pnl)?;
        write!(f, "Current Cash: {}, ", self.cur_cash)?;
        write!(f, "Current Funding Cost: {:?}, ", self.cur_cum_funding_cost)?;
        write!(
            f,
            "Current Portfolio MtM Value: {}, ",
            self.cur_portfolio_mtm_value
        )?;
        write!(f, "Current Order Ticket: {:?}, ", self.cur_order_ticket)?;
        write!(f, "Current Margin: {}, ", self.cur_total_margin)?;
        write!(f, "Current Notional: {}, ", self.cur_total_notional)?;
        write!(f, "Current Total Leverage: {}", self.cur_total_leverage)
    }
}
// Additional structs and enums like PositionManager, TradeStatus, OrderTicket, etc., should be defined in their respective files.

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtesting::order::OrderTicket;
    use crate::backtesting::position::PositionManager;

    struct BacktestStatusTest {
        tradables: Vec<String>,
        initial_cash: f64,
        backtest_status: BacktestStatus,
    }

    impl BacktestStatusTest {
        fn new() -> Self {
            let tradables = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
            let initial_cash = 10000.0;
            let backtest_status = BacktestStatus::new(tradables.clone(), initial_cash);

            Self {
                tradables,
                initial_cash,
                backtest_status,
            }
        }
    }

    #[test]
    fn test_update_positions() {
        let mut test = BacktestStatusTest::new();

        // Test updating the positions in the backtest status
        let position_manager = PositionManager::new("BTCUSDT".to_string());
        test.backtest_status
            .cur_positions
            .insert("BTCUSDT".to_string(), position_manager.clone());

        // Check if the positions are updated correctly
        assert_eq!(
            test.backtest_status.cur_positions["BTCUSDT"].imnt,
            position_manager.imnt
        );
    }

    #[test]
    fn test_initial_cash() {
        let test = BacktestStatusTest::new();

        // Test that the initial cash is set correctly
        assert_eq!(test.backtest_status.initial_cash, test.initial_cash);
    }

    #[test]
    fn test_update_price_vector() {
        let mut test = BacktestStatusTest::new();

        // Add this debug output
        println!(
            "Keys: {:?}",
            test.backtest_status
                .cur_price_vector
                .keys()
                .collect::<Vec<_>>()
        );
        println!("HashMap: {:?}", test.backtest_status.cur_price_vector);

        let price = 5000.0;
        test.backtest_status
            .cur_price_vector
            .insert("BTCUSDT".to_string(), price);

        // Add this after insertion
        println!(
            "After insertion - Keys: {:?}",
            test.backtest_status
                .cur_price_vector
                .keys()
                .collect::<Vec<_>>()
        );
        println!(
            "After insertion - HashMap: {:?}",
            test.backtest_status.cur_price_vector
        );

        // Check if the price vector is updated correctly
        assert_eq!(test.backtest_status.cur_price_vector["BTCUSDT"], price);
    }

    #[test]
    fn test_update_pnl_vector() {
        let mut test = BacktestStatusTest::new();

        // Test updating the PnL vector in the backtest status
        let pnl = 100.0;
        test.backtest_status
            .cur_cum_pnl
            .insert("BTCUSDT".to_string(), pnl);

        // Check if the PnL vector is updated correctly
        assert_eq!(test.backtest_status.cur_cum_pnl["BTCUSDT"], pnl);
    }

    #[test]
    fn test_update_cash() {
        let mut test = BacktestStatusTest::new();

        // Test updating the cash in the backtest status
        let cash = 5000.0;
        test.backtest_status.cur_cash = cash;

        // Check if the cash is updated correctly
        assert_eq!(test.backtest_status.cur_cash, cash);
    }

    #[test]
    fn test_update_portfolio_pnl() {
        let mut test = BacktestStatusTest::new();

        // Test updating the portfolio PnL in the backtest status
        let portfolio_pnl = 1000.0;
        test.backtest_status.cur_portfolio_mtm_value = portfolio_pnl;

        // Check if the portfolio PnL is updated correctly
        assert_eq!(test.backtest_status.cur_portfolio_mtm_value, portfolio_pnl);
    }

    #[test]
    fn test_update_order_ticket() {
        let mut test = BacktestStatusTest::new();

        // Test updating the order ticket in the backtest status
        let order_ticket = OrderTicket::new();
        test.backtest_status.cur_order_ticket = Some(order_ticket);

        // Check if the order ticket is updated correctly
        assert!(test.backtest_status.cur_order_ticket.is_some());
    }

    #[test]
    fn test_clone_functionality() {
        let test = BacktestStatusTest::new();

        // Test that cloning works properly (equivalent to deepcopy)
        let cloned_status = test.backtest_status.clone();

        assert_eq!(cloned_status.tradables, test.backtest_status.tradables);
        assert_eq!(
            cloned_status.initial_cash,
            test.backtest_status.initial_cash
        );
        assert_eq!(cloned_status.cur_cash, test.backtest_status.cur_cash);
        assert_eq!(
            cloned_status.cur_portfolio_mtm_value,
            test.backtest_status.cur_portfolio_mtm_value
        );
    }

    #[test]
    fn test_display_formatting() {
        let test = BacktestStatusTest::new();

        // Test that Display trait works
        let display_string = format!("{}", test.backtest_status);
        assert!(display_string.contains("Current Positions"));
        assert!(display_string.contains("Current Cash"));
        assert!(display_string.contains("Current Portfolio MtM Value"));
    }
}
