// This file defines the Trade struct, which encapsulates the details of a trade;
// A trade is an open position in a financial instrument, including its entry price, size, direction, and status.
// It also includes methods for closing the position and updating unrealized profit and loss (PnL).

use crate::utils::types::{Direction, TradeStatus};
use crate::{log_debug, log_info};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct Trade {
    pub imnt: String,
    pub filled_time: DateTime<Utc>,
    pub entry_price: f64,
    pub direction: Direction,
    pub size: f64,                         // Non-negative float
    pub leverage: u8,                      // Non-zero positive integer
    pub trade_status: TradeStatus,         // Defaults to OPEN
    pub closed_pnl: f64,                   // Defaults to 0.0
    pub open_pnl: f64,                     // Defaults to 0.0
    pub close_time: Option<DateTime<Utc>>, // Optional close time
}

impl Trade {
    pub fn new(
        imnt: String,
        filled_time: chrono::DateTime<Utc>,
        entry_price: f64,
        direction: Direction,
        size: f64,
        leverage: u8,
    ) -> Result<Self, String> {
        if size < 0.0 {
            return Err("Trade size must be non-negative".to_string());
        }

        if entry_price <= 0.0 {
            return Err("Entry price must be positive".to_string());
        }

        if leverage == 0 {
            return Err("Leverage must be non-zero integer".to_string());
        }

        Ok(Trade {
            imnt,
            filled_time,
            entry_price,
            direction,
            size,
            leverage,
            trade_status: TradeStatus::Open,
            closed_pnl: 0.0,
            open_pnl: 0.0,
            close_time: None,
        })
    }

    /// Returns the margin of the trade based on the entry price, size and leverage
    pub fn margin(&self) -> f64 {
        self.entry_price * self.size / (self.leverage as f64)
    }

    /// Returns a string summarizing the trade's details
    pub fn info(&self) -> String {
        format!(
            "Trade: {:?} {:?}: | Entry Price: {:.2} | Size: {:.2} | Leverage: {:.2} | Status: {:?} | Reailzed Pnl: {:.2} | Unrealized Pnl: {:.2} | Closed Time: {:?}",
            self.direction,
            self.imnt,
            self.entry_price,
            self.size,
            self.leverage,
            self.trade_status,
            self.closed_pnl,
            self.open_pnl,
            self.close_time.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()).unwrap_or("None".to_string())
        )
    }

    /// Add more positions to this existing trade; recalculate entry price
    pub fn add_position(&mut self, open_price: f64, open_size: f64) -> Result<(), String> {
        if open_price <= 0.0 {
            return Err("Open price must be positive".to_string());
        }
        if open_size <= 0.0 {
            return Err("Open size must be positive".to_string());
        }
        
        let new_entry_price =
            (open_price * open_size + self.entry_price * self.size) / (open_size + self.size);
        self.entry_price = new_entry_price;
        self.size += open_size;
        Ok(())
    }

    /// Close or partially close a trade and recalculate its open and closed pnL
    pub fn close_position(&mut self, close_price: f64, close_size: f64, close_time: DateTime<Utc>) -> Result<(), String> {
        if close_price <= 0.0 {
            return Err("Close price must be positive".to_string());
        }
        if close_size <= 0.0 {
            return Err("Close size must be positive".to_string());
        }
        if self.size <= close_size {
            // Full close
            self.trade_status = TradeStatus::Closed;
            let close_size = self.size; // Use actual size for calculation

            let pnl = match self.direction {
                Direction::Long => (close_price - self.entry_price) * close_size,
                Direction::Short => (self.entry_price - close_price) * close_size,
            };

            self.closed_pnl += pnl;
            self.open_pnl = 0.0;
            self.size = 0.0;
            self.close_time = Some(close_time);

            log_info!(
                "{}: {} trade closed with PnL: {:.2}",
                close_time.format("%Y-%m-%d %H:%M:%S"),
                self.imnt,
                self.closed_pnl
            );
        } else {
            // Partial close
            let partial_pnl = match self.direction {
                Direction::Long => (close_price - self.entry_price) * close_size,
                Direction::Short => (self.entry_price - close_price) * close_size,
            };

            self.closed_pnl += partial_pnl;
            self.size -= close_size;
        }
        Ok(())
    }

    /// Update the current MtM PnL of the trade
    pub fn update_unrealized_pnl(&mut self, cur_price: f64) -> Result<(), String> {
        if cur_price <= 0.0 {
            return Err("Current price must be positive".to_string());
        }
        
        self.open_pnl = match self.direction {
            Direction::Long => (cur_price - self.entry_price) * self.size,
            Direction::Short => (self.entry_price - cur_price) * self.size,
        };

        log_debug!(
            "Updated trade: {:?} {}. Open PnL: {:.2}",
            self.direction, self.imnt, self.open_pnl
        );
        Ok(())
    }
}

// Unit tests for the Trade struct
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    fn create_long_trade() -> Trade {
        Trade::new(
            "BTCUSDT".to_string(),
            DateTime::parse_from_rfc3339("2023-09-05T14:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            5000.0,
            Direction::Long,
            2.0,
            5,
        )
        .unwrap()
    }

    fn create_short_trade() -> Trade {
        Trade::new(
            "BTCUSDT".to_string(),
            DateTime::parse_from_rfc3339("2023-09-05T14:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            5500.0,
            Direction::Short,
            3.0,
            5,
        )
        .unwrap()
    }

    #[test]
    fn test_initial_margin() {
        let trade = create_long_trade();
        let expected_margin = 5000.0 * 2.0 / 5.0; // price * size / leverage
        assert_eq!(trade.margin(), expected_margin);
    }

    #[test]
    fn test_add_position() {
        let mut trade = create_long_trade();
        trade.add_position(5100.0, 1.0).unwrap();

        // New entry price calculation
        let expected_entry_price = (5000.0 * 2.0 + 5100.0 * 1.0) / (2.0 + 1.0);
        assert_eq!(trade.entry_price, expected_entry_price);
        assert_eq!(trade.size, 3.0); // size increases to 3
    }

    #[test]
    fn test_close_whole_position() {
        let mut trade = create_long_trade();
        let close_price = 5200.0;
        let close_time = Utc::now();

        // Close the whole position
        trade.close_position(close_price, 3.0, close_time).unwrap();

        // Check if the position is closed correctly
        assert!(matches!(trade.trade_status, TradeStatus::Closed));
        assert_eq!(trade.size, 0.0);
        assert_eq!(trade.open_pnl, 0.0);
        assert_eq!(trade.closed_pnl, (close_price - 5000.0) * 2.0);
    }

    #[test]
    fn test_close_partial_position() {
        let mut trade = create_long_trade();
        let close_price = 5200.0;
        let close_size = 1.0;
        let close_time = Utc::now();

        // Open PnL should be updated
        trade.update_unrealized_pnl(close_price).unwrap();

        // Close a partial position
        trade.close_position(close_price, close_size, close_time).unwrap();

        // Check remaining size and closed PnL
        assert_eq!(trade.size, 1.0); // size should reduce to 1.0
        let expected_closed_pnl = (close_price - 5000.0) * close_size;
        assert_eq!(trade.closed_pnl, expected_closed_pnl);
    }

    #[test]
    fn test_close_partial_position_short() {
        let mut short_trade = create_short_trade();
        let close_price = 5100.0;
        let close_size = 1.0;
        let close_time = Utc::now();

        // Close a partial position
        short_trade.close_position(close_price, close_size, close_time).unwrap();
        short_trade.update_unrealized_pnl(close_price).unwrap();

        // Check remaining size and closed PnL
        assert_eq!(short_trade.size, 2.0);
        assert!(matches!(short_trade.trade_status, TradeStatus::Open));

        let expected_closed_pnl = (5500.0 - close_price) * close_size;
        assert_eq!(short_trade.closed_pnl, expected_closed_pnl);
        let expected_open_pnl = (5500.0 - close_price) * short_trade.size;
        assert_eq!(short_trade.open_pnl, expected_open_pnl);
    }

    #[test]
    fn test_update_unrealized_pnl() {
        let mut trade = create_long_trade();
        let current_price = 5200.0;
        trade.update_unrealized_pnl(current_price).unwrap();

        // Open PnL should be calculated
        let expected_open_pnl = (current_price - 5000.0) * 2.0; // (current price - entry price) * size
        assert_eq!(trade.open_pnl, expected_open_pnl);

        let mut short_trade = create_short_trade();
        let cur_price = 5500.0;
        short_trade.update_unrealized_pnl(cur_price).unwrap();
        let expected_open_pnl = (5500.0 - cur_price) * short_trade.size;
        assert_eq!(short_trade.open_pnl, expected_open_pnl);
    }
}
