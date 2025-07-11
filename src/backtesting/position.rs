// This file defines the Position struct and related functions, managing the current positions in the market, including opening and closing trades.

use crate::backtesting::trade::Trade;
use crate::utils::types::{Direction, TradeStatus};
use crate::log_error;

/// Manages positions and trade history for a single instrument.
/// Ensures that at most one LONG and one SHORT trade is open simultaneously.
#[derive(Debug, Clone)]
pub struct PositionManager {
    /// The symbol of the instrument being traded
    pub imnt: String,
    /// Curent open long position (if any)
    pub long_position: Option<Trade>,
    /// Current open short position (if any)
    pub short_position: Option<Trade>,
    /// Closed positions for the instrument
    pub closed_positions: Vec<Trade>,
    /// The cumulative closed PnL for all trades
    pub cum_closed_pnl: f64,
}

impl PositionManager {
    /// Creates a new PositionManager for a specific instrument
    pub fn new(imnt: String) -> Self {
        PositionManager {
            imnt,
            long_position: None,
            short_position: None,
            closed_positions: Vec::new(),
            cum_closed_pnl: 0.0,
        }
    }

    /// Processes a new trade to either open, update, or close a position
    pub fn process_new_trade(&mut self, new_trade: Trade) {
        let current_position = match new_trade.direction {
            Direction::Long => &mut self.long_position,
            Direction::Short => &mut self.short_position,
        };

        // Determine the current position based on the direction of the new trade
        // If there is an existing open position
        if let Some(position) = current_position {
            if position.trade_status == TradeStatus::Open {
                if position.direction == new_trade.direction {
                    // Add to existing position with the same direction
                    if let Err(e) = position.add_position(new_trade.entry_price, new_trade.size) {
                        log_error!("Error adding position for {}: {}", self.imnt, e);
                        return;
                    }
                } else {
                    // Close or partially close the position with opposite direction
                    let initial_closed_pnl = position.closed_pnl;
                    if let Err(e) = position.close_position(
                        new_trade.entry_price,
                        new_trade.size,
                        new_trade.filled_time,
                    ) {
                        log_error!("Error closing position for {}: {}", self.imnt, e);
                        return;
                    }

                    // Update cumulative PnL with only the new closed PnL
                    self.cum_closed_pnl += position.closed_pnl - initial_closed_pnl;
                }
            } else {
                // No open position, add the new trade
                if new_trade.direction == Direction::Long {
                    self.long_position = Some(new_trade);
                } else {
                    self.short_position = Some(new_trade);
                }
            }
        } else {
            // No open position, add the new trade
            if new_trade.direction == Direction::Long {
                self.long_position = Some(new_trade);
            } else {
                self.short_position = Some(new_trade);
            }
        }
    }

    /// Updates the unrealized PnL for the current open position at the current market price
    pub fn update_position_unrealized_pnl(&mut self, current_price: f64) {
        if let Some(position) = self.long_position.as_mut() {
            if position.trade_status == TradeStatus::Open {
                if let Err(e) = position.update_unrealized_pnl(current_price) {
                    log_error!("Error updating long position PnL for {}: {}", self.imnt, e);
                }
            }
        }

        if let Some(position) = self.short_position.as_mut() {
            if position.trade_status == TradeStatus::Open {
                if let Err(e) = position.update_unrealized_pnl(current_price) {
                    log_error!("Error updating short position PnL for {}: {}", self.imnt, e);
                }
            }
        }
    }
}
