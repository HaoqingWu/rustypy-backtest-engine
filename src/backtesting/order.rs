// This file defines the Order struct and related functions, encapsulating the details of trading orders, including attributes like price, size, and type.

use crate::backtesting::trade::Trade;
use crate::utils::types::{Direction, OrderType};
use crate::{log_debug, log_info, log_warn};
use chrono::{DateTime, Utc};
use std::collections::{HashMap, BTreeMap};

#[derive(Debug, Clone)]
pub struct Order {
    pub imnt: String,
    pub price: f64,
    pub size: f64,
    pub order_type: OrderType,
    pub direction: Direction,
    pub leverage: u8,          // Non-zero positive integer
    pub duration: Option<u32>, // Duration in time bars or None for GTC (Good-til-Cancelled)
    pub time_created: DateTime<Utc>,
}

impl Order {
    pub fn new(
        imnt: String,
        price: f64,
        direction: Direction,
        order_type: OrderType,       // The type of order (Open or Close)
        size: f64,                   // The size of the order (with leverage applied)
        leverage: u8,                // The leverage for the order
        time_created: DateTime<Utc>, // Timestamp when the order was created
        duration: Option<u32>,       // Duration in time bars or None for GTC (Good-til-Cancelled)
    ) -> Result<Self, String> {
        if price <= 0.0 {
            return Err("Order price must be positive".to_string());
        }
        if size <= 0.0 {
            return Err("Order size must be positive".to_string());
        }
        if leverage == 0 {
            return Err("Leverage must be non-zero".to_string());
        }
        Ok(Order {
            imnt,
            price,
            direction,
            order_type,
            leverage,
            size,
            time_created,
            duration,
        })
    }

    pub fn notional(&self) -> f64 {
        self.price * self.size
    }

    pub fn margin(&self) -> f64 {
        self.notional() / self.leverage as f64
    }

    /// Returns a string summarizing the order's details
    pub fn info(&self) -> String {
        format!(
            "Order: {:?} {:?}: | Price: {:.2} | Size: {:.2} | Leverage: {:.2} | Type: {:?} | Duration: {:?} | Time Created: {:?}",
            self.direction,
            self.imnt,
            self.price,
            self.size,
            self.leverage,
            self.order_type,
            self.duration,
            self.time_created.format("%Y-%m-%d %H:%M:%S")
        )
    }

    /// Transforms the Order into a Trade object if filled
    pub fn to_trade(&self, filled_time: DateTime<Utc>) -> Result<Trade, String> {
        Trade::new(
            self.imnt.clone(),
            filled_time,
            self.price,
            self.direction.clone(),
            self.size,
            self.leverage,
        )
    }
}

/// Manages trading orders as a BTreeMap for deterministic iteration order.
/// Keys are tradable instruments and values are vectors of Order objects.
/// BTreeMap ensures orders are processed alphabetically by instrument name.
#[derive(Debug, Clone)]
pub struct OrderTicket {
    pub orders: BTreeMap<String, Vec<Order>>,
}

impl OrderTicket {
    pub fn new() -> Self {
        OrderTicket {
            orders: BTreeMap::new(),
        }
    }

    /// Initialize OrderTicket with existing data
    pub fn from_map(orders: HashMap<String, Vec<Order>>) -> Self {
        OrderTicket { orders: orders.into_iter().collect() }
    }

    // Helper function to check if two orders match for removal
    fn orders_match(order1: &Order, order2: &Order) -> bool {
        order1.imnt == order2.imnt
            && order1.price == order2.price
            && order1.direction == order2.direction
            && order1.order_type == order2.order_type
            && order1.size == order2.size
            && order1.leverage == order2.leverage
    }

    /// Add an order to the current `OrderTicket` instance.
    pub fn add_order(&mut self, order: Order) {
        self.orders
            .entry(order.imnt.clone())
            .or_insert_with(Vec::new)
            .push(order.clone());

        log_debug!(
            "{} Added order: {:?} {:?} Price: {:.2} Size: {:.2}",
            order.time_created.format("%Y-%m-%d %H:%M:%S"),
            order.direction,
            order.imnt,
            order.price,
            order.size
        );
    }

    /// Remove an order for a given instrument.
    /// If no price, direction, or size is given, remove all orders for that instrument.
    pub fn remove_order(&mut self, order: &Order) {
        let imnt = &order.imnt;

        if let Some(orders) = self.orders.get_mut(imnt) {
            // Find and remove the matching order
            if let Some(pos) = orders
                .iter()
                .position(|o: &Order| Self::orders_match(o, order))
            {
                orders.remove(pos);
                log_debug!(
                    "{}: Order removed for {}.",
                    order.time_created.format("%Y-%m-%d %H:%M:%S"),
                    imnt
                );

                // Remove the instrument entry if no orders remain
                if orders.is_empty() {
                    self.orders.remove(imnt);
                }
            } else {
                log_warn!("No matching orders found for {}.", imnt);
            }
        } else {
            log_warn!("No existing orders for {} to remove.", imnt);
        }
    }

    /// Update the duration of all active orders.
    /// - Decrease the duration of orders that have a finite duration.
    /// - Remove orders with duration <= 0.
    /// - Leave GTC orders (duration=None) unchanged.
    pub fn update_orders(&mut self) {
        let mut instruments_to_remove = Vec::new();

        for (imnt, orders) in self.orders.iter_mut() {
            let mut updated_orders = Vec::new();

            for order in orders.iter() {
                if let Some(duration) = order.duration {
                    // Time-limited order
                    if duration > 1 {
                        // Create updated order with decreased duration
                        let mut updated_order = order.clone();
                        updated_order.duration = Some(duration - 1);
                        updated_orders.push(updated_order);
                    } else if duration == 1 {
                        // Order expires this turn, remove it
                        log_info!(
                            "{} order: at Price {} has expired and will be removed.",
                            imnt, order.price
                        );
                    }
                    // duration == 0 orders are not added to updated_orders (removed)
                } else {
                    // GTC order remains active indefinitely
                    updated_orders.push(order.clone());
                }
            }

            // Update the orders list for this instrument
            if !updated_orders.is_empty() {
                *orders = updated_orders;
            } else {
                instruments_to_remove.push(imnt.clone());
            }
        }

        // Remove instruments with no active orders
        for imnt in instruments_to_remove {
            self.orders.remove(&imnt);
        }
    }

    /// Aggregate orders from another OrderTicket into this one
    pub fn aggregate_orders(&mut self, other: OrderTicket) {
        for (_imnt, orders) in other.orders {
            for order in orders {
                self.add_order(order);
            }
        }
    }
}

// Unit tests for the Order struct
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::new(
            "BTCUSDT".to_string(),
            50000.0,
            Direction::Long,
            OrderType::Open,
            1.0,
            10,
            Utc::now(),
            Some(5),
        ).unwrap();

        assert_eq!(order.imnt, "BTCUSDT");
        assert_eq!(order.price, 50000.0);
        assert_eq!(order.direction, Direction::Long);
        assert_eq!(order.order_type, OrderType::Open);
        assert_eq!(order.size, 1.0);
        assert_eq!(order.leverage, 10);
        assert!(order.time_created.timestamp() > 0);
        assert_eq!(order.duration, Some(5));
    }

    #[test]
    fn test_order_notional() {
        let order = Order::new(
            "ETHUSDT".to_string(),
            3000.0,
            Direction::Short,
            OrderType::Close,
            2.0,
            5,
            Utc::now(),
            None,
        ).unwrap();

        assert_eq!(order.notional(), 6000.0);
    }

    #[test]
    fn test_order_margin() {
        let order = Order::new(
            "LTCUSDT".to_string(),
            200.0,
            Direction::Long,
            OrderType::Open,
            3.0,
            20,
            Utc::now(),
            None,
        ).unwrap();

        assert_eq!(order.margin(), 30.0);
    }
}
