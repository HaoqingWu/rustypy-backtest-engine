// This file defines utility types and constants that are used throughout the project.

#[derive(Debug, Clone, PartialEq)]
pub enum TradeStatus {
    Open,
    Closed,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Direction {
    Long,
    Short,
}

/// Whether an order is to open or close a trade
#[derive(Debug, Clone, PartialEq)]
pub enum OrderType {
    Open,
    Close,
}
