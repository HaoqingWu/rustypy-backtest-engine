// This file defines the library interface for the Rust project.
// It can export functions and types that can be used by other modules or external applications.

pub mod backtesting;
pub mod python_bindings;
pub mod utils;

// Export the Python module
use pyo3::prelude::*;

#[pymodule]
fn rustypy_backtest_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    // Import and add the PyBacktest class from python_bindings
    m.add_class::<python_bindings::PyBacktest>()?;
    Ok(())
}
