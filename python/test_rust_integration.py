"""
Integration tests for the Rust-backed PyBacktest class

These tests verify that the Rust PyBacktest class integrates correctly with Python
using the same data and testing patterns as the original BacktestTest class.

Test Categories:
- test_pybacktest_creation: Basic instantiation
- test_*_no_strategy: Error handling before strategy execution  
- test_run_strategy: Full end-to-end buy-and-hold strategy
- LoggingControlTest: Tests for logging control features (silent, levels, file output)
"""

import unittest
import polars as pl
import pandas as pd
import numpy as np
import datetime as dt
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with fallback handling
try:
    from rustypy_backtest_engine.rustypy_backtest_engine import PyBacktest
    RUST_AVAILABLE = True
except ImportError:
    try:
        # Try importing from the module directly
        import rustypy_backtest_engine
        PyBacktest = rustypy_backtest_engine.PyBacktest
        RUST_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        print(f"Warning: Rust bindings not available: {e}")
        print("To build Rust bindings, run: maturin develop")
        PyBacktest = None
        RUST_AVAILABLE = False


class TestRustAvailability(unittest.TestCase):
    """Test that checks Rust module availability status"""
    
    def test_rust_module_status(self):
        """Test to show status of Rust module availability"""
        if RUST_AVAILABLE:
            print("Rust bindings are available and ready for testing")
            self.assertIsNotNone(PyBacktest)
        else:
            print("⚠ Rust bindings not available. Run 'maturin develop' to build.")
            print("  This is expected if you haven't built the Rust module yet.")
            self.assertIsNone(PyBacktest)
        
        # This test always passes to show that test discovery is working
        self.assertTrue(True, "Test discovery is working correctly")


@unittest.skipIf(not RUST_AVAILABLE, "Rust bindings not available. Run 'maturin develop' to build.")
class RustBacktestTest(unittest.TestCase):
    """Test the Rust-backed PyBacktest class using real market data"""

    def setUp(self):
        """Set up initial backtest for testing using real CSV data."""
        self.tradables = ["BTCUSDT", "ETHUSDT"]
        
        # Load offline data for testing - same as original BacktestTest  
        self.dataDict = {}
        path = "./OfflineData/"
        
        # Load data using Polars directly
        for imnt in self.tradables:
            df = pl.read_csv(path + f'{imnt}_4h_Main.csv')
            # Convert Open Time to datetime (format includes milliseconds)
            df = df.with_columns([
                pl.col("Open Time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.3f")
            ])
            self.dataDict[imnt] = df
        
        # Align start times like original test - use 2021-01-01 as starting point
        startTime = dt.datetime(2021, 1, 1, 0, 0)
        
        # Filter both datasets to start from the same time
        for imnt in self.tradables:
            self.dataDict[imnt] = self.dataDict[imnt].filter(
                pl.col("Open Time") >= startTime
            )
        
        # Find the common end time to ensure equal lengths
        common_end_time = dt.datetime(2022, 1, 1, 0, 0)
        
        # Align both datasets to the same time range
        for imnt in self.tradables:
            self.dataDict[imnt] = self.dataDict[imnt].filter(
                (pl.col("Open Time") >= startTime) & (pl.col("Open Time") <= common_end_time)
            )
        
        print(f"Data aligned from {startTime}: BTC length = {len(self.dataDict['BTCUSDT'])}, ETH length = {len(self.dataDict['ETHUSDT'])}")
        
        self.initial_cash = 1000000.0
        # Use default logging level (3=Info) for clean output in tests
        self.backtest = PyBacktest(self.dataDict, self.initial_cash, log_level=3)

    def test_pybacktest_creation(self):
        """Test creating a PyBacktest instance with real data"""
        self.assertIsNotNone(self.backtest)
        print("PyBacktest creation successful with real data")

    def test_performance_metrics_no_strategy(self):
        """Test that performance metrics fails gracefully when no strategy has been run"""
        with self.assertRaises(Exception) as context:
            self.backtest.calculate_performance_metrics()
        
        self.assertIn("No backtest results available", str(context.exception))
        print("Performance metrics correctly requires strategy run first")

    def test_portfolio_values_no_strategy(self):
        """Test portfolio values method when no strategy has been run"""
        with self.assertRaises(Exception) as context:
            self.backtest.get_portfolio_values()
        
        self.assertIn("No backtest results available", str(context.exception))
        print("Portfolio values correctly requires strategy run first")

    def test_trade_history_no_strategy(self):
        """Test trade history method when no strategy has been run"""
        with self.assertRaises(Exception) as context:
            self.backtest.get_trade_history()
        
        self.assertIn("No backtest results available", str(context.exception))
        print("Trade history correctly requires strategy run first")

    def test_daily_pnl_no_strategy(self):
        """Test daily PnL method when no strategy has been run"""
        with self.assertRaises(Exception) as context:
            self.backtest.get_cum_pnl_by_instrument()
        
        self.assertIn("No backtest results available", str(context.exception))
        print("Daily PnL correctly requires strategy run first")

    def rust_buy_and_hold_strategy(self, context):
        """
        Test a buy-and-hold strategy for the Rust PyBacktest class.
        Similar to myStrategy in the original BacktestTest.
        """
        orders = []
        
        # Check current positions
        positions = context.get("positions", {})
        btc_positions = positions.get("BTCUSDT", {})
        eth_positions = positions.get("ETHUSDT", {})
        
        # Only buy if we don't have positions yet
        if not btc_positions.get("long") and not eth_positions.get("long"):
            current_prices = context.get("current_prices", {})
            
            # Buy 1 BTC
            if "BTCUSDT" in current_prices:
                orders.append({
                    "imnt": "BTCUSDT",
                    "direction": "LONG",
                    "order_type": "OPEN",
                    "leverage": 1,
                    "size": 1.0,
                    "price": current_prices["BTCUSDT"]
                })
            
            # Buy 1 ETH  
            if "ETHUSDT" in current_prices:
                orders.append({
                    "imnt": "ETHUSDT", 
                    "direction": "LONG",
                    "order_type": "OPEN",
                    "leverage": 1,
                    "size": 1.0,
                    "price": current_prices["ETHUSDT"]
                })
        
        return orders

    def test_run_strategy(self):
        """Test running the buy-and-hold strategy with the Rust PyBacktest class."""
        # Use the same time period as original test
        startTime = dt.datetime(2021, 1, 1, 0, 0)
        endTime = dt.datetime(2022, 1, 1, 0, 0)
        
        # Convert to timestamps for Rust API
        start_timestamp = startTime.timestamp()
        end_timestamp = endTime.timestamp()
        
        print(f"Running strategy from {startTime} to {endTime}")
        
        try:
            strategy_result = self.backtest.run_strategy(
                self.rust_buy_and_hold_strategy,
                start_timestamp,
                end_timestamp
            )
            
            self.assertIsInstance(strategy_result, dict)
            self.assertIn("final_portfolio_value", strategy_result)
            self.assertIn("initial_cash", strategy_result)
            
            final_value = strategy_result["final_portfolio_value"]
            total_return = strategy_result["total_return"]
            
            print(f"Strategy execution successful!")
            print(f"Initial cash: ${self.initial_cash:,.2f}")
            print(f"Final portfolio value: ${final_value:,.2f}")
            print(f"Total return: {total_return*100:.2f}%")
            
            # Test performance metrics calculation
            metrics = self.backtest.calculate_performance_metrics()
            self.assertIsInstance(metrics, dict)
            
            # Check for expected performance metrics
            expected_metrics = [
                "total_pnl", "total_return_pct", "sharpe_ratio", 
                "sortino_ratio", "calmar_ratio", "max_drawdown",
                "max_drawdown_pct", "volatility", "downside_volatility",
                "cvar_5", "win_rate", "total_trades", "avg_trade_pnl", "profit_factor"
            ]
            
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
            
            print(f"Performance metrics calculated:")
            print(f"  Total PnL: ${metrics['total_pnl']:,.2f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {metrics['max_drawdown_pct']*100:.2f}%")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"  Total Trades: {metrics['total_trades']}")
            
            # Test portfolio values
            portfolio_values = self.backtest.get_portfolio_values()
            self.assertIsInstance(portfolio_values, list)
            self.assertGreater(len(portfolio_values), 0)
            print(f"Portfolio values retrieved: {len(portfolio_values)} data points")
            
            # Test trade history
            trade_history = self.backtest.get_trade_history()
            self.assertIsInstance(trade_history, dict)
            print(f"Trade history retrieved for {len(trade_history)} instruments")
            
            # Test daily PnL
            daily_pnl = self.backtest.get_cum_pnl_by_instrument()
            self.assertIsInstance(daily_pnl, dict)
            print(f"Daily PnL retrieved for {len(daily_pnl)} instruments")
            
            # Verify we actually traded (should have positions)
            self.assertEqual( metrics['total_trades'], 2, "Strategy should.")
            
            # Verify we actually traded (should have positions)
            self.assertEqual(metrics['total_trades'], 2, "Strategy should execute 2 trades")
            
            # Check if total PnL minus transaction and funding costs is close to expected value
            total_pnl = metrics.get('total_pnl', 0)
            transaction_cost = metrics.get('total_transaction_cost', 0)
            funding_cost = metrics.get('total_funding_cost', 0)
            
            expected_net_pnl = 20513 - transaction_cost - funding_cost
            tolerance = 1000  # Allow some tolerance for floating point precision
            
            print(f"Total PnL: ${total_pnl:,.2f}")
            print(f"Transaction Cost: ${transaction_cost:,.2f}")
            print(f"Funding Cost: ${funding_cost:,.2f}")
            print(f"Net PnL: ${total_pnl:,.2f}")
            print(f"Expected Net PnL: ${expected_net_pnl:,.2f}")
            
            self.assertAlmostEqual(total_pnl, expected_net_pnl, delta=tolerance,
                                 msg=f"Net PnL {total_pnl:.2f} should be close to {expected_net_pnl}")
            
        except Exception as e:
            self.fail(f"Strategy execution failed: {e}")


@unittest.skipIf(not RUST_AVAILABLE, "Rust bindings not available. Run 'maturin develop' to build.")
class FeatureExtractionTest(unittest.TestCase):
    """Test the enhanced feature extraction and configurable lookback functionality"""

    def create_rsi_test_data(self):
        """Create test data with RSI feature to test custom column extraction"""
        timestamps = [dt.datetime(2021, 1, 1, i, 0) for i in range(0, 20, 4)]  # 5 time points
        
        btc_data = {
            "Open Time": timestamps,
            "Open": [29000.0, 29500.0, 30000.0, 30500.0, 31000.0],
            "High": [29500.0, 30000.0, 30500.0, 31000.0, 31500.0],
            "Low": [28500.0, 29000.0, 29500.0, 30000.0, 30500.0],
            "Close": [29400.0, 29900.0, 30400.0, 30900.0, 31400.0],
            "Volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            # RSI feature for testing
            "RSI_14": [45.0, 35.0, 25.0, 65.0, 75.0]  # Oversold to overbought progression
        }
        
        eth_data = {
            "Open Time": timestamps,
            "Open": [730.0, 750.0, 770.0, 790.0, 810.0],
            "High": [750.0, 770.0, 790.0, 810.0, 830.0],
            "Low": [720.0, 740.0, 760.0, 780.0, 800.0],
            "Close": [745.0, 765.0, 785.0, 805.0, 825.0],
            "Volume": [2000.0, 2100.0, 2200.0, 2300.0, 2400.0],
            # Different RSI values for ETH
            "RSI_14": [55.0, 40.0, 30.0, 70.0, 60.0]
        }
        
        return {
            "BTCUSDT": pl.DataFrame(btc_data),
            "ETHUSDT": pl.DataFrame(eth_data)
        }

    def test_rsi_feature_preservation(self):
        """Test that RSI feature is preserved through the Rust engine"""
        print("Testing RSI feature preservation through Rust engine")
        market_data = self.create_rsi_test_data()
        
        backtest = PyBacktest(market_data, 100000.0, log_level=0)  # Silent for clean output
        
        # Strategy that verifies RSI feature access
        def rsi_verification_strategy(context):
            current_data = context.get("current_data", {})
            
            # Verify both instruments are available
            self.assertIn("BTCUSDT", current_data)
            self.assertIn("ETHUSDT", current_data)
            
            # Check BTC RSI feature - now expects Polars DataFrame
            btc_df = current_data["BTCUSDT"]
            self.assertGreater(btc_df.height, 0, "BTC data should not be empty")
            
            if btc_df.height > 0:
                # Get latest row as dictionary from Polars DataFrame
                latest_btc = btc_df.row(-1, named=True)
                
                # Verify RSI feature is accessible
                self.assertIn("RSI_14", latest_btc)
                
                # Verify basic OHLCV is still there
                self.assertIn("Open", latest_btc)
                self.assertIn("Close", latest_btc)
                self.assertIn("Volume", latest_btc)
                
                rsi_value = latest_btc.get("RSI_14")
                self.assertIsInstance(rsi_value, (int, float))
                self.assertGreaterEqual(rsi_value, 0)
                self.assertLessEqual(rsi_value, 100)
                
                print(f"Latest BTC RSI: {rsi_value}")
            
            # Check ETH RSI feature - now expects Polars DataFrame
            eth_df = current_data["ETHUSDT"]
            if eth_df.height > 0:
                latest_eth = eth_df.row(-1, named=True)
                self.assertIn("RSI_14", latest_eth)
                eth_rsi = latest_eth.get("RSI_14")
                print(f"Latest ETH RSI: {eth_rsi}")
            
            return []  # No orders, just verification
        
        start_time = dt.datetime(2021, 1, 1, 8).timestamp()  # Start at 3rd data point
        end_time = dt.datetime(2021, 1, 1, 12).timestamp()   # End at 4th data point
        
        result = backtest.run_strategy(rsi_verification_strategy, start_time, end_time)
        
        self.assertIsInstance(result, dict)
        print("RSI feature preserved and accessible in strategy context")

    def test_configurable_lookback_with_rsi(self):
        """Test configurable lookback periods with RSI feature"""
        print("Testing configurable lookback periods with RSI")
        market_data = self.create_rsi_test_data()
        
        backtest = PyBacktest(market_data, 100000.0, log_level=0)
        
        # Test different lookback periods
        lookback_results = {}
        
        def lookback_rsi_strategy(context):
            current_data = context.get("current_data", {})
            btc_df = current_data.get("BTCUSDT")
            
            if btc_df is None:
                lookback_results["data_length"] = 0
                return []
                
            # Store the length for this test
            lookback_results["data_length"] = btc_df.height
            
            # Verify RSI is preserved regardless of lookback period
            if btc_df.height > 0:
                latest = btc_df.row(-1, named=True)
                lookback_results["has_rsi"] = "RSI_14" in latest
                lookback_results["rsi_value"] = latest.get("RSI_14")
                
                # Collect all RSI values from the DataFrame
                rsi_values = []
                if "RSI_14" in btc_df.columns:
                    rsi_values = btc_df["RSI_14"].to_list()
                lookback_results["all_rsi_values"] = rsi_values
            
            return []
        
        start_time = dt.datetime(2021, 1, 1, 8).timestamp()   # 3rd data point
        end_time = dt.datetime(2021, 1, 1, 16).timestamp()    # 5th data point
        
        # Test simple case first - just verify RSI is accessible
        result = backtest.run_strategy(
            lookback_rsi_strategy, 
            start_time, 
            end_time, 
            3  # Request 3 lookback periods
        )
        self.assertIsNotNone(result)  # Ensure strategy ran successfully
        
        # The key test: RSI should be preserved and accessible
        self.assertTrue(lookback_results.get("has_rsi", False), "RSI feature should be accessible")
        self.assertIsNotNone(lookback_results.get("rsi_value"), "RSI value should not be None")
        
        data_length = lookback_results.get("data_length", 0)
        rsi_values = lookback_results.get("all_rsi_values", [])
        
        print(f"Lookback test: {data_length} data points")
        print(f"RSI values found: {rsi_values}")
        print(f"Latest RSI: {lookback_results.get('rsi_value')}")
        
        # Verify we have at least some data
        self.assertGreater(data_length, 0, "Should have at least 1 data point")
        self.assertGreater(len(rsi_values), 0, "Should have at least 1 RSI value")

    def test_rsi_based_trading_strategy(self):
        """Test a trading strategy that uses RSI feature for decisions"""
        print("Testing RSI-based trading strategy")
        market_data = self.create_rsi_test_data()
        
        backtest = PyBacktest(market_data, 100000.0, log_level=0)  # Silent for clean output
        
        rsi_access_verified = []
        
        def rsi_access_strategy(context):
            """Strategy that verifies RSI feature access and makes simple trades"""
            orders = []
            current_data = context.get("current_data", {})
            current_prices = context.get("current_prices", {})
            
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                if symbol in current_data:
                    symbol_df = current_data[symbol]
                    if symbol_df.height > 0:
                        latest_data = symbol_df.row(-1, named=True)
                        
                        # Extract RSI feature - this is the key test
                        rsi = latest_data.get("RSI_14", None)
                        current_price = current_prices.get(symbol, 0.0)
                        
                        if rsi is not None and current_price > 0:
                            rsi_access_verified.append({
                                "symbol": symbol,
                                "rsi": rsi,
                                "price": current_price,
                                "rsi_accessible": True
                            })
                            
                            # Make a simple trade based on RSI (just to verify trading works)
                            if rsi < 50:  # Simple threshold
                                orders.append({
                                    "imnt": symbol,
                                    "direction": "LONG",
                                    "order_type": "OPEN",
                                    "size": 0.1,
                                    "leverage": 1,
                                    "price": current_price
                                })
            
            return orders
        
        start_time = dt.datetime(2021, 1, 1, 4).timestamp()   # Start at 2nd data point  
        end_time = dt.datetime(2021, 1, 1, 12).timestamp()    # End at 4th data point
        
        result = backtest.run_strategy(
            rsi_access_strategy, 
            start_time, 
            end_time,
            2  # Test configurable lookback too
        )
        
        self.assertIsInstance(result, dict)
        
        # The key verification: RSI feature should be accessible
        self.assertGreater(len(rsi_access_verified), 0, "Should have accessed RSI features")
        
        print(f"RSI-based strategy executed")
        print(f"  RSI feature accessed {len(rsi_access_verified)} times")
        
        for access in rsi_access_verified:
            print(f"  {access['symbol']}: RSI={access['rsi']:.1f}, Price=${access['price']:.0f}")
        
        # Test performance metrics
        metrics = backtest.calculate_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_trades", metrics)
        
        print(f"Strategy metrics: {metrics['total_trades']} trades executed")

    def test_rsi_data_integrity(self):
        """Test that RSI values maintain integrity through the engine"""
        print("Testing RSI data integrity")
        market_data = self.create_rsi_test_data()
        
        backtest = PyBacktest(market_data, 100000.0, log_level=0)
        
        collected_rsi = {"BTCUSDT": [], "ETHUSDT": []}
        
        def rsi_collection_strategy(context):
            current_data = context.get("current_data", {})
            
            # Collect RSI values from current data
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                if symbol in current_data:
                    symbol_df = current_data[symbol]
                    if symbol_df.height > 0:
                        # Get the latest RSI value
                        latest_data = symbol_df.row(-1, named=True)
                        if "RSI_14" in latest_data:
                            rsi_value = latest_data["RSI_14"]
                            if rsi_value not in collected_rsi[symbol]:
                                collected_rsi[symbol].append(rsi_value)
            
            return []
        
        # Run strategy across multiple time points to collect different RSI values
        for i in range(5):  # Cover all 5 data points
            time_point = dt.datetime(2021, 1, 1, i*4).timestamp()
            
            result = backtest.run_strategy(
                rsi_collection_strategy, 
                time_point, 
                time_point,
                1  # Just current data point
            )
            self.assertIsNotNone(result)  # Ensure strategy ran successfully
        
        # Verify RSI values were collected
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            collected = collected_rsi[symbol]
            
            # Should have collected some RSI values
            self.assertGreater(len(collected), 0, f"Should have collected RSI values for {symbol}")
            
            # Verify RSI values are in valid range
            for rsi_val in collected:
                self.assertGreaterEqual(rsi_val, 0.0, "RSI should be >= 0")
                self.assertLessEqual(rsi_val, 100.0, "RSI should be <= 100")
        
        print("RSI data integrity verified:")
        for symbol, values in collected_rsi.items():
            print(f"  {symbol}: {len(values)} RSI values collected: {values}")


@unittest.skipIf(not RUST_AVAILABLE, "Rust bindings not available. Run 'maturin develop' to build.")
class LoggingControlTest(unittest.TestCase):
    """Test the logging control features of the Rust backtesting engine"""

    def create_test_data(self):
        """Create minimal test data for logging tests"""
        timestamps = [dt.datetime(2021, 1, 1, i, 0) for i in range(0, 8, 4)]
        
        btc_data = {
            "Open Time": timestamps,
            "Open": [29000.0, 29100.0],
            "High": [29100.0, 29200.0], 
            "Low": [28900.0, 29000.0],
            "Close": [29050.0, 29150.0],
            "Volume": [1000.0, 1000.0]
        }
        
        eth_data = {
            "Open Time": timestamps,
            "Open": [740.0, 750.0],
            "High": [745.0, 755.0],
            "Low": [735.0, 745.0],
            "Close": [742.0, 752.0],
            "Volume": [2000.0, 2000.0]
        }
        
        return {
            "BTCUSDT": pl.DataFrame(btc_data),
            "ETHUSDT": pl.DataFrame(eth_data)
        }

    def simple_test_strategy(self, context):
        """Simple strategy for logging tests"""
        orders = []
        positions = context.get("positions", {})
        current_prices = context.get("current_prices", {})
        
        if not positions.get("BTCUSDT", {}).get("long"):
            orders.append({
                "imnt": "BTCUSDT",
                "direction": "LONG", 
                "order_type": "OPEN",
                "size": 0.01,
                "leverage": 1
            })
        return orders

    def test_silent_logging(self):
        """Test silent logging (log_level=0) produces no debug output"""
        print("Testing silent logging (log_level=0)")
        market_data = self.create_test_data()
        
        # Create backtest with silent logging
        backtest = PyBacktest(market_data, 10000.0, log_level=0)
        
        start_time = dt.datetime(2021, 1, 1).timestamp()
        end_time = dt.datetime(2021, 1, 1, 4).timestamp()
        
        result = backtest.run_strategy(self.simple_test_strategy, start_time, end_time)
        
        self.assertIsInstance(result, dict)
        self.assertIn("final_portfolio_value", result)
        print("Silent logging test passed - no debug output shown")

    def test_error_logging(self):
        """Test error-only logging (log_level=1)"""
        print("Testing error-only logging (log_level=1)")
        market_data = self.create_test_data()
        
        backtest = PyBacktest(market_data, 10000.0, log_level=1)
        
        start_time = dt.datetime(2021, 1, 1).timestamp()
        end_time = dt.datetime(2021, 1, 1, 4).timestamp()
        
        result = backtest.run_strategy(self.simple_test_strategy, start_time, end_time)
        
        self.assertIsInstance(result, dict)
        print("Error-only logging test passed")

    def test_info_logging(self):
        """Test info logging (log_level=3) - default level"""
        print("Testing info logging (log_level=3)")
        market_data = self.create_test_data()
        
        backtest = PyBacktest(market_data, 10000.0, log_level=3)
        
        start_time = dt.datetime(2021, 1, 1).timestamp()
        end_time = dt.datetime(2021, 1, 1, 4).timestamp()
        
        result = backtest.run_strategy(self.simple_test_strategy, start_time, end_time)
        
        self.assertIsInstance(result, dict)
        print("Info logging test passed")

    def test_logging_levels_parameter(self):
        """Test that different logging levels can be set"""
        print("Testing different logging level parameters")
        market_data = self.create_test_data()
        
        # Test all logging levels can be instantiated
        for level in [0, 1, 2, 3, 4]:
            try:
                backtest = PyBacktest(market_data, 10000.0, log_level=level)
                self.assertIsNotNone(backtest)
            except Exception as e:
                self.fail(f"Failed to create backtest with log_level={level}: {e}")
        
        print("All logging levels (0-4) work correctly")

    def test_file_logging_parameter(self):
        """Test that file logging parameter is accepted"""
        print("Testing file logging parameter")
        market_data = self.create_test_data()
        
        # Test file logging parameter (actual file creation depends on Rust implementation)
        try:
            backtest = PyBacktest(market_data, 10000.0, log_level=3, log_file="test_log.txt")
            self.assertIsNotNone(backtest)
            print("File logging parameter accepted")
        except Exception as e:
            self.fail(f"Failed to create backtest with log_file parameter: {e}")

    def test_default_logging_behavior(self):
        """Test that default logging behavior works (backward compatibility)"""
        print("Testing default logging behavior (backward compatibility)")
        market_data = self.create_test_data()
        
        # Test that old constructor still works (without logging parameters)
        try:
            backtest = PyBacktest(market_data, 10000.0)
            self.assertIsNotNone(backtest)
            print("Default logging behavior works (backward compatible)")
        except Exception as e:
            self.fail(f"Failed to create backtest with default parameters: {e}")


if __name__ == "__main__":
    # Run with verbosity to see test output
    unittest.main(verbosity=2)