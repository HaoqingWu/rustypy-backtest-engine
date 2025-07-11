use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::{log_info, log_debug};

#[derive(Debug, Default)]
pub struct BacktestProfiler {
    pub timings: HashMap<String, Vec<Duration>>,
    pub counters: HashMap<String, u64>,
    pub enabled: bool,
}

impl BacktestProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            timings: HashMap::new(),
            counters: HashMap::new(),
            enabled,
        }
    }

    pub fn start_timer(&self, label: &str) -> ProfileTimer {
        ProfileTimer::new(label.to_string(), self.enabled)
    }

    pub fn record_duration(&mut self, label: String, duration: Duration) {
        if !self.enabled {
            return;
        }
        self.timings.entry(label).or_default().push(duration);
    }

    pub fn increment_counter(&mut self, label: &str) {
        if !self.enabled {
            return;
        }
        *self.counters.entry(label.to_string()).or_insert(0) += 1;
    }

    pub fn get_summary(&self) -> ProfileSummary {
        let mut summary = ProfileSummary::default();
        
        for (label, durations) in &self.timings {
            let total_duration: Duration = durations.iter().sum();
            let avg_duration = if !durations.is_empty() {
                total_duration / durations.len() as u32
            } else {
                Duration::ZERO
            };
            let max_duration = durations.iter().max().copied().unwrap_or(Duration::ZERO);
            let min_duration = durations.iter().min().copied().unwrap_or(Duration::ZERO);
            
            summary.operation_stats.insert(label.clone(), OperationStats {
                total_duration,
                avg_duration,
                max_duration,
                min_duration,
                call_count: durations.len(),
            });
        }
        
        summary.counters = self.counters.clone();
        summary
    }

    pub fn print_summary(&self) {
        let summary = self.get_summary();
        
        log_info!("=== BACKTEST PERFORMANCE PROFILE ===");
        
        // Sort operations by total time spent
        let mut ops: Vec<_> = summary.operation_stats.iter().collect();
        ops.sort_by(|a, b| b.1.total_duration.cmp(&a.1.total_duration));
        
        log_info!("{:<30} {:>12} {:>12} {:>12} {:>12} {:>8}", 
                  "Operation", "Total (ms)", "Avg (ms)", "Max (ms)", "Min (ms)", "Calls");
        log_info!("{}", "-".repeat(90));
        
        for (operation, stats) in ops {
            log_info!("{:<30} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>8}",
                operation,
                stats.total_duration.as_millis() as f64,
                stats.avg_duration.as_millis() as f64,
                stats.max_duration.as_millis() as f64,
                stats.min_duration.as_millis() as f64,
                stats.call_count
            );
        }
        
        log_info!("");
        log_info!("=== COUNTERS ===");
        for (counter, value) in &summary.counters {
            log_info!("{:<30} {:>12}", counter, value);
        }
        
        // Calculate percentages
        if let Some(total_runtime) = summary.operation_stats.get("total_backtest_runtime") {
            log_info!("");
            log_info!("=== TIME BREAKDOWN ===");
            for (operation, stats) in &summary.operation_stats {
                if operation != "total_backtest_runtime" {
                    let percentage = (stats.total_duration.as_nanos() as f64 / 
                                    total_runtime.total_duration.as_nanos() as f64) * 100.0;
                    log_info!("{:<30} {:>8.1}%", operation, percentage);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct ProfileTimer {
    label: String,
    start: Instant,
    enabled: bool,
}

impl ProfileTimer {
    fn new(label: String, enabled: bool) -> Self {
        Self {
            label,
            start: Instant::now(),
            enabled,
        }
    }

    pub fn finish(self, profiler: &mut BacktestProfiler) {
        if !self.enabled {
            return;
        }
        let duration = self.start.elapsed();
        profiler.record_duration(self.label, duration);
    }
}

#[derive(Debug, Default)]
pub struct ProfileSummary {
    pub operation_stats: HashMap<String, OperationStats>,
    pub counters: HashMap<String, u64>,
}

#[derive(Debug)]
pub struct OperationStats {
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub max_duration: Duration,
    pub min_duration: Duration,
    pub call_count: usize,
}

// Macro for easy profiling
#[macro_export]
macro_rules! profile_block {
    ($profiler:expr, $label:expr, $block:block) => {
        {
            let timer = $profiler.start_timer($label);
            let result = $block;
            timer.finish($profiler);
            result
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_basic_functionality() {
        let mut profiler = BacktestProfiler::new(true);
        
        // Test timer
        {
            let timer = profiler.start_timer("test_operation");
            thread::sleep(Duration::from_millis(10));
            timer.finish(&mut profiler);
        }
        
        // Test counter
        profiler.increment_counter("test_counter");
        profiler.increment_counter("test_counter");
        
        let summary = profiler.get_summary();
        
        assert_eq!(summary.operation_stats.len(), 1);
        assert!(summary.operation_stats.contains_key("test_operation"));
        assert_eq!(summary.counters.get("test_counter"), Some(&2));
        
        let stats = &summary.operation_stats["test_operation"];
        assert_eq!(stats.call_count, 1);
        assert!(stats.total_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_disabled() {
        let mut profiler = BacktestProfiler::new(false);
        
        let timer = profiler.start_timer("should_not_record");
        timer.finish(&mut profiler);
        profiler.increment_counter("should_not_count");
        
        let summary = profiler.get_summary();
        assert!(summary.operation_stats.is_empty());
        assert!(summary.counters.is_empty());
    }

    #[test]
    fn test_profile_block_macro() {
        let mut profiler = BacktestProfiler::new(true);
        
        let result = profile_block!(&mut profiler, "macro_test", {
            thread::sleep(Duration::from_millis(5));
            42
        });
        
        assert_eq!(result, 42);
        
        let summary = profiler.get_summary();
        assert!(summary.operation_stats.contains_key("macro_test"));
        assert!(summary.operation_stats["macro_test"].total_duration >= Duration::from_millis(5));
    }
}