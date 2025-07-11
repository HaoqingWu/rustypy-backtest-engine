use std::sync::{Arc, Mutex};
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Utc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Silent = 0,  // No output
    Error = 1,   // Only errors
    Warn = 2,    // Errors and warnings
    Info = 3,    // Info, warnings, and errors
    Debug = 4,   // All output including debug
}

impl From<u8> for LogLevel {
    fn from(level: u8) -> Self {
        match level {
            0 => LogLevel::Silent,
            1 => LogLevel::Error,
            2 => LogLevel::Warn,
            3 => LogLevel::Info,
            4 => LogLevel::Debug,
            _ => LogLevel::Info, // Default to Info for invalid values
        }
    }
}

impl From<&str> for LogLevel {
    fn from(level: &str) -> Self {
        match level.to_lowercase().as_str() {
            "silent" => LogLevel::Silent,
            "error" => LogLevel::Error,
            "warn" | "warning" => LogLevel::Warn,
            "info" => LogLevel::Info,
            "debug" => LogLevel::Debug,
            _ => LogLevel::Info, // Default to Info for invalid strings
        }
    }
}

#[derive(Debug)]
pub struct Logger {
    level: LogLevel,
    log_file: Option<Arc<Mutex<std::fs::File>>>,
}

impl Logger {
    pub fn new(level: LogLevel, log_file_path: Option<&str>) -> Result<Self, String> {
        let log_file = if let Some(path) = log_file_path {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("Failed to open log file '{}': {}", path, e))?;
            Some(Arc::new(Mutex::new(file)))
        } else {
            None
        };

        Ok(Logger {
            level,
            log_file,
        })
    }

    pub fn set_level(&mut self, level: LogLevel) {
        self.level = level;
    }

    pub fn error(&self, message: &str) {
        if self.level >= LogLevel::Error {
            self.write_log("ERROR", message);
        }
    }

    pub fn warn(&self, message: &str) {
        if self.level >= LogLevel::Warn {
            self.write_log("WARN", message);
        }
    }

    pub fn info(&self, message: &str) {
        if self.level >= LogLevel::Info {
            self.write_log("INFO", message);
        }
    }

    pub fn debug(&self, message: &str) {
        if self.level >= LogLevel::Debug {
            self.write_log("DEBUG", message);
        }
    }

    fn write_log(&self, level_str: &str, message: &str) {
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let log_line = format!("[{}] {}: {}", timestamp, level_str, message);

        // Write to file if configured
        if let Some(ref file_mutex) = self.log_file {
            if let Ok(mut file) = file_mutex.lock() {
                let _ = writeln!(file, "{}", log_line);
                let _ = file.flush();
            }
        } else {
            // Only write to console if no log file is configured
            println!("{}", log_line);
        }
    }
}

// Global logger instance
static GLOBAL_LOGGER: Mutex<Option<Logger>> = Mutex::new(None);

pub fn init_logger(level: LogLevel, log_file_path: Option<&str>) -> Result<(), String> {
    let logger = Logger::new(level, log_file_path).unwrap_or_else(|_| {
        Logger::new(LogLevel::Info, None).expect("Failed to create default logger")
    });
    
    // Always replace the global logger with the new one
    if let Ok(mut global_logger) = GLOBAL_LOGGER.lock() {
        *global_logger = Some(logger);
        Ok(())
    } else {
        Err("Failed to acquire logger lock".to_string())
    }
}

pub fn get_logger() -> &'static Mutex<Option<Logger>> {
    &GLOBAL_LOGGER
}

pub fn set_log_level(level: LogLevel) {
    if let Ok(mut logger_opt) = get_logger().lock() {
        if let Some(ref mut logger) = logger_opt.as_mut() {
            logger.set_level(level);
        }
    }
}

// Convenience macros for logging
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        if let Ok(logger_opt) = $crate::utils::logger::get_logger().lock() {
            if let Some(ref logger) = logger_opt.as_ref() {
                logger.error(&format!($($arg)*));
            }
        }
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        if let Ok(logger_opt) = $crate::utils::logger::get_logger().lock() {
            if let Some(ref logger) = logger_opt.as_ref() {
                logger.warn(&format!($($arg)*));
            }
        }
    };
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        if let Ok(logger_opt) = $crate::utils::logger::get_logger().lock() {
            if let Some(ref logger) = logger_opt.as_ref() {
                logger.info(&format!($($arg)*));
            }
        }
    };
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        if let Ok(logger_opt) = $crate::utils::logger::get_logger().lock() {
            if let Some(ref logger) = logger_opt.as_ref() {
                logger.debug(&format!($($arg)*));
            }
        }
    };
}