//! Shared test harness for validation layer integration tests.
//!
//! Provides a custom logger that captures ERROR-level validation messages,
//! and a helper to create headless devices with validation enabled.

use std::sync::{Arc, LazyLock, Mutex, Once};

use ash::vk;

use tephra::core::context::ContextConfig;
use tephra::core::device::Device;

/// Collected validation errors from the Vulkan debug callback.
///
/// The `Context` debug callback routes `[VALIDATION]` errors to `log::error!`.
/// This logger captures those messages so tests can assert zero errors.
static VALIDATION_ERRORS: LazyLock<Arc<Mutex<Vec<String>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(Vec::new())));

static INIT_LOGGER: Once = Once::new();

/// A logger that captures ERROR-severity validation messages.
struct ValidationCapture;

impl log::Log for ValidationCapture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Error
    }

    fn log(&self, record: &log::Record<'_>) {
        if record.level() == log::Level::Error {
            let msg = format!("{}", record.args());
            if msg.contains("[VALIDATION]") {
                VALIDATION_ERRORS.lock().unwrap().push(msg);
            }
        }
    }

    fn flush(&self) {}
}

/// Install the validation-capturing logger (idempotent).
pub fn init_capture_logger() {
    INIT_LOGGER.call_once(|| {
        log::set_boxed_logger(Box::new(ValidationCapture))
            .map(|()| log::set_max_level(log::LevelFilter::Error))
            .ok();
    });
}

/// Clear any previously captured validation errors.
pub fn clear_errors() {
    VALIDATION_ERRORS.lock().unwrap().clear();
}

/// Return all captured validation errors since the last `clear_errors()`.
pub fn collected_errors() -> Vec<String> {
    VALIDATION_ERRORS.lock().unwrap().clone()
}

/// Assert that no validation errors have been captured.
///
/// Panics with the list of errors if any are present.
pub fn assert_no_validation_errors() {
    let errors = collected_errors();
    if !errors.is_empty() {
        panic!(
            "Validation errors detected ({}):\n{}",
            errors.len(),
            errors.join("\n")
        );
    }
}

/// Create a headless device with validation layers enabled.
pub fn create_headless_device(name: &str) -> Device {
    let config = ContextConfig {
        app_name: std::ffi::CString::new(name).unwrap(),
        app_version: vk::make_api_version(0, 1, 0, 0),
        enable_validation: true,
        required_instance_extensions: vec![],
    };

    Device::new(&config).expect("failed to create headless device")
}
