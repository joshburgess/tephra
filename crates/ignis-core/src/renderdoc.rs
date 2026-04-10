//! RenderDoc in-application capture API.
//!
//! Provides runtime loading of the RenderDoc shared library and programmatic
//! frame capture control. This is useful for triggering captures from code
//! without needing to launch through the RenderDoc UI.
//!
//! RenderDoc must already be injected into the process (e.g., by launching
//! through RenderDoc) or the shared library must be loadable from the system
//! library path.
//!
//! # Platform support
//!
//! - **Linux:** loads `librenderdoc.so`
//! - **Windows:** loads `renderdoc.dll`
//! - **macOS:** not supported (RenderDoc does not support macOS/MoltenVK)

use std::ffi::c_void;
use std::sync::OnceLock;

use log::{debug, info};

/// RenderDoc API version 1.6.0 (latest stable).
#[cfg(any(target_os = "linux", target_os = "windows"))]
const RENDERDOC_API_VERSION: i32 = 10600;

// Function pointer types matching renderdoc_app.h signatures.
#[cfg(any(target_os = "linux", target_os = "windows"))]
type GetApiFn = unsafe extern "C" fn(version: i32, out: *mut *mut c_void) -> i32;
type StartFrameCaptureFn = unsafe extern "C" fn(device: *mut c_void, window: *mut c_void);
type EndFrameCaptureFn = unsafe extern "C" fn(device: *mut c_void, window: *mut c_void);
type IsFrameCapturingFn = unsafe extern "C" fn() -> u32;
type TriggerCaptureFn = unsafe extern "C" fn();
type TriggerMultiFrameCaptureFn = unsafe extern "C" fn(num_frames: u32);
type SetCaptureFilePathTemplateFn = unsafe extern "C" fn(path: *const std::ffi::c_char);
type GetCaptureFilePathTemplateFn = unsafe extern "C" fn() -> *const std::ffi::c_char;
type GetNumCapturesFn = unsafe extern "C" fn() -> u32;
type SetCaptureOptionU32Fn = unsafe extern "C" fn(option: i32, value: u32) -> i32;
type LaunchReplayUIFn = unsafe extern "C" fn(connect: u32, cmd: *const std::ffi::c_char) -> u32;

/// Offsets into the RENDERDOC_API_1_6_0 vtable (function pointer indices).
/// These match the order in renderdoc_app.h's RENDERDOC_API_1_6_0 struct.
#[allow(dead_code)]
mod offsets {
    // Each entry is a function pointer (8 bytes on 64-bit).
    pub const GET_API_VERSION: usize = 0;
    pub const SET_CAPTURE_OPTION_U32: usize = 1;
    pub const SET_CAPTURE_OPTION_F32: usize = 2;
    pub const GET_CAPTURE_OPTION_U32: usize = 3;
    pub const GET_CAPTURE_OPTION_F32: usize = 4;
    pub const SET_FOCUS_TOGGLE_KEYS: usize = 5;
    pub const SET_CAPTURE_KEYS: usize = 6;
    pub const GET_OVERLAY_BITS: usize = 7;
    pub const MASK_OVERLAY_BITS: usize = 8;
    pub const REMOVE_HOOKS: usize = 9;
    pub const UNLOAD_CRASH_HANDLER: usize = 10;
    pub const SET_CAPTURE_FILE_PATH_TEMPLATE: usize = 11;
    pub const GET_CAPTURE_FILE_PATH_TEMPLATE: usize = 12;
    pub const GET_NUM_CAPTURES: usize = 13;
    pub const GET_CAPTURE: usize = 14;
    pub const TRIGGER_CAPTURE: usize = 15;
    pub const IS_TARGET_CONTROL_CONNECTED: usize = 16;
    pub const LAUNCH_REPLAY_UI: usize = 17;
    pub const SET_ACTIVE_WINDOW: usize = 18;
    pub const START_FRAME_CAPTURE: usize = 19;
    pub const IS_FRAME_CAPTURING: usize = 20;
    pub const END_FRAME_CAPTURE: usize = 21;
    pub const TRIGGER_MULTI_FRAME_CAPTURE: usize = 22;
    // 1.1.2+
    pub const SET_CAPTURE_FILE_COMMENTS: usize = 23;
    // 1.4.0+
    pub const DISCARD_FRAME_CAPTURE: usize = 24;
    // 1.5.0+
    pub const SHOW_REPLAY_UI: usize = 25;
    // 1.6.0+
    pub const SET_CAPTURE_TITLE: usize = 26;
}

/// Capture options matching `CYCLER_CaptureOption` in renderdoc_app.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CaptureOption {
    /// Allow Vulkan full-screen exclusive mode.
    AllowFullscreen = 0,
    /// Allow VSync in the application.
    AllowVSync = 1,
    /// Capture API validation/debug layer messages.
    ApiValidation = 2,
    /// Capture all command lists immediately (D3D11 specific).
    CaptureAllCmdLists = 3,
    /// Capture callstacks for API calls.
    CaptureCallstacks = 4,
    /// Only capture callstacks for draw/dispatch calls.
    CaptureCallstacksOnlyDraws = 5,
    /// Delay the backend (seconds) after a capture is made.
    DelayForDebugger = 6,
    /// Verify buffer access (D3D11/12 specific).
    VerifyBufferAccess = 7,
    /// Hook into children of the process.
    HookIntoChildren = 8,
    /// Reference all resources, not just those used during capture.
    RefAllResources = 9,
    /// Save all initial contents of all resources at start of capture.
    SaveAllInitials = 10,
    /// Capture CPU counters during capture.
    CaptureAllCmdLists2 = 11,
    /// Debug output mute.
    DebugOutputMute = 12,
    /// Allow unsupported vendor extensions.
    AllowUnsupportedVendorExtensions = 13,
    /// Soft-memory limit for capture (MB).
    SoftMemoryLimit = 14,
}

/// RenderDoc capture API handle.
///
/// Provides methods for programmatic frame capture control. Obtain via
/// [`RenderDoc::load()`].
pub struct RenderDoc {
    /// Pointer to the RenderDoc API vtable.
    api: *mut c_void,
}

// SAFETY: The RenderDoc API is thread-safe per its documentation.
// All API functions can be called from any thread.
unsafe impl Send for RenderDoc {}
// SAFETY: The RenderDoc API is thread-safe per its documentation.
unsafe impl Sync for RenderDoc {}

/// Global singleton for the loaded RenderDoc API.
static RENDERDOC: OnceLock<Option<RenderDoc>> = OnceLock::new();

impl RenderDoc {
    /// Attempt to load the RenderDoc API.
    ///
    /// Returns `Some` if RenderDoc is injected into the process or the shared
    /// library is found on the system library path. Returns `None` if
    /// RenderDoc is not available.
    ///
    /// This caches the result — subsequent calls return the same handle.
    pub fn load() -> Option<&'static RenderDoc> {
        RENDERDOC
            .get_or_init(|| {
                let api = load_renderdoc_api();
                match api {
                    Some(ptr) => {
                        info!("RenderDoc API loaded successfully");
                        Some(RenderDoc { api: ptr })
                    }
                    None => {
                        debug!("RenderDoc API not available");
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Start a frame capture.
    ///
    /// Pass `std::ptr::null_mut()` for both parameters to capture the next
    /// frame on any device/window. For Vulkan, `device` should be the
    /// `VkInstance` cast to `*mut c_void`.
    ///
    /// # Safety
    ///
    /// `device` and `window` must be valid handles or null pointers.
    pub unsafe fn start_frame_capture(&self, device: *mut c_void, window: *mut c_void) {
        // SAFETY: api is a valid RenderDoc API pointer, function offset is correct.
        // Caller guarantees device/window validity.
        unsafe {
            let fp = self.get_fn::<StartFrameCaptureFn>(offsets::START_FRAME_CAPTURE);
            fp(device, window);
        }
        debug!("RenderDoc: frame capture started");
    }

    /// End a frame capture.
    ///
    /// Must be paired with a prior [`start_frame_capture`](Self::start_frame_capture) call.
    ///
    /// # Safety
    ///
    /// `device` and `window` must match the values passed to `start_frame_capture`.
    pub unsafe fn end_frame_capture(&self, device: *mut c_void, window: *mut c_void) {
        // SAFETY: api is a valid RenderDoc API pointer, function offset is correct.
        // Caller guarantees device/window validity.
        unsafe {
            let fp = self.get_fn::<EndFrameCaptureFn>(offsets::END_FRAME_CAPTURE);
            fp(device, window);
        }
        debug!("RenderDoc: frame capture ended");
    }

    /// Returns `true` if a frame capture is currently in progress.
    pub fn is_frame_capturing(&self) -> bool {
        // SAFETY: api is a valid RenderDoc API pointer, function offset is correct.
        unsafe {
            let fp = self.get_fn::<IsFrameCapturingFn>(offsets::IS_FRAME_CAPTURING);
            fp() != 0
        }
    }

    /// Trigger a single-frame capture on the next present.
    pub fn trigger_capture(&self) {
        // SAFETY: api is a valid RenderDoc API pointer, function offset is correct.
        unsafe {
            let fp = self.get_fn::<TriggerCaptureFn>(offsets::TRIGGER_CAPTURE);
            fp();
        }
        debug!("RenderDoc: capture triggered");
    }

    /// Trigger a multi-frame capture starting on the next present.
    pub fn trigger_multi_frame_capture(&self, num_frames: u32) {
        // SAFETY: api is a valid RenderDoc API pointer, function offset is correct.
        unsafe {
            let fp =
                self.get_fn::<TriggerMultiFrameCaptureFn>(offsets::TRIGGER_MULTI_FRAME_CAPTURE);
            fp(num_frames);
        }
        debug!("RenderDoc: {num_frames}-frame capture triggered");
    }

    /// Set the file path template for captures (without extension).
    ///
    /// For example, `"/tmp/my_capture"` will produce files like
    /// `/tmp/my_capture_frame123.rdc`.
    pub fn set_capture_file_path_template(&self, path: &str) {
        let c_path = std::ffi::CString::new(path).expect("path contains null byte");
        // SAFETY: api is a valid RenderDoc API pointer, c_path is a valid C string.
        unsafe {
            let fp = self
                .get_fn::<SetCaptureFilePathTemplateFn>(offsets::SET_CAPTURE_FILE_PATH_TEMPLATE);
            fp(c_path.as_ptr());
        }
    }

    /// Get the current file path template for captures.
    pub fn get_capture_file_path_template(&self) -> &str {
        // SAFETY: api is a valid RenderDoc API pointer, returned pointer is valid
        // for the lifetime of the RenderDoc API (static).
        unsafe {
            let fp = self
                .get_fn::<GetCaptureFilePathTemplateFn>(offsets::GET_CAPTURE_FILE_PATH_TEMPLATE);
            let ptr = fp();
            if ptr.is_null() {
                ""
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or_default()
            }
        }
    }

    /// Get the number of captures that have been made.
    pub fn get_num_captures(&self) -> u32 {
        // SAFETY: api is a valid RenderDoc API pointer, function offset is correct.
        unsafe {
            let fp = self.get_fn::<GetNumCapturesFn>(offsets::GET_NUM_CAPTURES);
            fp()
        }
    }

    /// Set a capture option (u32 value).
    ///
    /// Returns `true` if the option was set successfully.
    pub fn set_capture_option_u32(&self, option: CaptureOption, value: u32) -> bool {
        // SAFETY: api is a valid RenderDoc API pointer, function offset is correct.
        unsafe {
            let fp = self.get_fn::<SetCaptureOptionU32Fn>(offsets::SET_CAPTURE_OPTION_U32);
            fp(option as i32, value) == 1
        }
    }

    /// Launch the RenderDoc replay UI.
    ///
    /// If `connect` is true, the UI will try to connect to the running
    /// application for live capture. Returns the PID of the replay UI process,
    /// or 0 on failure.
    pub fn launch_replay_ui(&self, connect: bool) -> u32 {
        // SAFETY: api is a valid RenderDoc API pointer, null command line is valid.
        unsafe {
            let fp = self.get_fn::<LaunchReplayUIFn>(offsets::LAUNCH_REPLAY_UI);
            fp(connect as u32, std::ptr::null())
        }
    }

    /// Read a function pointer from the API vtable at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index` is a valid offset and `F` matches the
    /// function signature at that offset.
    unsafe fn get_fn<F>(&self, index: usize) -> F {
        // SAFETY: The API pointer is a vtable of function pointers. Each entry
        // is a pointer-sized function pointer. The caller guarantees index and
        // type correctness.
        unsafe {
            let vtable = self.api as *const *const c_void;
            let fp_ptr = vtable.add(index);
            std::mem::transmute_copy(&*fp_ptr)
        }
    }
}

/// Platform-specific library loading.
#[cfg(target_os = "linux")]
fn load_renderdoc_api() -> Option<*mut c_void> {
    // SAFETY: dlopen/dlsym are standard POSIX functions.
    unsafe {
        // First try to find RenderDoc already loaded in the process.
        let lib = libc::dlopen(
            c"librenderdoc.so".as_ptr(),
            libc::RTLD_NOW | libc::RTLD_NOLOAD,
        );
        let lib = if lib.is_null() {
            // Try loading from system path.
            let lib = libc::dlopen(c"librenderdoc.so".as_ptr(), libc::RTLD_NOW);
            if lib.is_null() {
                debug!("RenderDoc: librenderdoc.so not found");
                return None;
            }
            lib
        } else {
            lib
        };

        let get_api = libc::dlsym(lib, c"RENDERDOC_GetAPI".as_ptr());
        if get_api.is_null() {
            warn!("RenderDoc: RENDERDOC_GetAPI symbol not found");
            return None;
        }

        let get_api: GetApiFn = std::mem::transmute(get_api);
        let mut api: *mut c_void = std::ptr::null_mut();
        let ret = get_api(RENDERDOC_API_VERSION, &mut api);
        if ret != 1 || api.is_null() {
            warn!("RenderDoc: RENDERDOC_GetAPI returned {ret}");
            return None;
        }

        Some(api)
    }
}

#[cfg(target_os = "windows")]
fn load_renderdoc_api() -> Option<*mut c_void> {
    use std::ffi::CStr;

    // SAFETY: GetModuleHandleA/GetProcAddress are standard Win32 functions.
    unsafe {
        let module = windows_sys::Win32::System::LibraryLoader::GetModuleHandleA(
            c"renderdoc.dll".as_ptr() as *const u8,
        );
        let module = if module == 0 {
            let module = windows_sys::Win32::System::LibraryLoader::LoadLibraryA(
                c"renderdoc.dll".as_ptr() as *const u8,
            );
            if module == 0 {
                debug!("RenderDoc: renderdoc.dll not found");
                return None;
            }
            module
        } else {
            module
        };

        let get_api = windows_sys::Win32::System::LibraryLoader::GetProcAddress(
            module,
            c"RENDERDOC_GetAPI".as_ptr() as *const u8,
        );
        let get_api = match get_api {
            Some(f) => f,
            None => {
                warn!("RenderDoc: RENDERDOC_GetAPI symbol not found");
                return None;
            }
        };

        let get_api: GetApiFn = std::mem::transmute(get_api);
        let mut api: *mut c_void = std::ptr::null_mut();
        let ret = get_api(RENDERDOC_API_VERSION, &mut api);
        if ret != 1 || api.is_null() {
            warn!("RenderDoc: RENDERDOC_GetAPI returned {ret}");
            return None;
        }

        Some(api)
    }
}

/// macOS stub — RenderDoc does not support macOS.
#[cfg(target_os = "macos")]
fn load_renderdoc_api() -> Option<*mut c_void> {
    debug!("RenderDoc: not supported on macOS");
    None
}

/// Fallback for other platforms.
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
fn load_renderdoc_api() -> Option<*mut c_void> {
    debug!("RenderDoc: unsupported platform");
    None
}
