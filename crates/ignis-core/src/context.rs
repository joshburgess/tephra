//! Vulkan instance, physical device selection, logical device creation, and queue discovery.
//!
//! The [`Context`] struct owns the Vulkan instance, device, and queues. It is the
//! entry point for all Vulkan operations.

use std::collections::HashSet;
use std::ffi::{CStr, CString};

use ash::vk;
use gpu_allocator::vulkan as vma;
use parking_lot::Mutex;
use thiserror::Error;

/// Errors that can occur during context creation.
#[derive(Debug, Error)]
pub enum ContextError {
    /// Failed to load the Vulkan library.
    #[error("failed to load Vulkan library: {0}")]
    LoadingError(String),

    /// Failed to create the Vulkan instance.
    #[error("failed to create Vulkan instance: {0}")]
    InstanceCreation(vk::Result),

    /// No suitable physical device found.
    #[error("no suitable Vulkan physical device found")]
    NoSuitableDevice,

    /// Failed to create the logical device.
    #[error("failed to create logical device: {0}")]
    DeviceCreation(vk::Result),

    /// Failed to create the memory allocator.
    #[error("failed to create GPU allocator: {0}")]
    AllocatorCreation(String),
}

/// Configuration for creating a [`Context`].
pub struct ContextConfig {
    /// Application name reported to the Vulkan driver.
    pub app_name: CString,
    /// Application version reported to the Vulkan driver.
    pub app_version: u32,
    /// Whether to enable validation layers and debug messenger.
    pub enable_validation: bool,
    /// Additional instance extensions to enable (e.g., surface extensions).
    pub required_instance_extensions: Vec<*const std::ffi::c_char>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            app_name: CString::new("ignis application").unwrap(),
            app_version: vk::make_api_version(0, 1, 0, 0),
            enable_validation: cfg!(debug_assertions),
            required_instance_extensions: Vec::new(),
        }
    }
}

/// Information about a device queue.
pub struct QueueInfo {
    /// The raw Vulkan queue handle.
    pub queue: vk::Queue,
    /// The queue family index this queue belongs to.
    pub family_index: u32,
}

/// Queue types supported by the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueType {
    /// Graphics-capable queue (also supports compute and transfer).
    Graphics,
    /// Dedicated or shared compute queue.
    Compute,
    /// Dedicated or shared transfer queue.
    Transfer,
}

/// Summary of device features enabled at creation time.
pub struct DeviceFeatures {
    /// Whether timeline semaphores are supported (Vulkan 1.2+).
    pub timeline_semaphore: bool,
    /// Whether synchronization2 is supported (Vulkan 1.3+).
    pub synchronization2: bool,
    /// Whether dynamic rendering is supported (Vulkan 1.3+).
    pub dynamic_rendering: bool,
    /// Whether push descriptors are supported (VK_KHR_push_descriptor).
    pub push_descriptors: bool,
    /// Whether descriptor indexing features are available for bindless rendering (Vulkan 1.2+).
    pub descriptor_indexing: bool,
    /// Whether buffer device address is supported (Vulkan 1.2+).
    pub buffer_device_address: bool,
}

/// The core Vulkan context owning instance, device, queues, and allocator.
///
/// Created via [`Context::new`]. This is the foundational object from which
/// all other ignis objects are derived.
pub struct Context {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue: QueueInfo,
    compute_queue: QueueInfo,
    transfer_queue: QueueInfo,
    allocator: Mutex<Option<vma::Allocator>>,
    device_properties: vk::PhysicalDeviceProperties,
    device_features: DeviceFeatures,
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    push_descriptor_device: Option<ash::khr::push_descriptor::Device>,
}

/// Vulkan debug messenger callback. Routes validation messages to the `log` crate.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if callback_data.is_null() {
        return vk::FALSE;
    }

    // SAFETY: callback_data is a valid pointer provided by the Vulkan runtime.
    let data = unsafe { &*callback_data };
    let message = if data.p_message.is_null() {
        "<no message>"
    } else {
        // SAFETY: p_message is a valid null-terminated string from Vulkan.
        unsafe { CStr::from_ptr(data.p_message) }
            .to_str()
            .unwrap_or("<invalid utf8>")
    };

    let type_str = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
        _ => "[GENERAL]",
    };

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!("{type_str} {message}");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("{type_str} {message}");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!("{type_str} {message}");
        }
        _ => {
            log::debug!("{type_str} {message}");
        }
    }

    vk::FALSE
}

/// Result of queue family discovery.
struct QueueFamilyIndices {
    graphics: u32,
    compute: u32,
    transfer: u32,
}

impl Context {
    /// Create a new Vulkan context with the given configuration.
    ///
    /// This initializes the Vulkan library, creates an instance and logical device,
    /// discovers queues, and sets up the GPU memory allocator.
    pub fn new(config: &ContextConfig) -> Result<Self, ContextError> {
        // Load Vulkan library
        // SAFETY: Loading the Vulkan library is safe assuming a conformant Vulkan driver is installed.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| ContextError::LoadingError(e.to_string()))?;

        // Build instance extensions list
        let mut instance_extensions: Vec<*const std::ffi::c_char> = Vec::new();
        instance_extensions.extend_from_slice(&config.required_instance_extensions);

        // Layers
        let mut layers: Vec<*const std::ffi::c_char> = Vec::new();

        if config.enable_validation {
            instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }

        // On macOS/MoltenVK, we need the portability enumeration extension
        #[cfg(target_os = "macos")]
        {
            instance_extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            instance_extensions
                .push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let app_info = vk::ApplicationInfo::default()
            .application_name(&config.app_name)
            .application_version(config.app_version)
            .engine_name(c"ignis")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let mut instance_create_flags = vk::InstanceCreateFlags::empty();
        #[cfg(target_os = "macos")]
        {
            instance_create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let mut instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&instance_extensions)
            .flags(instance_create_flags);

        // Chain debug messenger create info into instance creation so we get
        // validation messages during instance creation/destruction too.
        let mut debug_messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        if config.enable_validation {
            instance_ci = instance_ci.push_next(&mut debug_messenger_ci);
        }

        // SAFETY: instance_ci is well-formed with valid extension/layer names.
        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(ContextError::InstanceCreation)?;

        // Set up persistent debug messenger
        let (debug_utils, debug_messenger) = if config.enable_validation {
            let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));

            // SAFETY: debug_utils_loader and messenger_ci are valid.
            let messenger = unsafe {
                debug_utils_loader.create_debug_utils_messenger(&messenger_ci, None)
            }
            .map_err(|e| ContextError::InstanceCreation(e))?;

            (Some(debug_utils_loader), Some(messenger))
        } else {
            (None, None)
        };

        // Select physical device
        // SAFETY: instance is valid.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(ContextError::InstanceCreation)?;

        if physical_devices.is_empty() {
            return Err(ContextError::NoSuitableDevice);
        }

        let (physical_device, queue_indices) = Self::select_physical_device(&instance, &physical_devices)?;

        // SAFETY: physical_device is a valid handle from enumeration.
        let device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let device_name = {
            // SAFETY: device_name is a null-terminated C string from the driver.
            let name = unsafe { CStr::from_ptr(device_properties.device_name.as_ptr()) };
            name.to_string_lossy()
        };
        log::info!(
            "Selected device: {} (Vulkan {}.{})",
            device_name,
            vk::api_version_major(device_properties.api_version),
            vk::api_version_minor(device_properties.api_version),
        );
        log::info!(
            "Queue families — graphics: {}, compute: {}, transfer: {}",
            queue_indices.graphics,
            queue_indices.compute,
            queue_indices.transfer,
        );

        // Query supported features
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);

        // SAFETY: physical_device is valid, features2 chain is properly constructed.
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

        // Check for optional device extensions
        let device_extension_props = unsafe {
            instance.enumerate_device_extension_properties(physical_device)
        }
        .unwrap_or_default();

        let has_push_descriptor = device_extension_props.iter().any(|ext| {
            // SAFETY: extension_name is a null-terminated C string from the driver.
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == ash::khr::push_descriptor::NAME
        });

        let has_descriptor_indexing =
            vulkan_12_features.descriptor_binding_partially_bound == vk::TRUE
                && vulkan_12_features.descriptor_binding_sampled_image_update_after_bind
                    == vk::TRUE
                && vulkan_12_features.descriptor_binding_storage_buffer_update_after_bind
                    == vk::TRUE
                && vulkan_12_features.runtime_descriptor_array == vk::TRUE
                && vulkan_12_features.shader_sampled_image_array_non_uniform_indexing == vk::TRUE
                && vulkan_12_features.shader_storage_buffer_array_non_uniform_indexing == vk::TRUE;

        let has_buffer_device_address =
            vulkan_12_features.buffer_device_address == vk::TRUE;

        let device_features = DeviceFeatures {
            timeline_semaphore: vulkan_12_features.timeline_semaphore == vk::TRUE,
            synchronization2: vulkan_13_features.synchronization2 == vk::TRUE,
            dynamic_rendering: vulkan_13_features.dynamic_rendering == vk::TRUE,
            push_descriptors: has_push_descriptor,
            descriptor_indexing: has_descriptor_indexing,
            buffer_device_address: has_buffer_device_address,
        };

        // Create logical device
        let priorities = [1.0f32];

        // Deduplicate queue family indices — Vulkan requires unique families in create infos.
        let unique_families: Vec<u32> = {
            let mut set = HashSet::new();
            set.insert(queue_indices.graphics);
            set.insert(queue_indices.compute);
            set.insert(queue_indices.transfer);
            set.into_iter().collect()
        };

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo<'_>> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities)
            })
            .collect();

        // Device extensions — only enable VK_KHR_swapchain when the instance
        // has VK_KHR_surface (i.e., this is a windowed application, not headless).
        let has_surface = config.required_instance_extensions.iter().any(|&ext| {
            // SAFETY: the pointers in required_instance_extensions are valid C strings.
            let name = unsafe { CStr::from_ptr(ext) };
            name == ash::khr::surface::NAME
        });
        let mut device_extensions: Vec<*const std::ffi::c_char> = Vec::new();
        if has_surface {
            device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }

        #[cfg(target_os = "macos")]
        {
            device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        if has_push_descriptor {
            device_extensions.push(ash::khr::push_descriptor::NAME.as_ptr());
        }

        // Enable Vulkan 1.2 and 1.3 features we need
        let mut enabled_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(device_features.timeline_semaphore)
            .descriptor_binding_partially_bound(device_features.descriptor_indexing)
            .descriptor_binding_sampled_image_update_after_bind(
                device_features.descriptor_indexing,
            )
            .descriptor_binding_storage_buffer_update_after_bind(
                device_features.descriptor_indexing,
            )
            .runtime_descriptor_array(device_features.descriptor_indexing)
            .shader_sampled_image_array_non_uniform_indexing(
                device_features.descriptor_indexing,
            )
            .shader_storage_buffer_array_non_uniform_indexing(
                device_features.descriptor_indexing,
            )
            .buffer_device_address(device_features.buffer_device_address);
        let mut enabled_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(device_features.synchronization2)
            .dynamic_rendering(device_features.dynamic_rendering);

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut enabled_12_features)
            .push_next(&mut enabled_13_features);

        // SAFETY: device_ci is well-formed with valid queue families and extensions.
        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(ContextError::DeviceCreation)?;

        // Retrieve queues
        // SAFETY: device is valid, queue families are valid indices.
        let graphics_queue = QueueInfo {
            queue: unsafe { device.get_device_queue(queue_indices.graphics, 0) },
            family_index: queue_indices.graphics,
        };
        let compute_queue = QueueInfo {
            queue: unsafe { device.get_device_queue(queue_indices.compute, 0) },
            family_index: queue_indices.compute,
        };
        let transfer_queue = QueueInfo {
            queue: unsafe { device.get_device_queue(queue_indices.transfer, 0) },
            family_index: queue_indices.transfer,
        };

        // Create extension loaders
        let push_descriptor_device = if has_push_descriptor {
            Some(ash::khr::push_descriptor::Device::new(&instance, &device))
        } else {
            None
        };

        // Create GPU allocator
        let allocator = vma::Allocator::new(&vma::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings {
                log_leaks_on_shutdown: true,
                log_memory_information: cfg!(debug_assertions),
                log_allocations: false,
                log_frees: false,
                log_stack_traces: false,
                store_stack_traces: false,
            },
            buffer_device_address: device_features.buffer_device_address,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })
        .map_err(|e| ContextError::AllocatorCreation(e.to_string()))?;

        Ok(Self {
            entry,
            instance,
            device,
            physical_device,
            graphics_queue,
            compute_queue,
            transfer_queue,
            allocator: Mutex::new(Some(allocator)),
            device_properties,
            device_features,
            debug_utils,
            debug_messenger,
            push_descriptor_device,
        })
    }

    /// Select the best physical device using a scoring heuristic.
    fn select_physical_device(
        instance: &ash::Instance,
        devices: &[vk::PhysicalDevice],
    ) -> Result<(vk::PhysicalDevice, QueueFamilyIndices), ContextError> {
        let mut best: Option<(vk::PhysicalDevice, QueueFamilyIndices, i32)> = None;

        for &phys_dev in devices {
            // SAFETY: phys_dev is a valid handle from enumeration.
            let properties = unsafe { instance.get_physical_device_properties(phys_dev) };
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };

            // Must have a graphics queue
            let graphics = queue_families.iter().position(|qf| {
                qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            });
            let graphics = match graphics {
                Some(idx) => idx as u32,
                None => continue,
            };

            // Find separate compute queue (prefer one that is NOT graphics)
            let compute = queue_families
                .iter()
                .enumerate()
                .position(|(i, qf)| {
                    qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && i as u32 != graphics
                })
                .or_else(|| {
                    // Fall back to any compute-capable queue
                    queue_families.iter().position(|qf| {
                        qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    })
                })
                .unwrap_or(graphics as usize) as u32;

            // Find separate transfer queue (prefer one that is neither graphics nor compute)
            let transfer = queue_families
                .iter()
                .enumerate()
                .position(|(i, qf)| {
                    qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && !qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && i as u32 != graphics
                        && i as u32 != compute
                })
                .or_else(|| {
                    // Fall back to any transfer-capable queue that isn't graphics
                    queue_families.iter().enumerate().position(|(i, qf)| {
                        qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                            && i as u32 != graphics
                    })
                })
                .unwrap_or(graphics as usize) as u32;

            let mut score: i32 = 0;

            // Strongly prefer discrete GPUs
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                score += 10000;
            } else if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                score += 1000;
            }

            // Prefer devices with separate compute queue
            if compute != graphics {
                score += 100;
            }

            // Prefer devices with separate transfer queue
            if transfer != graphics {
                score += 50;
            }

            // Prefer higher Vulkan API version
            score += vk::api_version_minor(properties.api_version) as i32 * 10;

            let indices = QueueFamilyIndices {
                graphics,
                compute,
                transfer,
            };

            if best.as_ref().is_none_or(|(_, _, best_score)| score > *best_score) {
                best = Some((phys_dev, indices, score));
            }
        }

        match best {
            Some((dev, indices, _)) => Ok((dev, indices)),
            None => Err(ContextError::NoSuitableDevice),
        }
    }

    /// The Vulkan entry point.
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// The Vulkan instance.
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// The Vulkan logical device.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// The selected physical device.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Queue info for the given queue type.
    pub fn queue(&self, queue_type: QueueType) -> &QueueInfo {
        match queue_type {
            QueueType::Graphics => &self.graphics_queue,
            QueueType::Compute => &self.compute_queue,
            QueueType::Transfer => &self.transfer_queue,
        }
    }

    /// The GPU memory allocator (behind a mutex for thread safety).
    pub fn allocator(&self) -> &Mutex<Option<vma::Allocator>> {
        &self.allocator
    }

    /// Physical device properties (name, limits, API version, etc.).
    pub fn device_properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.device_properties
    }

    /// Summary of enabled device features.
    pub fn device_features(&self) -> &DeviceFeatures {
        &self.device_features
    }

    /// The push descriptor extension loader, if available.
    pub fn push_descriptor_device(&self) -> Option<&ash::khr::push_descriptor::Device> {
        self.push_descriptor_device.as_ref()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: We own all these handles and this is the correct destruction order.
        unsafe {
            // Take and drop the allocator before destroying the device —
            // gpu-allocator may call vkFreeMemory during its own Drop.
            {
                let mut guard = self.allocator.lock();
                if let Some(alloc) = guard.take() {
                    if cfg!(debug_assertions) {
                        alloc.report_memory_leaks(log::Level::Warn);
                    }
                    drop(alloc);
                }
            }

            self.device.destroy_device(None);

            if let (Some(debug_utils), Some(messenger)) =
                (&self.debug_utils, self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}
