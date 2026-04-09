//! Phase 1+2 validation: clear-screen window.
//!
//! Creates a window, initializes WSI with Vulkan, and runs a main loop that
//! clears the swapchain image to a cycling color. Demonstrates:
//! - Winit window creation
//! - WSI initialization (device + swapchain)
//! - Frame lifecycle (begin_frame / end_frame)
//! - Command buffer allocation and barrier transitions

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use ignis::command::barriers::ImageBarrierInfo;
use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::core::context::QueueType;
use ignis::wsi::platform::WinitPlatform;
use ignis::wsi::wsi::{WSI, WSIConfig};

struct App {
    wsi: Option<WSI>,
    window: Option<Window>,
    frame: u64,
}

impl App {
    fn new() -> Self {
        Self {
            wsi: None,
            window: None,
            frame: 0,
        }
    }

    fn render(&mut self) {
        let wsi = self.wsi.as_mut().unwrap();

        let swapchain_image = match wsi.begin_frame() {
            Ok(img) => img,
            Err(e) => {
                log::warn!("begin_frame failed: {e}");
                return;
            }
        };

        // Allocate and begin a command buffer
        let raw_cmd = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("failed to allocate command buffer");
        let mut cmd =
            CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, wsi.device().raw().clone());

        // Transition swapchain image: UNDEFINED → TRANSFER_DST for clear
        cmd.image_barrier(&ImageBarrierInfo::undefined_to(
            swapchain_image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageAspectFlags::COLOR,
        ));

        // Cycle clear color over time
        let t = self.frame as f32 * 0.01;
        let r = (t.sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let g = ((t + 2.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let b = ((t + 4.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);

        let clear_color = vk::ClearColorValue {
            float32: [r, g, b, 1.0],
        };

        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        // SAFETY: command buffer and image are valid, layout is TRANSFER_DST.
        unsafe {
            wsi.device().raw().cmd_clear_color_image(
                cmd.raw(),
                swapchain_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[range],
            );
        }

        // Transition: TRANSFER_DST → PRESENT_SRC
        cmd.image_barrier(&ImageBarrierInfo {
            image: swapchain_image.image,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_stage: vk::PipelineStageFlags2::TRANSFER,
            src_access: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            dst_access: vk::AccessFlags2::NONE,
            subresource_range: range,
        });

        // Submit and present
        if let Err(e) = wsi.end_frame(cmd.raw()) {
            log::warn!("end_frame failed: {e}");
        }

        self.frame += 1;
        self.window.as_ref().unwrap().request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ignis — 01_triangle")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop
            .create_window(window_attrs)
            .expect("failed to create window");

        let platform =
            WinitPlatform::new(&window).expect("failed to create winit platform");

        let wsi = WSI::new(&platform, &WSIConfig::default())
            .expect("failed to initialize WSI");

        self.window = Some(window);
        self.wsi = Some(wsi);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(wsi) = &mut self.wsi {
                    wsi.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}
