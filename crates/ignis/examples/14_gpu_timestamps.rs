//! GPU timestamp queries example.
//!
//! Renders a triangle and measures GPU timing using `TimestampQueryPool`.
//! Demonstrates `begin(name)` / `end(name)` timestamp regions, readback
//! of GPU timing intervals, and printing per-pass GPU time in milliseconds.
//!
//! Requires: compiled SPIR-V shaders `shaders/triangle.{vert,frag}.spv`.

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use ignis::command::barriers::ImageBarrierInfo;
use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::command::query_pool::TimestampQueryPool;
use ignis::core::context::QueueType;
use ignis::pipeline::draw_context::{DrawContext, FrameResources, RenderingAttachment};
use ignis::pipeline::pipeline::VertexInputLayout;
use ignis::pipeline::program::Program;
use ignis::pipeline::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use ignis::pipeline::shader::Shader;
use ignis::wsi::platform::WinitPlatform;
use ignis::wsi::wsi::{WSI, WSIConfig};

fn spirv_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct App {
    wsi: Option<WSI>,
    window: Option<Window>,
    frame: u64,
    vert_shader: Option<Shader>,
    frag_shader: Option<Shader>,
    program: Option<Program>,
    frame_resources: Option<FrameResources>,
    timestamps: Option<TimestampQueryPool>,
    timestamp_period: f32,
    /// Stores last frame's timestamp results for display.
    last_timings: Vec<(String, f64)>,
}

impl App {
    fn new() -> Self {
        Self {
            wsi: None,
            window: None,
            frame: 0,
            vert_shader: None,
            frag_shader: None,
            program: None,
            frame_resources: None,
            timestamps: None,
            timestamp_period: 1.0,
            last_timings: Vec::new(),
        }
    }

    fn init(&mut self) {
        let wsi = self.wsi.as_ref().unwrap();
        let device = wsi.device();

        // Get timestamp period from device properties
        let props = device.context().device_properties();
        self.timestamp_period = props.limits.timestamp_period;
        println!(
            "Timestamp period: {} ns/tick",
            self.timestamp_period
        );

        // Create timestamp query pool (enough for several begin/end pairs)
        self.timestamps = Some(
            TimestampQueryPool::new(device.raw(), 32).expect("failed to create timestamp pool"),
        );

        // Load shaders
        let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv"));
        let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv"));

        let vert_shader =
            Shader::create(device.raw(), vk::ShaderStageFlags::VERTEX, &vert_spirv)
                .expect("vertex shader");
        let frag_shader =
            Shader::create(device.raw(), vk::ShaderStageFlags::FRAGMENT, &frag_spirv)
                .expect("fragment shader");
        let program = Program::create(device.raw(), &[&vert_shader, &frag_shader])
            .expect("program");

        self.vert_shader = Some(vert_shader);
        self.frag_shader = Some(frag_shader);
        self.program = Some(program);
        self.frame_resources = Some(FrameResources::new(vk::PipelineCache::null()));
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

        let extent = wsi.swapchain_extent();
        let format = wsi.swapchain_format();

        // Read back timestamps from the PREVIOUS frame (GPU has completed it)
        let timestamps = self.timestamps.as_mut().unwrap();
        if self.frame > 1 {
            match timestamps.readback(wsi.device().raw(), self.timestamp_period) {
                Ok(results) => {
                    self.last_timings = results;
                }
                Err(e) => {
                    log::warn!("Timestamp readback failed: {e}");
                }
            }
        }

        // Print timings periodically
        if self.frame > 0 && self.frame % 60 == 0 {
            println!("--- Frame {} GPU Timings ---", self.frame);
            for (name, ns) in &self.last_timings {
                println!("  {}: {:.3} ms", name, ns / 1_000_000.0);
            }
            if self.last_timings.is_empty() {
                println!("  (no timing data yet)");
            }
        }

        // Reset timestamp pool for this frame
        timestamps.host_reset(wsi.device().raw());

        // Record commands
        let raw_cmd = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("cmd alloc");
        let mut cmd = CommandBuffer::from_raw(
            raw_cmd,
            CommandBufferType::Graphics,
            wsi.device().raw().clone(),
        );

        // Timestamp: beginning of frame
        timestamps.begin(&mut cmd, "full_frame", vk::PipelineStageFlags2::TOP_OF_PIPE);

        // Transition: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        cmd.image_barrier(&ImageBarrierInfo::undefined_to(
            swapchain_image.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageAspectFlags::COLOR,
        ));

        // Timestamp: render pass begin
        timestamps.begin(
            &mut cmd,
            "render_pass",
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        // Draw triangle using DrawContext
        let frame_resources = self.frame_resources.as_mut().unwrap();
        let program = self.program.as_mut().unwrap();

        {
            let mut ctx = DrawContext::new(&mut cmd, wsi.device().raw(), frame_resources);

            let color_attachment = RenderingAttachment {
                view: swapchain_image.view,
                format,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.02, 0.02, 0.05, 1.0],
                    },
                },
                resolve_view: None,
            };

            ctx.begin_rendering(extent, &[color_attachment], None)
                .expect("begin_rendering");

            let vertex_layout = VertexInputLayout::default();
            ctx.draw(program, &vertex_layout, 3, 1, 0, 0)
                .expect("draw");

            ctx.end_rendering();
        }

        // Timestamp: render pass end
        timestamps.end(
            &mut cmd,
            "render_pass",
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        // Transition: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
        cmd.image_barrier(&ImageBarrierInfo {
            image: swapchain_image.image,
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            src_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            dst_access: vk::AccessFlags2::NONE,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_queue_family: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
        });

        // Timestamp: end of frame
        timestamps.end(&mut cmd, "full_frame", vk::PipelineStageFlags2::BOTTOM_OF_PIPE);

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
            .with_title("ignis - 14 gpu timestamps")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = event_loop
            .create_window(window_attrs)
            .expect("failed to create window");

        let platform = WinitPlatform::new(&window).expect("failed to create winit platform");
        let wsi = WSI::new(&platform, &WSIConfig::default()).expect("failed to initialize WSI");

        self.window = Some(window);
        self.wsi = Some(wsi);
        self.init();
        self.window.as_ref().unwrap().request_redraw();
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
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

impl Drop for App {
    fn drop(&mut self) {
        if let Some(wsi) = &self.wsi {
            // SAFETY: waiting for device idle before cleanup.
            unsafe { wsi.device().raw().device_wait_idle().ok(); }
        }
        if let Some(mut timestamps) = self.timestamps.take() {
            if let Some(wsi) = &self.wsi {
                timestamps.destroy(wsi.device().raw());
            }
        }
        if let Some(mut fr) = self.frame_resources.take() {
            if let Some(wsi) = &self.wsi {
                fr.destroy(wsi.device().raw());
            }
        }
        if let Some(mut program) = self.program.take() {
            if let Some(wsi) = &self.wsi {
                program.destroy(wsi.device().raw());
            }
        }
        if let Some(mut shader) = self.vert_shader.take() {
            if let Some(wsi) = &self.wsi {
                shader.destroy(wsi.device().raw());
            }
        }
        if let Some(mut shader) = self.frag_shader.take() {
            if let Some(wsi) = &self.wsi {
                shader.destroy(wsi.device().raw());
            }
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
