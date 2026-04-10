//! Phase 6 validation: render graph execution with physical resource allocation.
//!
//! Builds a 2-pass render graph:
//!   1. An offscreen pass that clears a color attachment to a cycling color
//!   2. A final pass that copies from the offscreen target to the backbuffer
//!
//! The graph compiler orders passes and places barriers automatically.
//! `PhysicalResources::allocate` creates Vulkan images for all virtual resources.
//! `GraphExecutor::record` handles dynamic rendering (begin/end), barriers, and
//! the final present transition.

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use tephra::command::command_buffer::{CommandBuffer, CommandBufferType};
use tephra::core::context::QueueType;
use tephra::graph::{
    AttachmentInfo, CompiledGraph, GraphExecutor, PhysicalResources, RenderGraph,
    RenderPassCallback,
};
use tephra::wsi::platform::WinitPlatform;
use tephra::wsi::wsi::{WSI, WSIConfig};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Pass callback that clears to a cycling color based on a shared frame counter.
struct ClearPass {
    frame_counter: Arc<AtomicU64>,
}

impl RenderPassCallback for ClearPass {
    fn build_render_pass(&self, _cmd: &mut CommandBuffer) {
        // The executor begins dynamic rendering with CLEAR load op.
        // No draw commands needed — the clear color does the work.
    }

    fn clear_color(&self, _attachment_index: usize) -> vk::ClearColorValue {
        let t = self.frame_counter.load(Ordering::Relaxed) as f32 * 0.01;
        vk::ClearColorValue {
            float32: [
                (t.sin() * 0.5 + 0.5).clamp(0.0, 1.0),
                (t.cos() * 0.5 + 0.5).clamp(0.0, 1.0),
                ((t * 0.7).sin() * 0.5 + 0.5).clamp(0.0, 1.0),
                1.0,
            ],
        }
    }
}

struct App {
    wsi: Option<WSI>,
    window: Option<Window>,
    frame_counter: Arc<AtomicU64>,
    compiled: Option<CompiledGraph>,
    physical: Option<PhysicalResources>,
}

impl App {
    fn new() -> Self {
        Self {
            wsi: None,
            window: None,
            frame_counter: Arc::new(AtomicU64::new(0)),
            compiled: None,
            physical: None,
        }
    }

    fn build_graph(&mut self) {
        let wsi = self.wsi.as_mut().unwrap();
        let extent = wsi.swapchain_extent();
        let format = wsi.swapchain_format();

        let mut graph = RenderGraph::new();

        let counter = self.frame_counter.clone();

        // Pass 1: offscreen clear to cycling color (writes intermediate target)
        let mut offscreen = graph.add_pass("offscreen_clear");
        offscreen.set_render_callback(Box::new(ClearPass {
            frame_counter: counter.clone(),
        }));
        let intermediate = offscreen.add_color_output(
            "intermediate",
            AttachmentInfo::swapchain_relative(vk::Format::R8G8B8A8_UNORM, 1.0),
        );

        // Pass 2: final pass reads intermediate as texture, writes backbuffer.
        // The cycling color comes from the shared frame counter.
        // The texture input dependency ensures correct barrier placement.
        let mut final_pass = graph.add_pass("final_composite");
        final_pass.set_render_callback(Box::new(ClearPass {
            frame_counter: counter,
        }));
        final_pass.add_texture_input(intermediate);
        let backbuffer = final_pass.add_color_output(
            "backbuffer",
            AttachmentInfo::swapchain_relative(format, 1.0),
        );

        graph.set_backbuffer_source(backbuffer);

        log::info!("Compiling render graph...");
        let compiled = graph.bake();
        log::info!("Graph compiled: {} steps", compiled.step_count());
        for step in 0..compiled.step_count() {
            log::info!("  step[{}]: \"{}\"", step, compiled.step_name(step));
        }

        log::info!("Allocating physical resources...");
        let physical = PhysicalResources::allocate(wsi.device_mut(), &compiled, extent, format)
            .expect("failed to allocate physical resources");

        self.compiled = Some(compiled);
        self.physical = Some(physical);
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

        // Inject swapchain image into the backbuffer slot.
        self.physical
            .as_mut()
            .unwrap()
            .set_backbuffer(swapchain_image.image, swapchain_image.view);

        let raw_cmd = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("failed to allocate command buffer");
        let mut cmd = CommandBuffer::from_raw(
            raw_cmd,
            CommandBufferType::Graphics,
            wsi.device().raw().clone(),
        );

        // Record the entire graph: barriers, dynamic rendering, present transition.
        GraphExecutor::record(
            self.compiled.as_ref().unwrap(),
            &mut cmd,
            self.physical.as_ref().unwrap(),
        );

        if let Err(e) = wsi.end_frame(cmd.raw()) {
            log::warn!("end_frame failed: {e}");
        }

        self.frame_counter.fetch_add(1, Ordering::Relaxed);
        self.window.as_ref().unwrap().request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("tephra — 04 render graph")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop
            .create_window(window_attrs)
            .expect("failed to create window");

        let platform = WinitPlatform::new(&window).expect("failed to create winit platform");
        let wsi = WSI::new(&platform, &WSIConfig::default()).expect("failed to initialize WSI");

        self.window = Some(window);
        self.wsi = Some(wsi);
        self.build_graph();
        self.window.as_ref().unwrap().request_redraw();
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(wsi) = &mut self.wsi {
                    wsi.resize(size.width, size.height);
                    // Recreate graph resources on resize.
                    if let Some(physical) = self.physical.take() {
                        physical.destroy(wsi.device_mut());
                    }
                    self.compiled = None;
                }
                self.build_graph();
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
            // SAFETY: waiting for GPU idle before destroying resources.
            unsafe {
                wsi.device().raw().device_wait_idle().ok();
            }
        }
        if let Some(physical) = self.physical.take() {
            if let Some(wsi) = &mut self.wsi {
                physical.destroy(wsi.device_mut());
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
