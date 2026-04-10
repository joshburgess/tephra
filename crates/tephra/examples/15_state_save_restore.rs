//! State save/restore and convenience presets example.
//!
//! Demonstrates `DrawContext::save_state()` / `restore_state()` for temporarily
//! changing pipeline state (e.g., drawing a debug overlay) and the convenience
//! state presets (`set_opaque_state()`, `set_transparent_sprite_state()`, etc.).
//!
//! Renders a scene triangle with opaque state, saves state, draws a second
//! triangle with transparent blending (simulating a UI overlay), then restores
//! the original state and draws a third triangle to prove restoration worked.
//!
//! Requires: compiled SPIR-V shaders `shaders/triangle.{vert,frag}.spv`.

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use tephra::command::barriers::ImageBarrierInfo;
use tephra::command::command_buffer::{CommandBuffer, CommandBufferType};
use tephra::core::context::QueueType;
use tephra::pipeline::draw_context::{DrawContext, FrameResources, RenderingAttachment};
use tephra::pipeline::pipeline::VertexInputLayout;
use tephra::pipeline::program::Program;
use tephra::pipeline::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use tephra::pipeline::shader::Shader;
use tephra::wsi::platform::WinitPlatform;
use tephra::wsi::wsi::{WSI, WSIConfig};

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
        }
    }

    fn init(&mut self) {
        let wsi = self.wsi.as_ref().unwrap();
        let device = wsi.device().raw();

        let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv"));
        let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv"));

        let vert_shader = Shader::create(device, vk::ShaderStageFlags::VERTEX, &vert_spirv)
            .expect("vertex shader");
        let frag_shader = Shader::create(device, vk::ShaderStageFlags::FRAGMENT, &frag_spirv)
            .expect("fragment shader");
        let program = Program::create(device, &[&vert_shader, &frag_shader]).expect("program");

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

        let raw_cmd = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("cmd alloc");
        let mut cmd = CommandBuffer::from_raw(
            raw_cmd,
            CommandBufferType::Graphics,
            wsi.device().raw().clone(),
        );

        // Transition: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        cmd.image_barrier(&ImageBarrierInfo::undefined_to(
            swapchain_image.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageAspectFlags::COLOR,
        ));

        let frame_resources = self.frame_resources.as_mut().unwrap();
        let program = self.program.as_mut().unwrap();
        let vertex_layout = VertexInputLayout::default();

        {
            let mut ctx = DrawContext::new(&mut cmd, wsi.device().raw(), frame_resources);

            let color_attachment = RenderingAttachment {
                view: swapchain_image.view,
                format,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.05, 0.05, 0.08, 1.0],
                    },
                },
                resolve_view: None,
            };

            ctx.begin_rendering(extent, &[color_attachment], None)
                .expect("begin_rendering");

            // ============================================================
            // Step 1: Draw main scene triangle with opaque state
            // ============================================================
            ctx.set_opaque_state();
            // Override depth since we don't have a depth buffer
            ctx.set_depth_test(false, false);

            // Draw in left half
            let half_w = extent.width as f32 * 0.5;
            ctx.cmd().set_viewport(
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: half_w,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            ctx.cmd().set_scissor(
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: half_w as u32,
                        height: extent.height,
                    },
                }],
            );

            ctx.draw(program, &vertex_layout, 3, 1, 0, 0)
                .expect("draw 1: opaque");

            if self.frame == 0 {
                println!("Step 1: Drew opaque triangle (left half)");
            }

            // ============================================================
            // Step 2: Save state, switch to transparent overlay, draw
            // ============================================================
            let saved = ctx.save_state();

            ctx.set_transparent_sprite_state();
            // Override depth since we don't have a depth buffer
            ctx.set_depth_test(false, false);

            // Draw in center (overlapping both halves)
            let quarter_w = extent.width as f32 * 0.25;
            ctx.cmd().set_viewport(
                0,
                &[vk::Viewport {
                    x: quarter_w,
                    y: 0.0,
                    width: half_w,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            ctx.cmd().set_scissor(
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: quarter_w as i32,
                        y: 0,
                    },
                    extent: vk::Extent2D {
                        width: half_w as u32,
                        height: extent.height,
                    },
                }],
            );

            ctx.draw(program, &vertex_layout, 3, 1, 0, 0)
                .expect("draw 2: transparent overlay");

            if self.frame == 0 {
                println!("Step 2: Drew transparent overlay triangle (center)");
            }

            // ============================================================
            // Step 3: Restore state — back to opaque, draw in right half
            // ============================================================
            ctx.restore_state(&saved);

            ctx.cmd().set_viewport(
                0,
                &[vk::Viewport {
                    x: half_w,
                    y: 0.0,
                    width: half_w,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            ctx.cmd().set_scissor(
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: half_w as i32,
                        y: 0,
                    },
                    extent: vk::Extent2D {
                        width: half_w as u32,
                        height: extent.height,
                    },
                }],
            );

            ctx.draw(program, &vertex_layout, 3, 1, 0, 0)
                .expect("draw 3: restored opaque");

            if self.frame == 0 {
                println!("Step 3: Restored state, drew opaque triangle (right half)");
                println!(
                    "\nYou should see: two opaque triangles (left/right) with a transparent overlay (center)"
                );
            }

            ctx.end_rendering();
        }

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
            .with_title("tephra - 15 state save/restore")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 600));

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

impl Drop for App {
    fn drop(&mut self) {
        if let Some(wsi) = &self.wsi {
            // SAFETY: waiting for device idle before cleanup.
            unsafe {
                wsi.device().raw().device_wait_idle().ok();
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
