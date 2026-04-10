//! Async pipeline compilation demo.
//!
//! Demonstrates `AsyncPipelineCompiler` compiling multiple pipeline variants on
//! background threads. Each variant uses a different polygon mode, blend state,
//! or cull mode. Draws are skipped until the pipeline is ready, so triangles
//! "pop in" as compilation completes.
//!
//! Requires: compiled SPIR-V shaders in `shaders/triangle.{vert,frag}.spv`.

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use ignis::command::barriers::ImageBarrierInfo;
use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::command::state::StaticPipelineState;
use ignis::core::context::QueueType;
use ignis::pipeline::async_pipeline::AsyncPipelineCompiler;
use ignis::pipeline::pipeline::VertexInputLayout;
use ignis::pipeline::program::Program;
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

/// Pipeline variant descriptions — each uses a different static state.
struct PipelineVariant {
    _label: &'static str,
    state: StaticPipelineState,
    viewport_offset: (f32, f32),
}

fn pipeline_variants() -> Vec<PipelineVariant> {
    let default = StaticPipelineState::default();

    vec![
        PipelineVariant {
            _label: "default (back-cull)",
            state: StaticPipelineState {
                cull_mode: vk::CullModeFlags::BACK,
                ..default.clone()
            },
            viewport_offset: (0.0, 0.0),
        },
        PipelineVariant {
            _label: "no cull",
            state: StaticPipelineState {
                cull_mode: vk::CullModeFlags::NONE,
                ..default.clone()
            },
            viewport_offset: (0.5, 0.0),
        },
        PipelineVariant {
            _label: "front cull",
            state: StaticPipelineState {
                cull_mode: vk::CullModeFlags::FRONT,
                ..default.clone()
            },
            viewport_offset: (0.0, 0.5),
        },
        PipelineVariant {
            _label: "alpha blend + no cull",
            state: StaticPipelineState {
                cull_mode: vk::CullModeFlags::NONE,
                blend_enable: true,
                src_color_blend: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend: vk::BlendFactor::ONE,
                dst_alpha_blend: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                ..default.clone()
            },
            viewport_offset: (0.5, 0.5),
        },
    ]
}

struct App {
    wsi: Option<WSI>,
    window: Option<Window>,
    frame: u64,
    vert_shader: Option<Shader>,
    frag_shader: Option<Shader>,
    program: Option<Program>,
    async_compiler: Option<AsyncPipelineCompiler>,
    variants: Vec<PipelineVariant>,
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
            async_compiler: None,
            variants: pipeline_variants(),
        }
    }

    fn init_pipeline(&mut self) {
        let wsi = self.wsi.as_ref().unwrap();
        let device = wsi.device().raw();

        let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv"));
        let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv"));

        let vert_shader = Shader::create(device, vk::ShaderStageFlags::VERTEX, &vert_spirv)
            .expect("failed to create vertex shader");
        let frag_shader = Shader::create(device, vk::ShaderStageFlags::FRAGMENT, &frag_spirv)
            .expect("failed to create fragment shader");

        let program = Program::create(device, &[&vert_shader, &frag_shader])
            .expect("failed to create program");

        let async_compiler = AsyncPipelineCompiler::new(device.clone(), vk::PipelineCache::null());

        self.vert_shader = Some(vert_shader);
        self.frag_shader = Some(frag_shader);
        self.program = Some(program);
        self.async_compiler = Some(async_compiler);

        log::info!(
            "Initialized with {} pipeline variants to compile asynchronously",
            self.variants.len()
        );
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

        // Poll async compiler for completed pipelines
        let compiler = self.async_compiler.as_mut().unwrap();
        compiler.poll_completed();

        let pending = compiler.pending_count();
        let ready = compiler.ready_count();
        if self.frame < 30 || self.frame % 60 == 0 {
            log::info!(
                "Frame {}: {} pipelines ready, {} pending",
                self.frame,
                ready,
                pending
            );
        }

        let raw_cmd = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("failed to allocate command buffer");
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

        // Begin dynamic rendering
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(swapchain_image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.05, 0.05, 0.08, 1.0],
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));

        cmd.begin_rendering(&rendering_info);

        // Draw each pipeline variant in a quadrant of the screen
        let program = self.program.as_ref().unwrap();
        let vertex_layout = VertexInputLayout::default();
        let color_formats = [format];
        let compiler = self.async_compiler.as_mut().unwrap();

        let half_w = extent.width as f32 * 0.5;
        let half_h = extent.height as f32 * 0.5;

        for variant in &self.variants {
            let pipeline = compiler.request_graphics_dynamic(
                program,
                &variant.state,
                &color_formats,
                vk::Format::UNDEFINED,
                vk::Format::UNDEFINED,
                &vertex_layout,
            );

            let Some(pipeline) = pipeline else {
                // Pipeline not ready yet — skip this draw
                continue;
            };

            // Set viewport to this variant's quadrant
            let vp_x = variant.viewport_offset.0 * extent.width as f32;
            let vp_y = variant.viewport_offset.1 * extent.height as f32;

            cmd.set_viewport(
                0,
                &[vk::Viewport {
                    x: vp_x,
                    y: vp_y,
                    width: half_w,
                    height: half_h,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            cmd.set_scissor(
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: vp_x as i32,
                        y: vp_y as i32,
                    },
                    extent: vk::Extent2D {
                        width: half_w as u32,
                        height: half_h as u32,
                    },
                }],
            );

            cmd.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
            cmd.draw(3, 1, 0, 0);
        }

        cmd.end_rendering();

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
            .with_title("ignis - 10 async pipeline compilation")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop
            .create_window(window_attrs)
            .expect("failed to create window");

        let platform = WinitPlatform::new(&window).expect("failed to create winit platform");
        let wsi = WSI::new(&platform, &WSIConfig::default()).expect("failed to initialize WSI");

        self.window = Some(window);
        self.wsi = Some(wsi);
        self.init_pipeline();
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
            unsafe {
                wsi.device().raw().device_wait_idle().ok();
            }
        }
        if let Some(mut compiler) = self.async_compiler.take() {
            if let Some(wsi) = &self.wsi {
                compiler.destroy(wsi.device().raw());
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
