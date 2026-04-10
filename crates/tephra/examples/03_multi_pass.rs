//! Two-pass rendering with manual barriers.
//!
//! Pass 1: renders a spinning triangle to a 512x512 offscreen render target.
//! Pass 2: samples the offscreen RT with a chromatic aberration + vignette
//! post-process effect and draws a fullscreen triangle to the swapchain.
//!
//! Demonstrates manual image layout transitions between render passes —
//! the kind of work that the render graph automates in Phase 6.
//!
//! Requires compiled SPIR-V shaders in `shaders/`:
//!   triangle.{vert,frag}.spv, fullscreen.vert.spv, postprocess.frag.spv

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use tephra::command::barriers::ImageBarrierInfo;
use tephra::command::command_buffer::{CommandBuffer, CommandBufferType};
use tephra::core::context::QueueType;
use tephra::core::handles::ImageHandle;
use tephra::core::image::ImageCreateInfo;
use tephra::core::sampler::StockSampler;
use tephra::pipeline::draw_context::{DrawContext, FrameResources};
use tephra::pipeline::pipeline::VertexInputLayout;
use tephra::pipeline::program::Program;
use tephra::pipeline::render_pass::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachmentInfo, RenderPassInfo,
};
use tephra::pipeline::shader::Shader;
use tephra::wsi::platform::WinitPlatform;
use tephra::wsi::wsi::{WSI, WSIConfig};

const OFFSCREEN_W: u32 = 512;
const OFFSCREEN_H: u32 = 512;

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
    // Pass 1: scene shaders (reuse triangle shaders)
    scene_vert: Option<Shader>,
    scene_frag: Option<Shader>,
    scene_program: Option<Program>,
    // Pass 2: post-process shaders
    post_vert: Option<Shader>,
    post_frag: Option<Shader>,
    post_program: Option<Program>,
    // Shared
    frame_resources: Option<FrameResources>,
    offscreen_image: Option<ImageHandle>,
    sampler: vk::Sampler,
}

impl App {
    fn new() -> Self {
        Self {
            wsi: None,
            window: None,
            frame: 0,
            scene_vert: None,
            scene_frag: None,
            scene_program: None,
            post_vert: None,
            post_frag: None,
            post_program: None,
            frame_resources: None,
            offscreen_image: None,
            sampler: vk::Sampler::null(),
        }
    }

    fn init_resources(&mut self) {
        let wsi = self.wsi.as_mut().unwrap();
        let device = wsi.device().raw();

        // -- Scene shaders (triangle) --
        let sv = Shader::create(
            device,
            vk::ShaderStageFlags::VERTEX,
            &spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv")),
        )
        .expect("scene vert");
        let sf = Shader::create(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            &spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv")),
        )
        .expect("scene frag");
        let sp = Program::create(device, &[&sv, &sf]).expect("scene program");

        // -- Post-process shaders --
        let pv = Shader::create(
            device,
            vk::ShaderStageFlags::VERTEX,
            &spirv_from_bytes(include_bytes!("../shaders/fullscreen.vert.spv")),
        )
        .expect("post vert");
        let pf = Shader::create(
            device,
            vk::ShaderStageFlags::FRAGMENT,
            &spirv_from_bytes(include_bytes!("../shaders/postprocess.frag.spv")),
        )
        .expect("post frag");
        let pp = Program::create(device, &[&pv, &pf]).expect("post program");

        self.scene_vert = Some(sv);
        self.scene_frag = Some(sf);
        self.scene_program = Some(sp);
        self.post_vert = Some(pv);
        self.post_frag = Some(pf);
        self.post_program = Some(pp);

        self.frame_resources = Some(FrameResources::new(vk::PipelineCache::null()));

        // -- Offscreen render target --
        let format = wsi.swapchain_format();
        let offscreen = wsi
            .device_mut()
            .create_image(&ImageCreateInfo::render_target(
                OFFSCREEN_W,
                OFFSCREEN_H,
                format,
            ))
            .expect("offscreen RT");
        self.offscreen_image = Some(offscreen);

        self.sampler = wsi.device().stock_sampler(StockSampler::LinearClamp);

        log::info!(
            "Multi-pass initialized: {}x{} offscreen RT, format={:?}",
            OFFSCREEN_W,
            OFFSCREEN_H,
            format
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

        let swapchain_extent = wsi.swapchain_extent();
        let format = wsi.swapchain_format();

        self.frame_resources
            .as_mut()
            .unwrap()
            .reset_frame(wsi.device().raw());

        let raw_cmd = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("cmd alloc");
        let mut cmd = CommandBuffer::from_raw(
            raw_cmd,
            CommandBufferType::Graphics,
            wsi.device().raw().clone(),
        );

        // Extract handles before entering the DrawContext borrow
        let offscreen_view = self.offscreen_image.as_ref().unwrap().default_view();
        let offscreen_raw = self.offscreen_image.as_ref().unwrap().raw();
        let sampler = self.sampler;
        let empty_layout = VertexInputLayout::default();

        let offscreen_extent = vk::Extent2D {
            width: OFFSCREEN_W,
            height: OFFSCREEN_H,
        };

        let offscreen_rp = RenderPassInfo {
            color_attachments: vec![ColorAttachmentInfo {
                format,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
            }],
            depth_stencil: None,
            samples: vk::SampleCountFlags::TYPE_1,
        };

        let swapchain_rp = RenderPassInfo {
            color_attachments: vec![ColorAttachmentInfo {
                format,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
            }],
            depth_stencil: None,
            samples: vk::SampleCountFlags::TYPE_1,
        };

        // Cycling clear color for the offscreen pass
        let t = self.frame as f32 * 0.005;
        let offscreen_clear = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [
                    (t.sin() * 0.15 + 0.1).clamp(0.0, 1.0),
                    (t.cos() * 0.15 + 0.1).clamp(0.0, 1.0),
                    0.2,
                    1.0,
                ],
            },
        }];
        let swapchain_clear = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        // -----------------------------------------------------------
        // Transition offscreen RT: UNDEFINED → COLOR_ATTACHMENT
        // -----------------------------------------------------------
        cmd.image_barrier(&ImageBarrierInfo::undefined_to(
            offscreen_raw,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageAspectFlags::COLOR,
        ));

        // -----------------------------------------------------------
        // Pass 1: render spinning triangle to offscreen RT
        // -----------------------------------------------------------
        {
            let device = wsi.device().raw();
            let resources = self.frame_resources.as_mut().unwrap();
            let scene_program = self.scene_program.as_mut().unwrap();
            let post_program = self.post_program.as_mut().unwrap();
            let mut ctx = DrawContext::new(&mut cmd, device, resources);

            ctx.begin_render_pass(
                &offscreen_rp,
                offscreen_extent,
                &offscreen_clear,
                &[offscreen_view],
            )
            .expect("offscreen render pass");
            ctx.set_cull_mode(vk::CullModeFlags::NONE);
            ctx.draw(scene_program, &empty_layout, 3, 1, 0, 0)
                .expect("scene draw");
            ctx.end_render_pass();

            // -----------------------------------------------------------
            // Barrier: offscreen COLOR_ATTACHMENT → SHADER_READ_ONLY
            // -----------------------------------------------------------
            ctx.cmd().image_barrier(&ImageBarrierInfo {
                image: offscreen_raw,
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                dst_access: vk::AccessFlags2::SHADER_READ,
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

            // -----------------------------------------------------------
            // Transition swapchain: UNDEFINED → COLOR_ATTACHMENT
            // -----------------------------------------------------------
            ctx.cmd().image_barrier(&ImageBarrierInfo::undefined_to(
                swapchain_image.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::ImageAspectFlags::COLOR,
            ));

            // -----------------------------------------------------------
            // Pass 2: post-process fullscreen quad → swapchain
            // -----------------------------------------------------------
            ctx.begin_render_pass(
                &swapchain_rp,
                swapchain_extent,
                &swapchain_clear,
                &[swapchain_image.view],
            )
            .expect("swapchain render pass");
            ctx.set_cull_mode(vk::CullModeFlags::NONE);
            ctx.set_texture(0, 0, offscreen_view, sampler);
            ctx.draw(post_program, &empty_layout, 3, 1, 0, 0)
                .expect("post draw");
            ctx.end_render_pass();
        }

        // -----------------------------------------------------------
        // Transition swapchain: COLOR_ATTACHMENT → PRESENT_SRC
        // -----------------------------------------------------------
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
            .with_title("tephra — 03 multi-pass")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop
            .create_window(window_attrs)
            .expect("create window");
        let platform = WinitPlatform::new(&window).expect("winit platform");
        let wsi = WSI::new(&platform, &WSIConfig::default()).expect("WSI");

        self.window = Some(window);
        self.wsi = Some(wsi);
        self.init_resources();
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
            unsafe {
                wsi.device().raw().device_wait_idle().ok();
            }
        }
        if let Some(mut p) = self.scene_program.take() {
            if let Some(wsi) = &self.wsi {
                p.destroy(wsi.device().raw());
            }
        }
        if let Some(mut p) = self.post_program.take() {
            if let Some(wsi) = &self.wsi {
                p.destroy(wsi.device().raw());
            }
        }
        if let Some(mut s) = self.scene_vert.take() {
            if let Some(wsi) = &self.wsi {
                s.destroy(wsi.device().raw());
            }
        }
        if let Some(mut s) = self.scene_frag.take() {
            if let Some(wsi) = &self.wsi {
                s.destroy(wsi.device().raw());
            }
        }
        if let Some(mut s) = self.post_vert.take() {
            if let Some(wsi) = &self.wsi {
                s.destroy(wsi.device().raw());
            }
        }
        if let Some(mut s) = self.post_frag.take() {
            if let Some(wsi) = &self.wsi {
                s.destroy(wsi.device().raw());
            }
        }
        if let Some(mut r) = self.frame_resources.take() {
            if let Some(wsi) = &self.wsi {
                r.destroy(wsi.device().raw());
            }
        }
        if let Some(img) = self.offscreen_image.take() {
            if let Some(wsi) = &mut self.wsi {
                wsi.device_mut().destroy_image(img);
            }
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}
