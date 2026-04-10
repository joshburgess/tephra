//! Async texture upload via the dedicated transfer queue.
//!
//! Demonstrates multi-queue operation with queue family ownership transfer:
//!   1. Uploads a 256x256 gradient texture using the transfer queue.
//!   2. Releases ownership from the transfer queue family.
//!   3. Acquires ownership on the graphics queue family (semaphore-synchronized).
//!   4. Renders the texture fullscreen to verify correctness.
//!
//! On hardware where the transfer queue family equals the graphics family,
//! the upload still runs on the "transfer" queue but ownership barriers are
//! replaced with regular layout transitions.
//!
//! Requires compiled SPIR-V shaders in `shaders/`:
//!   fullscreen.vert.spv, quad.frag.spv

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use tephra::command::barriers::ImageBarrierInfo;
use tephra::command::command_buffer::{CommandBuffer, CommandBufferType};
use tephra::core::buffer::BufferCreateInfo;
use tephra::core::context::QueueType;
use tephra::core::handles::ImageHandle;
use tephra::core::image::ImageCreateInfo;
use tephra::core::memory::MemoryDomain;
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

const TEX_SIZE: u32 = 256;

fn spirv_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Generate a 256x256 RGBA8 radial gradient with concentric rings.
fn generate_gradient_texture() -> Vec<u8> {
    let mut data = vec![0u8; (TEX_SIZE * TEX_SIZE * 4) as usize];
    let center = TEX_SIZE as f32 / 2.0;

    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt() / center;
            let ring = ((dist * 12.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);

            let r = ((1.0 - dist) * ring * 255.0) as u8;
            let g = (dist * ring * 200.0) as u8;
            let b = (ring * 255.0) as u8;

            let idx = ((y * TEX_SIZE + x) * 4) as usize;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    }
    data
}

struct App {
    wsi: Option<WSI>,
    window: Option<Window>,
    frame: u64,
    vert_shader: Option<Shader>,
    frag_shader: Option<Shader>,
    program: Option<Program>,
    frame_resources: Option<FrameResources>,
    texture: Option<ImageHandle>,
    sampler: vk::Sampler,
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
            texture: None,
            sampler: vk::Sampler::null(),
        }
    }

    fn init_resources(&mut self) {
        let wsi = self.wsi.as_mut().unwrap();
        let device_raw = wsi.device().raw().clone();

        // -- Shaders: fullscreen triangle + texture sampler --
        let sv = Shader::create(
            &device_raw,
            vk::ShaderStageFlags::VERTEX,
            &spirv_from_bytes(include_bytes!("../shaders/fullscreen.vert.spv")),
        )
        .expect("vert shader");
        let sf = Shader::create(
            &device_raw,
            vk::ShaderStageFlags::FRAGMENT,
            &spirv_from_bytes(include_bytes!("../shaders/quad.frag.spv")),
        )
        .expect("frag shader");
        let sp = Program::create(&device_raw, &[&sv, &sf]).expect("program");

        self.vert_shader = Some(sv);
        self.frag_shader = Some(sf);
        self.program = Some(sp);
        self.frame_resources = Some(FrameResources::new(vk::PipelineCache::null()));

        // -- Query queue family topology --
        let gfx_family = wsi
            .device()
            .context()
            .queue(QueueType::Graphics)
            .family_index;
        let xfer_family = wsi
            .device()
            .context()
            .queue(QueueType::Transfer)
            .family_index;
        let needs_ownership_transfer = gfx_family != xfer_family;

        log::info!(
            "Queue families — graphics: {}, transfer: {} (ownership transfer: {})",
            gfx_family,
            xfer_family,
            needs_ownership_transfer
        );

        // -- Create the texture image (device-local, SAMPLED | TRANSFER_DST) --
        let tex_data = generate_gradient_texture();
        let image_info = ImageCreateInfo {
            width: TEX_SIZE,
            height: TEX_SIZE,
            depth: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            initial_layout: vk::ImageLayout::UNDEFINED,
            domain: tephra::core::memory::ImageDomain::Physical,
        };
        let texture = wsi.device_mut().create_image(&image_info).expect("texture");

        // -- Create staging buffer --
        let staging_info = BufferCreateInfo {
            size: tex_data.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            domain: MemoryDomain::Host,
        };
        let mut staging = wsi
            .device_mut()
            .create_buffer(&staging_info)
            .expect("staging");
        if let Some(slice) = staging.mapped_slice_mut() {
            slice[..tex_data.len()].copy_from_slice(&tex_data);
        }

        // -- Create sync objects --
        let sem_ci = vk::SemaphoreCreateInfo::default();
        let fence_ci = vk::FenceCreateInfo::default();
        // SAFETY: device is valid.
        let xfer_sem = unsafe { device_raw.create_semaphore(&sem_ci, None) }.expect("semaphore");
        let xfer_fence = unsafe { device_raw.create_fence(&fence_ci, None) }.expect("fence");
        let gfx_fence = unsafe { device_raw.create_fence(&fence_ci, None) }.expect("fence");

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth: 1,
            });

        // =====================================================================
        // Transfer queue: UNDEFINED -> TRANSFER_DST, copy, release ownership
        // =====================================================================
        let xfer_cmd_raw = wsi
            .device()
            .request_command_buffer_raw(QueueType::Transfer)
            .expect("transfer cmd");
        let mut xfer_cmd = CommandBuffer::from_raw(
            xfer_cmd_raw,
            CommandBufferType::Transfer,
            device_raw.clone(),
        );

        // Transition UNDEFINED -> TRANSFER_DST_OPTIMAL
        xfer_cmd.image_barrier(&ImageBarrierInfo::undefined_to(
            texture.raw(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageAspectFlags::COLOR,
        ));

        // Copy staging buffer -> image
        xfer_cmd.copy_buffer_to_image(
            staging.raw(),
            texture.raw(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );

        // Release ownership to graphics queue (or just finish if same family)
        if needs_ownership_transfer {
            xfer_cmd.image_barrier(&ImageBarrierInfo {
                image: texture.raw(),
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::TRANSFER,
                src_access: vk::AccessFlags2::TRANSFER_WRITE,
                dst_stage: vk::PipelineStageFlags2::NONE,
                dst_access: vk::AccessFlags2::NONE,
                subresource_range,
                src_queue_family: xfer_family,
                dst_queue_family: gfx_family,
            });
        }

        // Submit transfer work
        wsi.device_mut()
            .submit_command_buffer(
                xfer_cmd.raw(),
                QueueType::Transfer,
                &[],
                &[],
                &[xfer_sem],
                xfer_fence,
            )
            .expect("submit transfer");

        // =====================================================================
        // Graphics queue: acquire ownership + transition to SHADER_READ_ONLY
        // =====================================================================
        let gfx_cmd_raw = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("graphics cmd");
        let mut gfx_cmd =
            CommandBuffer::from_raw(gfx_cmd_raw, CommandBufferType::Graphics, device_raw.clone());

        if needs_ownership_transfer {
            // Acquire ownership from transfer queue
            gfx_cmd.image_barrier(&ImageBarrierInfo {
                image: texture.raw(),
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::NONE,
                src_access: vk::AccessFlags2::NONE,
                dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                dst_access: vk::AccessFlags2::SHADER_READ,
                subresource_range,
                src_queue_family: xfer_family,
                dst_queue_family: gfx_family,
            });
        } else {
            // Same queue family — just transition the layout
            gfx_cmd.image_barrier(&ImageBarrierInfo {
                image: texture.raw(),
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::TRANSFER,
                src_access: vk::AccessFlags2::TRANSFER_WRITE,
                dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                dst_access: vk::AccessFlags2::SHADER_READ,
                subresource_range,
                src_queue_family: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
            });
        }

        // Submit graphics work, waiting on the transfer semaphore
        wsi.device_mut()
            .submit_command_buffer(
                gfx_cmd.raw(),
                QueueType::Graphics,
                &[xfer_sem],
                &[vk::PipelineStageFlags::FRAGMENT_SHADER],
                &[],
                gfx_fence,
            )
            .expect("submit graphics");

        // Wait for both queues to finish
        // SAFETY: fences are valid.
        unsafe {
            device_raw
                .wait_for_fences(&[xfer_fence, gfx_fence], true, u64::MAX)
                .expect("wait fences");
        }

        log::info!(
            "Async transfer complete: {}x{} texture uploaded via queue family {}",
            TEX_SIZE,
            TEX_SIZE,
            xfer_family
        );

        // Clean up sync objects and staging buffer
        // SAFETY: GPU work complete (fences signaled).
        unsafe {
            device_raw.destroy_semaphore(xfer_sem, None);
            device_raw.destroy_fence(xfer_fence, None);
            device_raw.destroy_fence(gfx_fence, None);
        }
        wsi.device_mut().destroy_buffer(staging);

        self.texture = Some(texture);
        self.sampler = wsi.device().stock_sampler(StockSampler::LinearClamp);
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

        let texture_view = self.texture.as_ref().unwrap().default_view();
        let sampler = self.sampler;
        let empty_layout = VertexInputLayout::default();

        // Transition swapchain: UNDEFINED -> COLOR_ATTACHMENT
        cmd.image_barrier(&ImageBarrierInfo::undefined_to(
            swapchain_image.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageAspectFlags::COLOR,
        ));

        let rp_info = RenderPassInfo {
            color_attachments: vec![ColorAttachmentInfo {
                format,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
            }],
            depth_stencil: None,
            samples: vk::SampleCountFlags::TYPE_1,
        };

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        // Render fullscreen quad sampling the async-transferred texture
        {
            let device = wsi.device().raw();
            let resources = self.frame_resources.as_mut().unwrap();
            let program = self.program.as_mut().unwrap();
            let mut ctx = DrawContext::new(&mut cmd, device, resources);

            ctx.begin_render_pass(&rp_info, extent, &clear_values, &[swapchain_image.view])
                .expect("begin render pass");

            ctx.set_cull_mode(vk::CullModeFlags::NONE);
            ctx.set_texture(0, 0, texture_view, sampler);
            ctx.draw(program, &empty_layout, 3, 1, 0, 0).expect("draw");
            ctx.end_render_pass();
        }

        // Transition swapchain: COLOR_ATTACHMENT -> PRESENT_SRC
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
            .with_title("tephra — 06 async transfer")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 800));

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
            // SAFETY: wait for GPU idle before destroying resources.
            unsafe {
                wsi.device().raw().device_wait_idle().ok();
            }
        }
        if let Some(mut p) = self.program.take() {
            if let Some(wsi) = &self.wsi {
                p.destroy(wsi.device().raw());
            }
        }
        if let Some(mut s) = self.vert_shader.take() {
            if let Some(wsi) = &self.wsi {
                s.destroy(wsi.device().raw());
            }
        }
        if let Some(mut s) = self.frag_shader.take() {
            if let Some(wsi) = &self.wsi {
                s.destroy(wsi.device().raw());
            }
        }
        if let Some(mut r) = self.frame_resources.take() {
            if let Some(wsi) = &self.wsi {
                r.destroy(wsi.device().raw());
            }
        }
        if let Some(img) = self.texture.take() {
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
