//! Textured spinning quad using DrawContext.
//!
//! Demonstrates vertex/index buffers, texture upload via staging,
//! sampler from cache, combined image sampler descriptor, push constant
//! MVP matrix, and indexed drawing — all through the DrawContext API.
//!
//! The quad spins around the Y axis with a checkerboard texture.
//!
//! Requires: compiled SPIR-V shaders in `shaders/quad.{vert,frag}.spv`.
//! Compile with: `glslc shaders/quad.vert -o shaders/quad.vert.spv`
//!               `glslc shaders/quad.frag -o shaders/quad.frag.spv`

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use ignis::command::barriers::ImageBarrierInfo;
use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::core::buffer::BufferCreateInfo;
use ignis::core::context::QueueType;
use ignis::core::handles::{BufferHandle, ImageHandle};
use ignis::core::image::ImageCreateInfo;
use ignis::core::sampler::StockSampler;
use ignis::pipeline::draw_context::{DrawContext, FrameResources};
use ignis::pipeline::pipeline::{VertexAttribute, VertexBinding, VertexInputLayout};
use ignis::pipeline::program::Program;
use ignis::pipeline::render_pass::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachmentInfo, RenderPassInfo,
};
use ignis::pipeline::shader::Shader;
use ignis::wsi::platform::WinitPlatform;
use ignis::wsi::wsi::{WSI, WSIConfig};

// ---------------------------------------------------------------------------
// Vertex type
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

/// Push constant data: a single MVP matrix.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    mvp: [[f32; 4]; 4],
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

const QUAD_VERTICES: [Vertex; 4] = [
    Vertex { pos: [-0.5, -0.5], uv: [0.0, 0.0] },
    Vertex { pos: [ 0.5, -0.5], uv: [1.0, 0.0] },
    Vertex { pos: [ 0.5,  0.5], uv: [1.0, 1.0] },
    Vertex { pos: [-0.5,  0.5], uv: [0.0, 1.0] },
];

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

// ---------------------------------------------------------------------------
// Math helpers (column-major for GLSL)
// ---------------------------------------------------------------------------

fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fovy * 0.5).tan();
    let nf = near - far;
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0], // negate for Vulkan Y-down
        [0.0, 0.0, far / nf, -1.0],
        [0.0, 0.0, (near * far) / nf, 0.0],
    ]
}

fn translate(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

fn rotate_y(angle: f32) -> [[f32; 4]; 4] {
    let c = angle.cos();
    let s = angle.sin();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut r = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            r[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Texture generation
// ---------------------------------------------------------------------------

fn generate_checkerboard(width: u32, height: u32, squares: u32) -> Vec<u8> {
    let sq_w = width / squares;
    let sq_h = height / squares;
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = if ((x / sq_w) + (y / sq_h)) % 2 == 0 {
                (220u8, 50, 120) // pink
            } else {
                (50u8, 200, 180) // teal
            };
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    data
}

// ---------------------------------------------------------------------------
// SPIR-V loader
// ---------------------------------------------------------------------------

fn spirv_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(
        bytes.len() % 4 == 0,
        "SPIR-V size must be a multiple of 4 bytes"
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    wsi: Option<WSI>,
    window: Option<Window>,
    frame: u64,
    // Pipeline
    vert_shader: Option<Shader>,
    frag_shader: Option<Shader>,
    program: Option<Program>,
    frame_resources: Option<FrameResources>,
    vertex_layout: VertexInputLayout,
    // Geometry
    vertex_buffer: Option<BufferHandle>,
    index_buffer: Option<BufferHandle>,
    // Texture
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
            vertex_layout: VertexInputLayout {
                bindings: vec![VertexBinding {
                    binding: 0,
                    stride: std::mem::size_of::<Vertex>() as u32,
                    input_rate: vk::VertexInputRate::VERTEX,
                }],
                attributes: vec![
                    VertexAttribute {
                        location: 0,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 0,
                    },
                    VertexAttribute {
                        location: 1,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 8,
                    },
                ],
            },
            vertex_buffer: None,
            index_buffer: None,
            texture: None,
            sampler: vk::Sampler::null(),
        }
    }

    fn init_resources(&mut self) {
        let wsi = self.wsi.as_mut().unwrap();

        // -- Shaders & program --
        let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/quad.vert.spv"));
        let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/quad.frag.spv"));

        let vert_shader =
            Shader::create(wsi.device().raw(), vk::ShaderStageFlags::VERTEX, &vert_spirv)
                .expect("failed to create vertex shader");
        let frag_shader =
            Shader::create(wsi.device().raw(), vk::ShaderStageFlags::FRAGMENT, &frag_spirv)
                .expect("failed to create fragment shader");

        let program = Program::create(wsi.device().raw(), &[&vert_shader, &frag_shader])
            .expect("failed to create program");

        self.vert_shader = Some(vert_shader);
        self.frag_shader = Some(frag_shader);
        self.program = Some(program);
        self.frame_resources = Some(FrameResources::new(vk::PipelineCache::null()));

        // -- Vertex & index buffers --
        let vb_data: &[u8] = bytemuck::cast_slice(&QUAD_VERTICES);
        let vertex_buffer = wsi
            .device_mut()
            .create_buffer_with_data(
                &BufferCreateInfo::vertex(vb_data.len() as u64),
                vb_data,
            )
            .expect("failed to create vertex buffer");

        let ib_data: &[u8] = bytemuck::cast_slice(&QUAD_INDICES);
        let index_buffer = wsi
            .device_mut()
            .create_buffer_with_data(
                &BufferCreateInfo::index(ib_data.len() as u64),
                ib_data,
            )
            .expect("failed to create index buffer");

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);

        // -- Checkerboard texture --
        let tex_w = 128;
        let tex_h = 128;
        let tex_data = generate_checkerboard(tex_w, tex_h, 8);
        let texture = wsi
            .device_mut()
            .create_image_with_data(
                &ImageCreateInfo::immutable_2d(tex_w, tex_h, vk::Format::R8G8B8A8_UNORM),
                &tex_data,
            )
            .expect("failed to create texture");
        self.texture = Some(texture);

        // -- Sampler --
        self.sampler = wsi.device().get_stock_sampler(StockSampler::LinearClamp);

        log::info!(
            "Resources initialized: shaders, buffers, {}x{} checkerboard texture",
            tex_w,
            tex_h
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

        self.frame_resources
            .as_mut()
            .unwrap()
            .reset_frame(wsi.device().raw());

        // Allocate command buffer
        let raw_cmd = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("failed to allocate command buffer");
        let mut cmd = CommandBuffer::from_raw(
            raw_cmd,
            CommandBufferType::Graphics,
            wsi.device().raw().clone(),
        );

        // Transition swapchain: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
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
                float32: [0.08, 0.08, 0.12, 1.0],
            },
        }];

        // Compute MVP
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let angle = self.frame as f32 * 0.02;
        let proj = perspective(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let view = translate(0.0, 0.0, -2.0);
        let model = rotate_y(angle);
        let mvp = mat4_mul(&proj, &mat4_mul(&view, &model));
        let pc = PushConstants { mvp };

        // Draw
        {
            let device = wsi.device().raw();
            let resources = self.frame_resources.as_mut().unwrap();
            let mut ctx = DrawContext::new(&mut cmd, device, resources);

            ctx.begin_render_pass(
                &rp_info,
                extent,
                &clear_values,
                &[swapchain_image.view],
            )
            .expect("failed to begin render pass");

            ctx.set_cull_mode(vk::CullModeFlags::NONE);

            // Bind texture
            let tex = self.texture.as_ref().unwrap();
            ctx.set_texture(0, 0, tex.default_view(), self.sampler);

            // Bind vertex & index buffers
            let vb = self.vertex_buffer.as_ref().unwrap();
            let ib = self.index_buffer.as_ref().unwrap();
            ctx.bind_vertex_buffers(0, &[vb.raw()], &[0]);
            ctx.bind_index_buffer(ib.raw(), 0, vk::IndexType::UINT16);

            // Push MVP
            let program = self.program.as_mut().unwrap();
            ctx.push_constants_typed(
                program,
                vk::ShaderStageFlags::VERTEX,
                &pc,
            );

            // Draw indexed quad
            ctx.draw_indexed(program, &self.vertex_layout, 6, 1, 0, 0, 0)
                .expect("draw_indexed failed");

            ctx.end_render_pass();
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
            .with_title("ignis — 02 textured quad")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop
            .create_window(window_attrs)
            .expect("failed to create window");

        let platform = WinitPlatform::new(&window).expect("failed to create winit platform");
        let wsi = WSI::new(&platform, &WSIConfig::default()).expect("failed to initialize WSI");

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
            // SAFETY: wait for GPU to be idle before destroying resources
            unsafe {
                wsi.device().raw().device_wait_idle().ok();
            }
        }

        // Destroy program and shaders (raw Vulkan handles)
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
        if let Some(mut resources) = self.frame_resources.take() {
            if let Some(wsi) = &self.wsi {
                resources.destroy(wsi.device().raw());
            }
        }

        // Destroy buffers and images (scheduled for deferred deletion)
        if let Some(buffer) = self.vertex_buffer.take() {
            if let Some(wsi) = &mut self.wsi {
                wsi.device_mut().destroy_buffer(buffer);
            }
        }
        if let Some(buffer) = self.index_buffer.take() {
            if let Some(wsi) = &mut self.wsi {
                wsi.device_mut().destroy_buffer(buffer);
            }
        }
        if let Some(image) = self.texture.take() {
            if let Some(wsi) = &mut self.wsi {
                wsi.device_mut().destroy_image(image);
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
