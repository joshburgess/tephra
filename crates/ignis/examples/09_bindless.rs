//! Bindless descriptor example: multiple textures indexed via push constant.
//!
//! Creates four procedural textures, registers them in a [`BindlessTable`],
//! and draws a full-screen quad that cycles through them using a push constant
//! texture index with `nonuniformEXT` indexing.
//!
//! Requires: Vulkan 1.2 descriptor indexing features and compiled SPIR-V
//! shaders in `shaders/fullscreen.vert.spv` and `shaders/bindless.frag.spv`.

use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use ignis::command::barriers::ImageBarrierInfo;
use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::core::context::QueueType;
use ignis::core::handles::ImageHandle;
use ignis::core::image::ImageCreateInfo;
use ignis::core::sampler::StockSampler;
use ignis::descriptors::bindless::BindlessTable;
use ignis::wsi::platform::WinitPlatform;
use ignis::wsi::wsi::{WSI, WSIConfig};

fn spirv_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Generate a 64x64 RGBA8 texture filled with a solid color.
fn generate_color_texture(r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(64 * 64 * 4);
    for _ in 0..64 * 64 {
        data.extend_from_slice(&[r, g, b, 255]);
    }
    data
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    texture_index: u32,
}

struct App {
    wsi: Option<WSI>,
    window: Option<Window>,
    frame: u64,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    bindless: Option<BindlessTable>,
    textures: Vec<ImageHandle>,
    texture_indices: Vec<u32>,
    sampler: vk::Sampler,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl App {
    fn new() -> Self {
        Self {
            wsi: None,
            window: None,
            frame: 0,
            vert_module: vk::ShaderModule::null(),
            frag_module: vk::ShaderModule::null(),
            bindless: None,
            textures: Vec::new(),
            texture_indices: Vec::new(),
            sampler: vk::Sampler::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        }
    }

    fn init(&mut self) {
        let wsi = self.wsi.as_mut().unwrap();
        let device = wsi.device().raw().clone();

        // Verify descriptor indexing support
        assert!(
            wsi.device().context().device_features().descriptor_indexing,
            "Descriptor indexing not supported — cannot run bindless example"
        );

        // Create raw shader modules (bypass spirv_reflect which can't handle unsized arrays)
        let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/fullscreen.vert.spv"));
        let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/bindless.frag.spv"));

        let vert_ci = vk::ShaderModuleCreateInfo::default().code(&vert_spirv);
        let frag_ci = vk::ShaderModuleCreateInfo::default().code(&frag_spirv);

        // SAFETY: device is valid, shader create infos are well-formed.
        let vert_module = unsafe { device.create_shader_module(&vert_ci, None) }
            .expect("failed to create vertex shader module");
        let frag_module = unsafe { device.create_shader_module(&frag_ci, None) }
            .expect("failed to create fragment shader module");

        // Create bindless table (small capacity for the example)
        let mut bindless =
            BindlessTable::with_capacity(&device, 16, 0).expect("failed to create bindless table");

        // Get a stock linear sampler
        self.sampler = wsi.device().stock_sampler(StockSampler::LinearClamp);

        // Create 4 colored textures and register them
        let colors: [(u8, u8, u8); 4] = [
            (220, 50, 50),  // red
            (50, 200, 50),  // green
            (50, 80, 220),  // blue
            (220, 200, 50), // yellow
        ];

        for (r, g, b) in colors {
            let data = generate_color_texture(r, g, b);
            let image = wsi
                .device_mut()
                .create_image_with_data(
                    &ImageCreateInfo::immutable_2d(64, 64, vk::Format::R8G8B8A8_UNORM),
                    &data,
                )
                .expect("failed to create texture");

            let index = bindless
                .register_texture(
                    &device,
                    image.default_view(),
                    self.sampler,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
                .expect("bindless table full");

            log::info!("Registered texture (r={r}, g={g}, b={b}) at bindless index {index}");
            self.texture_indices.push(index);
            self.textures.push(image);
        }

        // Create pipeline layout using bindless table's layout + push constants
        let set_layouts = [bindless.layout()];
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);
        let push_ranges = [push_range];

        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);

        // SAFETY: device is valid, layout_ci is well-formed.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_ci, None) }
            .expect("failed to create pipeline layout");

        // Create graphics pipeline with dynamic rendering
        let format = wsi.swapchain_format();
        let pipeline = create_pipeline(&device, vert_module, frag_module, pipeline_layout, format);

        self.vert_module = vert_module;
        self.frag_module = frag_module;
        self.bindless = Some(bindless);
        self.pipeline_layout = pipeline_layout;
        self.pipeline = pipeline;

        log::info!(
            "Bindless pipeline initialized with {} textures",
            self.texture_indices.len()
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
        let _format = wsi.swapchain_format();

        let raw_cmd = wsi
            .device()
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("failed to allocate command buffer");
        let device = wsi.device().raw().clone();
        let mut cmd = CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.clone());

        // Transition swapchain image: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
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
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let color_attachments = [color_attachment];

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);

        cmd.begin_rendering(&rendering_info);

        // Bind pipeline and bindless set
        cmd.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        cmd.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.bindless.as_ref().unwrap().descriptor_set()],
            &[],
        );

        // Set viewport and scissor
        cmd.set_viewport(
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        cmd.set_scissor(
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            }],
        );

        // Cycle texture every ~120 frames
        let tex_idx = (self.frame / 120) as usize % self.texture_indices.len();
        let push = PushConstants {
            texture_index: self.texture_indices[tex_idx],
        };

        // SAFETY: push constant data is valid, layout matches.
        unsafe {
            device.cmd_push_constants(
                cmd.raw(),
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
        }

        cmd.draw(3, 1, 0, 0);
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

fn create_pipeline(
    device: &ash::Device,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
) -> vk::Pipeline {
    let entry_name = c"main";

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let color_formats = [color_format];
    let mut rendering_ci =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

    let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(vk::RenderPass::null())
        .push_next(&mut rendering_ci);

    // SAFETY: device is valid, pipeline_ci is well-formed.
    let pipelines = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
    }
    .expect("failed to create graphics pipeline");

    pipelines[0]
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ignis — 09 bindless descriptors")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

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
            // SAFETY: waiting for GPU idle before destroying resources.
            unsafe {
                wsi.device().raw().device_wait_idle().ok();
            }
            let device = wsi.device().raw();

            if self.pipeline != vk::Pipeline::null() {
                unsafe {
                    device.destroy_pipeline(self.pipeline, None);
                }
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                unsafe {
                    device.destroy_pipeline_layout(self.pipeline_layout, None);
                }
            }
        }
        if let Some(mut bindless) = self.bindless.take() {
            if let Some(wsi) = &self.wsi {
                bindless.destroy(wsi.device().raw());
            }
        }
        if let Some(wsi) = &self.wsi {
            let device = wsi.device().raw();
            if self.vert_module != vk::ShaderModule::null() {
                // SAFETY: device is valid, module is valid, GPU is idle.
                unsafe {
                    device.destroy_shader_module(self.vert_module, None);
                }
            }
            if self.frag_module != vk::ShaderModule::null() {
                // SAFETY: device is valid, module is valid, GPU is idle.
                unsafe {
                    device.destroy_shader_module(self.frag_module, None);
                }
            }
        }
        // Textures are destroyed via Device's deferred deletion
        for tex in self.textures.drain(..) {
            if let Some(wsi) = &mut self.wsi {
                wsi.device_mut().destroy_image(tex);
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
