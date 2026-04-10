//! Asynchronous pipeline compilation with background worker threads.
//!
//! [`AsyncPipelineCompiler`] compiles pipelines on background threads so that
//! the main thread doesn't stall when encountering new pipeline state combinations.
//! On a cache miss, a compilation job is submitted and `None` is returned.
//! Call [`AsyncPipelineCompiler::poll_completed`] each frame to collect results.
//!
//! # Usage
//!
//! ```ignore
//! let mut compiler = AsyncPipelineCompiler::new(device.clone(), pipeline_cache);
//!
//! // Each frame:
//! compiler.poll_completed();
//!
//! // When drawing:
//! if let Some(pipeline) = compiler.request_graphics_dynamic(&program, &state, ...) {
//!     cmd.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
//!     cmd.draw(3, 1, 0, 0);
//! }
//! // Pipeline not ready -- draw skipped, will be available next frame
//! ```

use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use ash::vk;
use log::{debug, warn};
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHasher};

use ignis_command::state::StaticPipelineState;

use crate::pipeline::{
    build_compute_pipeline, build_dynamic_graphics_pipeline, build_graphics_pipeline,
    VertexInputLayout,
};
use crate::program::Program;

/// Status of a pipeline in the async compilation cache.
#[derive(Clone, Copy)]
enum PipelineStatus {
    /// Compilation is in progress on a worker thread.
    Pending,
    /// Pipeline is compiled and ready to use.
    Ready(vk::Pipeline),
    /// Compilation failed.
    Failed,
}

/// Which cache a completed pipeline belongs to.
enum CacheType {
    Graphics,
    DynamicGraphics,
    Compute,
}

/// A compilation job sent to a worker thread.
enum CompilationJob {
    Graphics {
        key_hash: u64,
        shaders: Vec<(vk::ShaderModule, vk::ShaderStageFlags)>,
        pipeline_layout: vk::PipelineLayout,
        state: StaticPipelineState,
        render_pass: vk::RenderPass,
        subpass: u32,
        vertex_layout: VertexInputLayout,
    },
    DynamicGraphics {
        key_hash: u64,
        shaders: Vec<(vk::ShaderModule, vk::ShaderStageFlags)>,
        pipeline_layout: vk::PipelineLayout,
        state: StaticPipelineState,
        color_formats: Vec<vk::Format>,
        depth_format: vk::Format,
        stencil_format: vk::Format,
        vertex_layout: VertexInputLayout,
    },
    Compute {
        key_hash: u64,
        module: vk::ShaderModule,
        pipeline_layout: vk::PipelineLayout,
    },
}

/// A completed pipeline returned from a worker.
struct CompletedPipeline {
    key_hash: u64,
    result: Result<vk::Pipeline, vk::Result>,
    cache_type: CacheType,
}

/// Shared state between the main thread and worker threads.
struct SharedState {
    device: ash::Device,
    pipeline_cache: Mutex<vk::PipelineCache>,
}

/// Asynchronous pipeline compiler with background worker threads.
///
/// Compiles pipelines on background threads to avoid stalling the render loop.
/// On a cache miss, a compilation job is submitted and `None` is returned.
/// Call [`poll_completed`](Self::poll_completed) each frame to collect results.
///
/// Worker count is `available_parallelism() / 2`, clamped to `[1, 4]`.
/// The `VkPipelineCache` is shared across workers with a mutex to maintain
/// Vulkan external synchronization requirements.
pub struct AsyncPipelineCompiler {
    // Kept alive to ensure SharedState (device + pipeline cache) outlives workers.
    #[allow(dead_code)]
    shared: Arc<SharedState>,
    job_sender: Option<mpsc::Sender<CompilationJob>>,
    result_receiver: mpsc::Receiver<CompletedPipeline>,
    workers: Vec<JoinHandle<()>>,
    graphics_cache: FxHashMap<u64, PipelineStatus>,
    dynamic_graphics_cache: FxHashMap<u64, PipelineStatus>,
    compute_cache: FxHashMap<u64, PipelineStatus>,
}

impl AsyncPipelineCompiler {
    /// Create a new async pipeline compiler with background worker threads.
    pub fn new(device: ash::Device, pipeline_cache: vk::PipelineCache) -> Self {
        let worker_count = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let worker_count = (worker_count / 2).clamp(1, 4);

        let shared = Arc::new(SharedState {
            device,
            pipeline_cache: Mutex::new(pipeline_cache),
        });

        let (job_sender, job_receiver) = mpsc::channel::<CompilationJob>();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let (result_sender, result_receiver) = mpsc::channel::<CompletedPipeline>();

        let mut workers = Vec::with_capacity(worker_count);
        for i in 0..worker_count {
            let shared = Arc::clone(&shared);
            let receiver = Arc::clone(&job_receiver);
            let sender = result_sender.clone();

            let handle = thread::Builder::new()
                .name(format!("pipeline-worker-{i}"))
                .spawn(move || {
                    worker_loop(shared, receiver, sender);
                })
                .expect("failed to spawn pipeline worker thread");

            workers.push(handle);
        }

        debug!(
            "Created async pipeline compiler with {} worker threads",
            worker_count
        );

        Self {
            shared,
            job_sender: Some(job_sender),
            result_receiver,
            workers,
            graphics_cache: FxHashMap::default(),
            dynamic_graphics_cache: FxHashMap::default(),
            compute_cache: FxHashMap::default(),
        }
    }

    /// Drain completed compilations into the cache.
    ///
    /// Call this at the start of each frame to make newly compiled pipelines
    /// available for drawing.
    pub fn poll_completed(&mut self) {
        while let Ok(completed) = self.result_receiver.try_recv() {
            let status = match completed.result {
                Ok(pipeline) => {
                    debug!(
                        "Async pipeline compiled (hash={:#x})",
                        completed.key_hash
                    );
                    PipelineStatus::Ready(pipeline)
                }
                Err(err) => {
                    warn!(
                        "Async pipeline compilation failed: {:?} (hash={:#x})",
                        err, completed.key_hash
                    );
                    PipelineStatus::Failed
                }
            };

            let cache = match completed.cache_type {
                CacheType::Graphics => &mut self.graphics_cache,
                CacheType::DynamicGraphics => &mut self.dynamic_graphics_cache,
                CacheType::Compute => &mut self.compute_cache,
            };

            cache.insert(completed.key_hash, status);
        }
    }

    /// Request a graphics pipeline (legacy render pass path).
    ///
    /// Returns `Some(pipeline)` if ready, `None` if pending or failed.
    /// On first request for a new key, submits a compilation job to a worker.
    #[allow(clippy::too_many_arguments)]
    pub fn request_graphics(
        &mut self,
        program: &Program,
        state: &StaticPipelineState,
        render_pass: vk::RenderPass,
        render_pass_hash: u64,
        subpass: u32,
        vertex_layout: &VertexInputLayout,
    ) -> Option<vk::Pipeline> {
        let key_hash = {
            let mut hasher = FxHasher::default();
            program.layout_hash().hash(&mut hasher);
            state.hash(&mut hasher);
            render_pass_hash.hash(&mut hasher);
            subpass.hash(&mut hasher);
            vertex_layout.compute_hash().hash(&mut hasher);
            hasher.finish()
        };

        match self.graphics_cache.get(&key_hash) {
            Some(PipelineStatus::Ready(pipeline)) => return Some(*pipeline),
            Some(PipelineStatus::Pending | PipelineStatus::Failed) => return None,
            None => {}
        }

        let shaders: Vec<_> = program
            .shaders()
            .iter()
            .map(|s| (s.module, s.stage))
            .collect();

        let job = CompilationJob::Graphics {
            key_hash,
            shaders,
            pipeline_layout: program.pipeline_layout(),
            state: state.clone(),
            render_pass,
            subpass,
            vertex_layout: vertex_layout.clone(),
        };

        self.graphics_cache
            .insert(key_hash, PipelineStatus::Pending);
        if let Some(sender) = &self.job_sender {
            let _ = sender.send(job);
        }

        debug!(
            "Submitted async graphics pipeline (hash={:#x})",
            key_hash
        );
        None
    }

    /// Request a graphics pipeline (dynamic rendering path).
    ///
    /// Returns `Some(pipeline)` if ready, `None` if pending or failed.
    /// On first request for a new key, submits a compilation job to a worker.
    #[allow(clippy::too_many_arguments)]
    pub fn request_graphics_dynamic(
        &mut self,
        program: &Program,
        state: &StaticPipelineState,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
        stencil_format: vk::Format,
        vertex_layout: &VertexInputLayout,
    ) -> Option<vk::Pipeline> {
        let key_hash = {
            let mut hasher = FxHasher::default();
            program.layout_hash().hash(&mut hasher);
            state.hash(&mut hasher);
            color_formats.hash(&mut hasher);
            depth_format.hash(&mut hasher);
            stencil_format.hash(&mut hasher);
            vertex_layout.compute_hash().hash(&mut hasher);
            hasher.finish()
        };

        match self.dynamic_graphics_cache.get(&key_hash) {
            Some(PipelineStatus::Ready(pipeline)) => return Some(*pipeline),
            Some(PipelineStatus::Pending | PipelineStatus::Failed) => return None,
            None => {}
        }

        let shaders: Vec<_> = program
            .shaders()
            .iter()
            .map(|s| (s.module, s.stage))
            .collect();

        let job = CompilationJob::DynamicGraphics {
            key_hash,
            shaders,
            pipeline_layout: program.pipeline_layout(),
            state: state.clone(),
            color_formats: color_formats.to_vec(),
            depth_format,
            stencil_format,
            vertex_layout: vertex_layout.clone(),
        };

        self.dynamic_graphics_cache
            .insert(key_hash, PipelineStatus::Pending);
        if let Some(sender) = &self.job_sender {
            let _ = sender.send(job);
        }

        debug!(
            "Submitted async dynamic pipeline (hash={:#x})",
            key_hash
        );
        None
    }

    /// Request a compute pipeline.
    ///
    /// Returns `Some(pipeline)` if ready, `None` if pending or failed.
    /// On first request for a new key, submits a compilation job to a worker.
    pub fn request_compute(&mut self, program: &Program) -> Option<vk::Pipeline> {
        let key_hash = program.layout_hash();

        match self.compute_cache.get(&key_hash) {
            Some(PipelineStatus::Ready(pipeline)) => return Some(*pipeline),
            Some(PipelineStatus::Pending | PipelineStatus::Failed) => return None,
            None => {}
        }

        let compute_shader = program
            .shaders()
            .iter()
            .find(|s| s.stage == vk::ShaderStageFlags::COMPUTE);

        let Some(shader) = compute_shader else {
            warn!("No compute shader stage in program for async compilation");
            return None;
        };

        let job = CompilationJob::Compute {
            key_hash,
            module: shader.module,
            pipeline_layout: program.pipeline_layout(),
        };

        self.compute_cache
            .insert(key_hash, PipelineStatus::Pending);
        if let Some(sender) = &self.job_sender {
            let _ = sender.send(job);
        }

        debug!(
            "Submitted async compute pipeline (hash={:#x})",
            key_hash
        );
        None
    }

    /// Number of pipelines currently pending compilation.
    pub fn pending_count(&self) -> usize {
        fn count_pending(cache: &FxHashMap<u64, PipelineStatus>) -> usize {
            cache
                .values()
                .filter(|s| matches!(s, PipelineStatus::Pending))
                .count()
        }
        count_pending(&self.graphics_cache)
            + count_pending(&self.dynamic_graphics_cache)
            + count_pending(&self.compute_cache)
    }

    /// Number of pipelines that have been compiled and are ready to use.
    pub fn ready_count(&self) -> usize {
        fn count_ready(cache: &FxHashMap<u64, PipelineStatus>) -> usize {
            cache
                .values()
                .filter(|s| matches!(s, PipelineStatus::Ready(_)))
                .count()
        }
        count_ready(&self.graphics_cache)
            + count_ready(&self.dynamic_graphics_cache)
            + count_ready(&self.compute_cache)
    }

    /// Shut down worker threads and destroy all cached pipelines.
    ///
    /// The GPU must be idle before calling this.
    pub fn destroy(&mut self, device: &ash::Device) {
        // Drop sender to signal workers to exit
        self.job_sender = None;

        // Join worker threads
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }

        // Drain any remaining completed pipelines
        while let Ok(completed) = self.result_receiver.try_recv() {
            if let Ok(pipeline) = completed.result {
                let cache = match completed.cache_type {
                    CacheType::Graphics => &mut self.graphics_cache,
                    CacheType::DynamicGraphics => &mut self.dynamic_graphics_cache,
                    CacheType::Compute => &mut self.compute_cache,
                };
                cache.insert(completed.key_hash, PipelineStatus::Ready(pipeline));
            }
        }

        // Destroy all cached pipelines
        for cache in [
            &mut self.graphics_cache,
            &mut self.dynamic_graphics_cache,
            &mut self.compute_cache,
        ] {
            for status in cache.values() {
                if let PipelineStatus::Ready(pipeline) = status {
                    // SAFETY: device is valid, pipeline is valid, GPU is idle.
                    unsafe {
                        device.destroy_pipeline(*pipeline, None);
                    }
                }
            }
            cache.clear();
        }

        debug!("Async pipeline compiler destroyed");
    }
}

/// Worker thread main loop.
///
/// Receives compilation jobs from the shared receiver, compiles pipelines,
/// and sends results back to the main thread.
fn worker_loop(
    shared: Arc<SharedState>,
    job_receiver: Arc<Mutex<mpsc::Receiver<CompilationJob>>>,
    result_sender: mpsc::Sender<CompletedPipeline>,
) {
    loop {
        // Lock receiver only long enough to dequeue one job
        let job = {
            let receiver = job_receiver.lock();
            match receiver.recv() {
                Ok(job) => job,
                Err(_) => return, // Channel closed — exit
            }
        };

        let completed = match job {
            CompilationJob::Graphics {
                key_hash,
                shaders,
                pipeline_layout,
                state,
                render_pass,
                subpass,
                vertex_layout,
            } => {
                // Hold pipeline cache lock during vkCreateGraphicsPipelines
                // for Vulkan external synchronization.
                let result = {
                    let pipeline_cache = shared.pipeline_cache.lock();
                    build_graphics_pipeline(
                        &shared.device,
                        *pipeline_cache,
                        &shaders,
                        pipeline_layout,
                        &state,
                        render_pass,
                        subpass,
                        &vertex_layout,
                    )
                };
                CompletedPipeline {
                    key_hash,
                    result,
                    cache_type: CacheType::Graphics,
                }
            }
            CompilationJob::DynamicGraphics {
                key_hash,
                shaders,
                pipeline_layout,
                state,
                color_formats,
                depth_format,
                stencil_format,
                vertex_layout,
            } => {
                let result = {
                    let pipeline_cache = shared.pipeline_cache.lock();
                    build_dynamic_graphics_pipeline(
                        &shared.device,
                        *pipeline_cache,
                        &shaders,
                        pipeline_layout,
                        &state,
                        &color_formats,
                        depth_format,
                        stencil_format,
                        &vertex_layout,
                    )
                };
                CompletedPipeline {
                    key_hash,
                    result,
                    cache_type: CacheType::DynamicGraphics,
                }
            }
            CompilationJob::Compute {
                key_hash,
                module,
                pipeline_layout,
            } => {
                let result = {
                    let pipeline_cache = shared.pipeline_cache.lock();
                    build_compute_pipeline(
                        &shared.device,
                        *pipeline_cache,
                        module,
                        pipeline_layout,
                    )
                };
                CompletedPipeline {
                    key_hash,
                    result,
                    cache_type: CacheType::Compute,
                }
            }
        };

        if result_sender.send(completed).is_err() {
            return; // Main thread dropped receiver — exit
        }
    }
}
