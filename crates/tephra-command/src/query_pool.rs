//! GPU query pool for timestamp and occlusion queries.
//!
//! [`QueryPool`] wraps a `VkQueryPool` with auto-growing capacity and
//! named timing intervals. [`TimestampQueryPool`] specializes for GPU
//! timestamp collection with named regions and readback.

use ash::vk;
use rustc_hash::FxHashMap;

/// A managed Vulkan query pool with auto-growing capacity.
pub struct QueryPool {
    pool: vk::QueryPool,
    query_type: vk::QueryType,
    capacity: u32,
    next_index: u32,
}

impl QueryPool {
    /// Create a new query pool.
    pub fn new(
        device: &ash::Device,
        query_type: vk::QueryType,
        initial_capacity: u32,
    ) -> Result<Self, vk::Result> {
        let ci = vk::QueryPoolCreateInfo::default()
            .query_type(query_type)
            .query_count(initial_capacity);
        // SAFETY: device is valid, ci is well-formed.
        let pool = unsafe { device.create_query_pool(&ci, None)? };

        Ok(Self {
            pool,
            query_type,
            capacity: initial_capacity,
            next_index: 0,
        })
    }

    /// The raw `VkQueryPool` handle.
    pub fn raw(&self) -> vk::QueryPool {
        self.pool
    }

    /// The query type of this pool.
    pub fn query_type(&self) -> vk::QueryType {
        self.query_type
    }

    /// Allocate the next query index.
    ///
    /// Returns `None` if the pool is full and needs to be recreated.
    pub fn allocate(&mut self) -> Option<u32> {
        if self.next_index < self.capacity {
            let idx = self.next_index;
            self.next_index += 1;
            Some(idx)
        } else {
            None
        }
    }

    /// Reset the pool using host reset (`vkResetQueryPool`).
    ///
    /// Requires Vulkan 1.2+ (host query reset is core in 1.2).
    pub fn host_reset(&mut self, device: &ash::Device) {
        // SAFETY: device and pool are valid.
        unsafe {
            device.reset_query_pool(self.pool, 0, self.capacity);
        }
        self.next_index = 0;
    }

    /// Number of queries allocated this frame.
    pub fn allocated_count(&self) -> u32 {
        self.next_index
    }

    /// Read back query results as `u64` values.
    ///
    /// Returns results for queries `0..count`.
    pub fn get_results_u64(
        &self,
        device: &ash::Device,
        first: u32,
        count: u32,
    ) -> Result<Vec<u64>, vk::Result> {
        let mut results = vec![0u64; count as usize];
        // SAFETY: device and pool are valid, results buffer is correctly sized.
        unsafe {
            device.get_query_pool_results(
                self.pool,
                first,
                &mut results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        Ok(results)
    }

    /// Destroy the query pool.
    pub fn destroy(&mut self, device: &ash::Device) {
        if self.pool != vk::QueryPool::null() {
            // SAFETY: device is valid, pool is valid.
            unsafe {
                device.destroy_query_pool(self.pool, None);
            }
            self.pool = vk::QueryPool::null();
        }
    }
}

/// A named timestamp interval (begin/end pair).
struct TimestampInterval {
    begin_index: u32,
    end_index: u32,
}

/// GPU timestamp query pool with named timing regions.
///
/// Provides `begin(name)` / `end(name)` pairs that map to GPU timestamp
/// queries. Call [`readback`](Self::readback) after the GPU work completes
/// to collect results as nanoseconds.
///
/// # Usage
///
/// ```ignore
/// // At frame start:
/// timestamps.reset(device);
///
/// // During recording:
/// timestamps.begin(&mut cmd, "shadow_pass");
/// // ... record shadow pass ...
/// timestamps.end(&mut cmd, "shadow_pass");
///
/// // After GPU completes (next frame):
/// let results = timestamps.readback(device, timestamp_period);
/// for (name, ns) in &results {
///     println!("{name}: {:.2} ms", ns / 1_000_000.0);
/// }
/// ```
pub struct TimestampQueryPool {
    pool: QueryPool,
    intervals: FxHashMap<String, TimestampInterval>,
}

impl TimestampQueryPool {
    /// Create a new timestamp query pool.
    pub fn new(device: &ash::Device, max_timestamps: u32) -> Result<Self, vk::Result> {
        Ok(Self {
            pool: QueryPool::new(device, vk::QueryType::TIMESTAMP, max_timestamps)?,
            intervals: FxHashMap::default(),
        })
    }

    /// Reset all queries for the next frame.
    pub fn host_reset(&mut self, device: &ash::Device) {
        self.pool.host_reset(device);
        self.intervals.clear();
    }

    /// Write a single timestamp at the given pipeline stage.
    ///
    /// Returns the query index, or `None` if the pool is full.
    pub fn write_timestamp(
        &mut self,
        cmd: &mut super::command_buffer::CommandBuffer,
        stage: vk::PipelineStageFlags2,
    ) -> Option<u32> {
        let idx = self.pool.allocate()?;
        cmd.write_timestamp(stage, self.pool.raw(), idx);
        Some(idx)
    }

    /// Begin a named timestamp region.
    ///
    /// Records a timestamp at the given pipeline stage.
    pub fn begin(
        &mut self,
        cmd: &mut super::command_buffer::CommandBuffer,
        name: &str,
        stage: vk::PipelineStageFlags2,
    ) {
        let Some(idx) = self.pool.allocate() else {
            return;
        };
        cmd.write_timestamp(stage, self.pool.raw(), idx);
        self.intervals.insert(
            name.to_string(),
            TimestampInterval {
                begin_index: idx,
                end_index: u32::MAX,
            },
        );
    }

    /// End a named timestamp region.
    ///
    /// Records a timestamp at the given pipeline stage.
    pub fn end(
        &mut self,
        cmd: &mut super::command_buffer::CommandBuffer,
        name: &str,
        stage: vk::PipelineStageFlags2,
    ) {
        let Some(idx) = self.pool.allocate() else {
            return;
        };
        cmd.write_timestamp(stage, self.pool.raw(), idx);
        if let Some(interval) = self.intervals.get_mut(name) {
            interval.end_index = idx;
        }
    }

    /// Read back all completed intervals as (name, nanoseconds) pairs.
    ///
    /// `timestamp_period` is from `VkPhysicalDeviceProperties::limits::timestampPeriod`
    /// (nanoseconds per timestamp tick).
    ///
    /// Must only be called after the GPU work is complete (e.g., after fence wait).
    pub fn readback(
        &self,
        device: &ash::Device,
        timestamp_period: f32,
    ) -> Result<Vec<(String, f64)>, vk::Result> {
        let count = self.pool.allocated_count();
        if count == 0 {
            return Ok(Vec::new());
        }

        let results = self.pool.get_results_u64(device, 0, count)?;

        let mut output = Vec::with_capacity(self.intervals.len());
        for (name, interval) in &self.intervals {
            if interval.end_index == u32::MAX {
                continue;
            }
            let begin = results[interval.begin_index as usize];
            let end = results[interval.end_index as usize];
            let ticks = end.wrapping_sub(begin);
            let ns = ticks as f64 * timestamp_period as f64;
            output.push((name.clone(), ns));
        }

        Ok(output)
    }

    /// Destroy the underlying query pool.
    pub fn destroy(&mut self, device: &ash::Device) {
        self.pool.destroy(device);
    }
}
