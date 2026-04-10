//! Benchmarks for pipeline cache key hashing and lookup.
//!
//! Measures FxHashMap lookup speed for pipeline cache hits,
//! which is the hot path on every draw call.

use std::hash::{Hash, Hasher};

use ash::vk::{self, Handle};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rustc_hash::{FxHashMap, FxHasher};

use ignis::command::state::StaticPipelineState;
use ignis::pipeline::pipeline::VertexInputLayout;

/// Simulate the pipeline key hash computation used in PipelineCompiler.
fn compute_pipeline_key_hash(
    program_hash: u64,
    state: &StaticPipelineState,
    render_pass_hash: u64,
    subpass: u32,
    vertex_layout_hash: u64,
) -> u64 {
    let mut hasher = FxHasher::default();
    program_hash.hash(&mut hasher);
    state.hash(&mut hasher);
    render_pass_hash.hash(&mut hasher);
    subpass.hash(&mut hasher);
    vertex_layout_hash.hash(&mut hasher);
    hasher.finish()
}

fn bench_pipeline_key_hash(c: &mut Criterion) {
    let state = StaticPipelineState::default();
    let program_hash = 0xDEADBEEF_u64;
    let rp_hash = 0xCAFEBABE_u64;
    let vertex_hash = VertexInputLayout::default().compute_hash();

    c.bench_function("pipeline_key_hash", |b| {
        b.iter(|| {
            compute_pipeline_key_hash(
                black_box(program_hash),
                black_box(&state),
                black_box(rp_hash),
                black_box(0),
                black_box(vertex_hash),
            )
        })
    });
}

fn bench_pipeline_cache_hit(c: &mut Criterion) {
    let state = StaticPipelineState::default();
    let program_hash = 0xDEADBEEF_u64;
    let rp_hash = 0xCAFEBABE_u64;
    let vertex_hash = VertexInputLayout::default().compute_hash();

    // Pre-populate a cache with 100 entries (realistic for a game scene)
    let mut cache: FxHashMap<u64, vk::Pipeline> = FxHashMap::default();
    let target_key = compute_pipeline_key_hash(program_hash, &state, rp_hash, 0, vertex_hash);
    cache.insert(target_key, vk::Pipeline::from_raw(0x1));

    // Add 99 more entries with different states
    for i in 1..100u64 {
        let mut varied_state = state.clone();
        varied_state.topology = match i % 3 {
            0 => vk::PrimitiveTopology::TRIANGLE_LIST,
            1 => vk::PrimitiveTopology::LINE_LIST,
            _ => vk::PrimitiveTopology::POINT_LIST,
        };
        let key = compute_pipeline_key_hash(i, &varied_state, rp_hash, 0, vertex_hash);
        cache.insert(key, vk::Pipeline::from_raw(i + 1));
    }

    let mut group = c.benchmark_group("pipeline_cache_lookup");

    group.bench_function("hit_100_entries", |b| {
        b.iter(|| {
            let key = compute_pipeline_key_hash(
                black_box(program_hash),
                black_box(&state),
                black_box(rp_hash),
                black_box(0),
                black_box(vertex_hash),
            );
            black_box(cache.get(&key))
        })
    });

    // Also benchmark a miss (to measure the hash + lookup path without the hit shortcut)
    group.bench_function("miss_100_entries", |b| {
        b.iter(|| {
            let key = compute_pipeline_key_hash(
                black_box(0xFFFFFFFF_u64),
                black_box(&state),
                black_box(rp_hash),
                black_box(0),
                black_box(vertex_hash),
            );
            black_box(cache.get(&key))
        })
    });

    group.finish();
}

fn bench_pipeline_cache_scaling(c: &mut Criterion) {
    let state = StaticPipelineState::default();
    let rp_hash = 0xCAFEBABE_u64;
    let vertex_hash = VertexInputLayout::default().compute_hash();

    let mut group = c.benchmark_group("pipeline_cache_scaling");

    for &size in &[10u64, 100, 1000] {
        let mut cache: FxHashMap<u64, vk::Pipeline> = FxHashMap::default();
        for i in 0..size {
            let key = compute_pipeline_key_hash(i, &state, rp_hash, 0, vertex_hash);
            cache.insert(key, vk::Pipeline::from_raw(i + 1));
        }

        // Lookup the middle entry
        let target = size / 2;
        let target_key = compute_pipeline_key_hash(target, &state, rp_hash, 0, vertex_hash);

        group.bench_function(format!("hit_{size}_entries"), |b| {
            b.iter(|| black_box(cache.get(black_box(&target_key))))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pipeline_key_hash,
    bench_pipeline_cache_hit,
    bench_pipeline_cache_scaling
);
criterion_main!(benches);
