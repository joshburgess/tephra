//! Benchmarks for descriptor binding table operations and cache lookups.
//!
//! Measures the throughput of:
//! - Setting bindings in the binding table
//! - Dirty flag checking
//! - Binding state hashing (for descriptor set cache keys)

use std::hash::{Hash, Hasher};

use ash::vk::{self, Handle};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rustc_hash::FxHasher;

use tephra::descriptors::binding_table::{BindingTable, DescriptorSetBindings};

fn hash_bindings(bindings: &DescriptorSetBindings) -> u64 {
    let mut hasher = FxHasher::default();
    bindings.hash(&mut hasher);
    hasher.finish()
}

fn bench_binding_set_uniform(c: &mut Criterion) {
    let buffer = vk::Buffer::from_raw(0x1234);

    c.bench_function("binding_set_uniform_buffer", |b| {
        let mut table = BindingTable::new();
        b.iter(|| {
            table.set_uniform_buffer(
                black_box(0),
                black_box(0),
                black_box(buffer),
                black_box(0),
                black_box(256),
            );
        })
    });
}

fn bench_binding_set_texture(c: &mut Criterion) {
    let view = vk::ImageView::from_raw(0x5678);
    let sampler = vk::Sampler::from_raw(0x9ABC);

    c.bench_function("binding_set_texture", |b| {
        let mut table = BindingTable::new();
        b.iter(|| {
            table.set_texture(
                black_box(0),
                black_box(1),
                black_box(view),
                black_box(sampler),
            );
        })
    });
}

fn bench_dirty_check(c: &mut Criterion) {
    let mut table = BindingTable::new();
    let buffer = vk::Buffer::from_raw(0x1234);
    table.set_uniform_buffer(0, 0, buffer, 0, 256);

    c.bench_function("dirty_check", |b| {
        b.iter(|| {
            let dirty = black_box(&table).dirty_sets();
            black_box(dirty);
        })
    });
}

fn bench_binding_hash(c: &mut Criterion) {
    let mut table = BindingTable::new();
    let buffer = vk::Buffer::from_raw(0x1234);
    let view = vk::ImageView::from_raw(0x5678);
    let sampler = vk::Sampler::from_raw(0x9ABC);

    // Fill set 0 with several bindings
    table.set_uniform_buffer(0, 0, buffer, 0, 256);
    table.set_storage_buffer(0, 1, buffer, 256, 1024);
    table.set_texture(0, 2, view, sampler);
    table.set_texture(0, 3, view, sampler);

    let bindings = table.set(0).clone();

    let mut group = c.benchmark_group("binding_hash");

    group.bench_function("fx_hasher_4_bindings", |b| {
        b.iter(|| hash_bindings(black_box(&bindings)))
    });

    // Also measure with empty bindings
    let empty = DescriptorSetBindings::default();
    group.bench_function("fx_hasher_empty", |b| {
        b.iter(|| hash_bindings(black_box(&empty)))
    });

    group.finish();
}

fn bench_binding_table_throughput(c: &mut Criterion) {
    let buffer = vk::Buffer::from_raw(0x1234);
    let view = vk::ImageView::from_raw(0x5678);
    let sampler = vk::Sampler::from_raw(0x9ABC);

    c.bench_function("binding_table_fill_4_sets", |b| {
        let mut table = BindingTable::new();
        b.iter(|| {
            // Simulate filling 4 sets with 2 bindings each
            for set in 0..4u32 {
                table.set_uniform_buffer(set, 0, buffer, 0, 256);
                table.set_texture(set, 1, view, sampler);
            }
            table.clear_all_dirty();
        })
    });
}

criterion_group!(
    benches,
    bench_binding_set_uniform,
    bench_binding_set_texture,
    bench_dirty_check,
    bench_binding_hash,
    bench_binding_table_throughput
);
criterion_main!(benches);
