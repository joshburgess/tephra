//! Benchmarks for pipeline state hashing.
//!
//! Measures the throughput of hashing `StaticPipelineState` with FxHasher
//! (the production hasher) vs the standard library's DefaultHasher.

use std::hash::{Hash, Hasher};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustc_hash::FxHasher;

use ignis::command::state::StaticPipelineState;

fn hash_with_fx(state: &StaticPipelineState) -> u64 {
    let mut hasher = FxHasher::default();
    state.hash(&mut hasher);
    hasher.finish()
}

fn hash_with_default(state: &StaticPipelineState) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    state.hash(&mut hasher);
    hasher.finish()
}

fn bench_state_hashing(c: &mut Criterion) {
    let state = StaticPipelineState::default();

    let mut group = c.benchmark_group("state_hashing");

    group.bench_function("fx_hasher", |b| {
        b.iter(|| hash_with_fx(black_box(&state)))
    });

    group.bench_function("default_hasher", |b| {
        b.iter(|| hash_with_default(black_box(&state)))
    });

    group.finish();
}

fn bench_state_clone(c: &mut Criterion) {
    let state = StaticPipelineState::default();

    c.bench_function("state_clone", |b| {
        b.iter(|| black_box(&state).clone())
    });
}

fn bench_state_equality(c: &mut Criterion) {
    let state_a = StaticPipelineState::default();
    let state_b = StaticPipelineState::default();

    c.bench_function("state_equality", |b| {
        b.iter(|| black_box(&state_a) == black_box(&state_b))
    });
}

criterion_group!(benches, bench_state_hashing, bench_state_clone, bench_state_equality);
criterion_main!(benches);
