use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use skillopt::gradient::{apply_patch, merge_patches, rank_and_select};
use skillopt::scheduler::{compute_lr, LrSchedule};
use skillopt::scoring::{exact_match, normalize_answer, skill_hash};
use skillopt::types::{Edit, EditOp};

fn mk(op: EditOp, anchor: &str, content: &str, u: f32, s: u32) -> Edit {
    Edit {
        op,
        anchor: anchor.into(),
        content: content.into(),
        rationale: String::new(),
        utility: u,
        source_type: "failure".into(),
        support_count: s,
    }
}

fn bench_lr(c: &mut Criterion) {
    let mut g = c.benchmark_group("compute_lr");
    for &n in &[10u32, 100, 1000] {
        g.bench_with_input(BenchmarkId::new("cosine", n), &n, |b, &n| {
            b.iter(|| {
                let mut sum = 0u64;
                for s in 0..n {
                    sum += compute_lr(black_box(&LrSchedule::Cosine), black_box(s), black_box(n), black_box(8)) as u64;
                }
                sum
            })
        });
    }
    g.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut g = c.benchmark_group("merge_patches");
    for &n in &[16usize, 64, 256] {
        let fail: Vec<Edit> = (0..n).map(|i| mk(EditOp::Add, &format!("a{}", i % 8), "x", 0.5, 1)).collect();
        let succ: Vec<Edit> = (0..n).map(|i| mk(EditOp::Add, &format!("a{}", i % 8), "y", 0.6, 1)).collect();
        g.throughput(Throughput::Elements((n * 2) as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| merge_patches(black_box(fail.clone()), black_box(succ.clone())))
        });
    }
    g.finish();
}

fn bench_select(c: &mut Criterion) {
    let mut g = c.benchmark_group("rank_and_select");
    for &n in &[32usize, 256, 1024] {
        let edits: Vec<Edit> = (0..n).map(|i| mk(EditOp::Add, "a", &format!("e{}", i), (i as f32 / n as f32), (i as u32 % 5) + 1)).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| rank_and_select(black_box(edits.clone()), 8))
        });
    }
    g.finish();
}

fn bench_apply(c: &mut Criterion) {
    let mut g = c.benchmark_group("apply_patch");
    let skill = (0..200).map(|i| format!("rule {}", i)).collect::<Vec<_>>().join("\n");
    for &n in &[1usize, 4, 8, 16] {
        let patch: Vec<Edit> = (0..n).map(|i| mk(EditOp::Add, &format!("rule {}", i * 7), "extra", 0.5, 1)).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| apply_patch(black_box(&skill), black_box(&patch)))
        });
    }
    g.finish();
}

fn bench_scoring(c: &mut Criterion) {
    let golds = vec!["Alexander Graham Bell".to_string(), "Bell".into()];
    c.bench_function("normalize_answer", |b| b.iter(|| normalize_answer(black_box("The Alexander Graham Bell."))));
    c.bench_function("exact_match", |b| b.iter(|| exact_match(black_box("alexander graham bell"), black_box(&golds))));
    let big = "lorem ipsum dolor sit amet ".repeat(400);
    c.bench_function("skill_hash_10kb", |b| b.iter(|| skill_hash(black_box(&big))));
}

criterion_group!(benches, bench_lr, bench_merge, bench_select, bench_apply, bench_scoring);
criterion_main!(benches);
