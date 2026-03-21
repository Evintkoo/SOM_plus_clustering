use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use som_plus_clustering::SomBuilder;

fn bench_fit_cpu_small(c: &mut Criterion) {
    let data = Array2::from_shape_fn((100, 3), |(i, j)| (i + j) as f64 * 0.01);
    c.bench_function("som_fit_cpu_small_100x3_10x10", |b| {
        b.iter(|| {
            let mut som = SomBuilder::new()
                .grid(10, 10)
                .dim(3)
                .learning_rate(0.5)
                .unwrap()
                .build();
            som.fit(&data.view(), 2, false, None).unwrap();
        });
    });
}

fn bench_fit_cpu_large(c: &mut Criterion) {
    let data = Array2::from_shape_fn((10_000, 50), |(i, j)| (i + j) as f64 * 0.0001);
    c.bench_function("som_fit_cpu_large_10kx50_20x20", |b| {
        b.iter(|| {
            let mut som = SomBuilder::new()
                .grid(20, 20)
                .dim(50)
                .learning_rate(0.5)
                .unwrap()
                .build();
            som.fit(&data.view(), 1, false, Some(256)).unwrap();
        });
    });
}

criterion_group!(som_benches, bench_fit_cpu_small, bench_fit_cpu_large);
criterion_main!(som_benches);
