use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use som_plus_clustering::{KMeansBuilder, KMeansInit};

fn bench_kmeans_fit_small(c: &mut Criterion) {
    let data = Array2::from_shape_fn((200, 4), |(i, j)| (i + j) as f64 * 0.01);
    c.bench_function("kmeans_fit_200x4_k8", |b| {
        b.iter(|| {
            let mut km = KMeansBuilder::new()
                .n_clusters(8)
                .method(KMeansInit::PlusPlus)
                .build();
            km.fit(&data.view()).unwrap();
        });
    });
}

criterion_group!(kmeans_benches, bench_kmeans_fit_small);
criterion_main!(kmeans_benches);
