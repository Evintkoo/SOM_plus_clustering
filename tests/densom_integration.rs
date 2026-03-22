use ndarray::Array2;
use som_plus_clustering::{DenSom, DenSomBuilder, InitMethod, SomBuilder};

/// Generate two concentric rings (inner radius 0.5, outer radius 1.0).
fn make_circles(n_per_ring: usize) -> Array2<f64> {
    let mut pts = Vec::with_capacity(n_per_ring * 4);
    for i in 0..n_per_ring {
        let t = 2.0 * std::f64::consts::PI * i as f64 / n_per_ring as f64;
        pts.push(t.cos() * 0.5);
        pts.push(t.sin() * 0.5);
    }
    for i in 0..n_per_ring {
        let t = 2.0 * std::f64::consts::PI * i as f64 / n_per_ring as f64;
        pts.push(t.cos() * 1.0);
        pts.push(t.sin() * 1.0);
    }
    Array2::from_shape_vec((n_per_ring * 2, 2), pts).unwrap()
}

#[test]
fn densom_circles() {
    // 150 clean ring points, no noise in data — noise_ratio should be low,
    // and SOM should find exactly 2 clusters (inner ring + outer ring).
    // NaiveSharding + shuffle=false gives deterministic results.
    let data = make_circles(75);
    let mut densom = DenSomBuilder::new()
        .grid(10, 10)
        .dim(2)
        .init_method(InitMethod::NaiveSharding)
        .build();
    let result = densom.fit_predict(&data.view(), 20, false, None).unwrap();
    assert!(
        result.noise_ratio < 0.15,
        "clean circles data should have noise_ratio < 0.15, got {}",
        result.noise_ratio
    );
    assert_eq!(
        result.n_clusters, 2,
        "circles should produce exactly 2 clusters (inner + outer ring), got {}",
        result.n_clusters
    );
}

#[test]
fn densom_from_som_matches_standalone() {
    // NaiveSharding is deterministic — same data + same config → identical weights.
    // This enables the spec-required identical labels assertion.
    let mut data_vec: Vec<f64> = Vec::new();
    for i in 0..50 { data_vec.push(i as f64 * 0.01); data_vec.push(0.0); }
    for i in 0..50 { data_vec.push(8.0 + i as f64 * 0.01); data_vec.push(0.0); }
    let data = Array2::from_shape_vec((100, 2), data_vec).unwrap();

    // Standalone path
    let mut densom_a = DenSomBuilder::new()
        .grid(6, 6)
        .dim(2)
        .init_method(InitMethod::NaiveSharding)
        .build();
    let result_a = densom_a.fit_predict(&data.view(), 5, false, None).unwrap();

    // from_som path — NaiveSharding + same data → identical weights to densom_a's Som
    let mut som = SomBuilder::new()
        .grid(6, 6)
        .dim(2)
        .init_method(InitMethod::NaiveSharding)
        .build();
    som.fit(&data.view(), 5, false, None).unwrap();
    let mut densom_b = DenSom::from_som(som);
    densom_b.refit_density(&data.view()).unwrap();
    let result_b = densom_b.predict(&data.view()).unwrap();

    // Identical weights → identical BMU assignments → identical labels
    assert_eq!(
        result_a.labels, result_b.labels,
        "standalone and from_som paths must produce identical labels with NaiveSharding + shuffle=false"
    );
}
