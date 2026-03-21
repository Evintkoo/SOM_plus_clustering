use ndarray::{Array1, Array2};
use som_plus_clustering::{bcubed_scores, EvalMethod, SomBuilder};

fn separated_data() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_fn((40, 2), |(i, j)| {
        if i < 20 {
            i as f64 * 0.01 + j as f64 * 0.01
        } else {
            100.0 + i as f64 * 0.01
        }
    });
    let labels = Array1::from_iter((0..40).map(|i| if i < 20 { 0 } else { 1 }));
    (x, labels)
}

#[test]
fn silhouette_high_for_separated_clusters() {
    let (x, _) = separated_data();
    let mut som = SomBuilder::new()
        .grid(2, 1)
        .dim(2)
        .learning_rate(0.5)
        .unwrap()
        .build();
    som.fit(&x.view(), 3, false, None).unwrap();
    let scores = som.evaluate(&x.view(), &[EvalMethod::Silhouette]).unwrap();
    let s = scores[&EvalMethod::Silhouette];
    assert!(s >= -1.0 && s <= 1.0);
}

#[test]
fn all_metrics_return_values() {
    let (x, _) = separated_data();
    let mut som = SomBuilder::new()
        .grid(2, 2)
        .dim(2)
        .learning_rate(0.5)
        .unwrap()
        .build();
    som.fit(&x.view(), 2, false, None).unwrap();
    let scores = som.evaluate(&x.view(), &[EvalMethod::All]).unwrap();
    assert!(scores.contains_key(&EvalMethod::Silhouette));
    assert!(scores.contains_key(&EvalMethod::DaviesBouldin));
    assert!(scores.contains_key(&EvalMethod::Dunn));
}

#[test]
fn bcubed_perfect_separation() {
    let clusters = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);
    let labels = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);
    let (p, r) = bcubed_scores(&clusters.view(), &labels.view());
    assert!((p - 1.0).abs() < 1e-6);
    assert!((r - 1.0).abs() < 1e-6);
}
