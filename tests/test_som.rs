use som_plus_clustering::{SomBuilder, EvalMethod};
use ndarray::Array2;

fn synthetic_data() -> Array2<f64> {
    Array2::from_shape_fn((100, 4), |(i, j)| ((i + j) as f64).sin() * 0.5 + 0.5)
}

#[test]
fn full_fit_predict_evaluate() {
    let d = synthetic_data();
    let mut som = SomBuilder::new()
        .grid(5, 5)
        .dim(4)
        .learning_rate(0.5)
        .unwrap()
        .build();
    som.fit(&d.view(), 3, true, None).unwrap();
    let labels = som.predict(&d.view()).unwrap();
    assert_eq!(labels.len(), 100);
    assert!(labels.iter().all(|&l| l < 25));
    let scores = som.evaluate(&d.view(), &[EvalMethod::All]).unwrap();
    assert!(scores.contains_key(&EvalMethod::Silhouette));
    assert!(scores.contains_key(&EvalMethod::DaviesBouldin));
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn bmu_in_grid_bounds(m in 2usize..6, n in 2usize..6, dim in 2usize..5) {
        let data = Array2::from_shape_fn((20, dim), |(i, j)| (i + j) as f64 * 0.1);
        let mut som = SomBuilder::new()
            .grid(m, n)
            .dim(dim)
            .learning_rate(0.5)
            .unwrap()
            .build();
        som.fit(&data.view(), 1, false, None).unwrap();
        let labels = som.predict(&data.view()).unwrap();
        prop_assert!(labels.iter().all(|&l| l < m * n));
    }

    #[test]
    fn neuron_count_matches_grid(m in 2usize..6, n in 2usize..6) {
        let dim = 3usize;
        let data = Array2::from_shape_fn((20, dim), |(i, j)| (i + j) as f64 * 0.1);
        let mut som = SomBuilder::new()
            .grid(m, n)
            .dim(dim)
            .learning_rate(0.5)
            .unwrap()
            .build();
        som.fit(&data.view(), 1, false, None).unwrap();
        prop_assert_eq!(som.cluster_centers().nrows(), m * n);
    }
}
