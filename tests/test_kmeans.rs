use som_plus_clustering::{KMeansBuilder, KMeansInit, SomError};
use ndarray::Array2;

fn blobs() -> Array2<f64> {
    Array2::from_shape_fn((40, 2), |(i, j)| {
        if i < 20 {
            i as f64 * 0.1 + j as f64 * 0.05
        } else {
            50.0 + i as f64 * 0.1
        }
    })
}

#[test]
fn kmeans_fit_predict_labels_valid() {
    let mut km = KMeansBuilder::new()
        .n_clusters(2)
        .method(KMeansInit::PlusPlus)
        .build();
    km.fit(&blobs().view()).unwrap();
    let labels = km.predict(&blobs().view()).unwrap();
    assert_eq!(labels.len(), 40);
    assert!(labels.iter().all(|&l| l < 2));
}

#[test]
fn kmeans_inertia_finite() {
    let mut km = KMeansBuilder::new()
        .n_clusters(2)
        .method(KMeansInit::Random)
        .build();
    km.fit(&blobs().view()).unwrap();
    let inertia = km.inertia().unwrap();
    assert!(inertia.is_finite() && inertia >= 0.0);
}

#[test]
fn kmeans_already_fitted() {
    let mut km = KMeansBuilder::new()
        .n_clusters(2)
        .method(KMeansInit::Random)
        .build();
    km.fit(&blobs().view()).unwrap();
    assert!(matches!(km.fit(&blobs().view()), Err(SomError::AlreadyFitted)));
}
