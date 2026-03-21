use som_plus_clustering::{Som, SomBuilder};
use ndarray::Array2;

#[test]
fn save_load_predict_identical() {
    let d = Array2::from_shape_fn((50, 3), |(i, j)| i as f64 + j as f64 * 0.1);
    let mut som = SomBuilder::new()
        .grid(3, 3)
        .dim(3)
        .learning_rate(0.5)
        .unwrap()
        .build();
    som.fit(&d.view(), 2, false, None).unwrap();
    let labels_before = som.predict(&d.view()).unwrap();
    som.save("/tmp/test_som.bin").unwrap();
    let som2 = Som::load("/tmp/test_som.bin").unwrap();
    let labels_after = som2.predict(&d.view()).unwrap();
    assert_eq!(labels_before, labels_after);
}
