use ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Selects which distance metric to use.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceFunction {
    Euclidean,
    Cosine,
}

/// Euclidean distance between two 1-D vectors.
pub fn euclidean(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let diff = a - b;
    diff.dot(&diff).sqrt()
}

/// Squared Euclidean distance (cheaper, used for BMU search).
pub fn euclidean_sq(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let diff = a - b;
    diff.dot(&diff)
}

/// Cosine distance (1 - cosine_similarity) between two 1-D vectors.
pub fn cosine(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt() + 1e-12;
    let norm_b = b.dot(b).sqrt() + 1e-12;
    1.0 - dot / (norm_a * norm_b)
}

/// Batch Euclidean distance matrix: shape [n_samples, n_neurons].
/// Uses ||a-b||² = ||a||² + ||b||² - 2·a·bᵀ (BLAS-accelerated).
pub fn batch_euclidean(data: &ArrayView2<f64>, neurons: &ArrayView2<f64>) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;
    let n = data.nrows();
    let k = neurons.nrows();
    let data_sq = data.mapv(|x| x * x).sum_axis(ndarray::Axis(1)); // (n,)
    let neuron_sq = neurons.mapv(|x| x * x).sum_axis(ndarray::Axis(1)); // (k,)
    let mut cross = Array2::<f64>::zeros((n, k));
    general_mat_mul(1.0, data, &neurons.t(), 0.0, &mut cross);
    // d²[i,j] = data_sq[i] + neuron_sq[j] - 2*cross[i,j]
    let mut dist = cross;
    for i in 0..n {
        for j in 0..k {
            dist[[i, j]] = data_sq[i] + neuron_sq[j] - 2.0 * dist[[i, j]];
            if dist[[i, j]] < 0.0 { dist[[i, j]] = 0.0; } // numerical safety
        }
    }
    dist
}

/// Batch cosine distance matrix: shape [n_samples, n_neurons].
pub fn batch_cosine(data: &ArrayView2<f64>, neurons: &ArrayView2<f64>) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;
    let n = data.nrows();
    let k = neurons.nrows();
    let data_norms = data.mapv(|x| x * x)
        .sum_axis(ndarray::Axis(1))
        .mapv(|x| x.sqrt() + 1e-12); // (n,)
    let neuron_norms = neurons.mapv(|x| x * x)
        .sum_axis(ndarray::Axis(1))
        .mapv(|x| x.sqrt() + 1e-12); // (k,)
    let mut dots = Array2::<f64>::zeros((n, k));
    general_mat_mul(1.0, data, &neurons.t(), 0.0, &mut dots);
    for i in 0..n {
        for j in 0..k {
            dots[[i, j]] = 1.0 - dots[[i, j]] / (data_norms[i] * neuron_norms[j]);
        }
    }
    dots
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn euclidean_zero_distance() {
        let a = array![1.0_f64, 2.0, 3.0];
        assert!((euclidean(&a.view(), &a.view())).abs() < 1e-10);
    }

    #[test]
    fn euclidean_known_value() {
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        assert!((euclidean(&a.view(), &b.view()) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = array![1.0_f64, 0.0, 0.0];
        assert!(cosine(&a.view(), &a.view()).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = array![1.0_f64, 0.0];
        let b = array![0.0_f64, 1.0];
        assert!((cosine(&a.view(), &b.view()) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn batch_euclidean_shape() {
        let data = Array2::<f64>::zeros((5, 3));
        let neurons = Array2::<f64>::zeros((10, 3));
        let d = batch_euclidean(&data.view(), &neurons.view());
        assert_eq!(d.shape(), &[5, 10]);
    }
}
