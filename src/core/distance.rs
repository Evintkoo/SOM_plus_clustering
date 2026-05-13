use ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use crate::core::optimized_math::fast_inv_sqrt;

/// Selects which distance metric to use.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceFunction {
    Euclidean,
    Cosine,
    Manhattan,
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
/// Uses fast inverse sqrt for normalization.
pub fn cosine(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let dot = a.dot(b);
    let norm_a_sq = a.dot(a);
    let norm_b_sq = b.dot(b);
    
    // Use fast inverse sqrt instead of separate sqrt calls
    let inv_norm_a = fast_inv_sqrt(norm_a_sq).max(1e-12);
    let inv_norm_b = fast_inv_sqrt(norm_b_sq).max(1e-12);
    
    (1.0 - dot * inv_norm_a * inv_norm_b).clamp(0.0, 2.0)
}

/// Manhattan distance between two 1-D vectors (L1 norm).
pub fn manhattan_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let diff = a - b;
    diff.mapv(|x| x.abs()).sum()
}

/// Batch Euclidean distance matrix with norm caching.
/// Uses ||a-b||² = ||a||² + ||b||² - 2·a·bᵀ (BLAS-accelerated).
pub fn batch_euclidean(data: &ArrayView2<f64>, neurons: &ArrayView2<f64>) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;
    let n = data.nrows();
    let k = neurons.nrows();
    
    // Cache norms once per epoch (optimization #9)
    let data_sq = data.mapv(|x| x * x).sum_axis(ndarray::Axis(1)); // (n,)
    let neuron_sq = neurons.mapv(|x| x * x).sum_axis(ndarray::Axis(1)); // (k,)
    
    let mut cross = Array2::<f64>::zeros((n, k));
    general_mat_mul(1.0, data, &neurons.t(), 0.0, &mut cross);
    
    // d[i,j]² = ||a||² + ||b||² - 2·a·bᵀ
    let mut dist = cross;
    for i in 0..n {
        for j in 0..k {
            dist[[i, j]] = (data_sq[i] + neuron_sq[j] - 2.0 * dist[[i, j]])
                .max(0.0)
                .sqrt();
        }
    }
    dist
}

/// Batch cosine distance matrix with fast inverse sqrt.
pub fn batch_cosine(data: &ArrayView2<f64>, neurons: &ArrayView2<f64>) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;
    let n = data.nrows();
    let k = neurons.nrows();
    
    // Cache norm squares (optimization #9)
    let data_norms_sq = data
        .mapv(|x| x * x)
        .sum_axis(ndarray::Axis(1)); // (n,)
    let neuron_norms_sq = neurons
        .mapv(|x| x * x)
        .sum_axis(ndarray::Axis(1)); // (k,)
    
    let mut dots = Array2::<f64>::zeros((n, k));
    general_mat_mul(1.0, data, &neurons.t(), 0.0, &mut dots);
    
    // Use fast inverse sqrt (optimization #1)
    for i in 0..n {
        for j in 0..k {
            let inv_norm_a = fast_inv_sqrt(data_norms_sq[i]).max(1e-12);
            let inv_norm_b = fast_inv_sqrt(neuron_norms_sq[j]).max(1e-12);
            dots[[i, j]] = (1.0 - dots[[i, j]] * inv_norm_a * inv_norm_b).clamp(0.0, 2.0);
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

    #[test]
    fn batch_euclidean_values_match_scalar() {
        let data = array![[0.0_f64, 0.0], [1.0_f64, 0.0]];
        let neurons = array![[3.0_f64, 4.0]];
        let d = batch_euclidean(&data.view(), &neurons.view());
        // data[0] -> neuron[0]: euclidean([0,0],[3,4]) = 5.0
        assert!((d[[0, 0]] - 5.0).abs() < 1e-10);
        // data[1] -> neuron[0]: euclidean([1,0],[3,4]) = sqrt(4+16) = sqrt(20)
        assert!((d[[1, 0]] - 20.0_f64.sqrt()).abs() < 1e-10);
    }
}
