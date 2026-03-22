#[allow(unused_imports)]
use crate::{
    core::{
        neighborhood,
        som::{InitMethod, Som, SomBuilder},
        distance::DistanceFunction,
    },
    SomError,
};
#[allow(unused_imports)]
use ndarray::{Array1, ArrayView2};
#[allow(unused_imports)]
use std::collections::VecDeque;

pub struct DenSomResult {
    pub labels:      Array1<i32>,
    pub density:     Array1<f64>,
    pub n_clusters:  usize,
    pub noise_ratio: f64,
}

pub struct DenSomBuilder {
    som_builder:  SomBuilder,
    smooth_sigma: f64,
}

pub struct DenSom {
    som:            Som,
    smooth_sigma:   f64,
    bmu_hits:       Array1<usize>,
    smooth_density: Array1<f64>,
    cluster_map:    Array1<i32>,
    n_clusters:     usize,
    fitted:         bool,
}

/// Otsu's method: finds the threshold that maximises between-class variance.
/// Returns the threshold value in the same units as `values`.
fn otsu(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 {
        return min;
    }
    const BINS: usize = 256;
    let mut hist = [0u32; BINS];
    for &v in values {
        let bin = (((v - min) / (max - min)) * (BINS - 1) as f64) as usize;
        hist[bin.min(BINS - 1)] += 1;
    }
    let total = values.len() as f64;
    let total_sum: f64 = (0..BINS).map(|b| b as f64 * hist[b] as f64).sum();
    let mut w0 = 0.0f64;
    let mut sum0 = 0.0f64;
    let mut best_var = f64::NEG_INFINITY;
    let mut best_t = 0usize;
    for t in 0..BINS {
        w0 += hist[t] as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }
        sum0 += t as f64 * hist[t] as f64;
        let mu0 = sum0 / w0;
        let mu1 = (total_sum - sum0) / w1;
        let var_between = w0 * w1 * (mu0 - mu1).powi(2);
        if var_between > best_var {
            best_var = var_between;
            best_t = t;
        }
    }
    min + (best_t as f64 / (BINS - 1) as f64) * (max - min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_two_class() {
        // Two well-separated classes: 10 values near 0.0, 10 near 1.0
        let mut vals: Vec<f64> = (0..10).map(|i| i as f64 * 0.01).collect();
        vals.extend((0..10).map(|i| 1.0 - i as f64 * 0.01));
        let tau = otsu(&vals);
        assert!(tau > 0.05, "threshold {tau} should be above the low class");
        assert!(tau < 0.95, "threshold {tau} should be below the high class");
    }
}
