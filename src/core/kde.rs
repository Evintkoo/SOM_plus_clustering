use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::f64::consts::PI;

/// Silverman-like bandwidth estimator: (max - min) / (1 + log2(n))
pub fn bandwidth_estimator(data: &ArrayView1<f64>) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }
    let max = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min = data.fold(f64::INFINITY, |a, &b| a.min(b));
    let bw = (max - min) / (1.0 + (n as f64).log2());
    bw.max(1e-10)
}

fn gaussian_kernel(x: &ArrayView1<f64>, xi: &ArrayView1<f64>, bandwidth: f64) -> f64 {
    let d = x.len() as f64;
    let norm = 1.0 / ((2.0 * PI).sqrt().powf(d) * bandwidth.powf(d));
    let diff = x - xi;
    let exponent = -0.5 * diff.dot(&diff) / (bandwidth * bandwidth);
    norm * exponent.exp()
}

/// Multi-dimensional KDE evaluated at `points`, trained on `data`.
pub fn kde_multidimensional(
    data: &ArrayView2<f64>,
    points: &ArrayView2<f64>,
    bandwidth: f64,
) -> Array1<f64> {
    let n = data.nrows() as f64;
    let m = points.nrows();
    let mut vals = Array1::<f64>::zeros(m);
    for i in 0..m {
        let pt = points.row(i);
        let kernel_sum: f64 = (0..data.nrows())
            .map(|j| gaussian_kernel(&pt, &data.row(j), bandwidth))
            .sum();
        vals[i] = kernel_sum / n;
    }
    vals
}

/// Find local maxima in `kde_values` and return the corresponding `points` rows.
pub fn find_local_maxima(kde_values: &[f64], points: &ArrayView2<f64>) -> Array2<f64> {
    let mut maxima = Vec::new();
    let n = kde_values.len();
    for i in 1..n.saturating_sub(1) {
        if kde_values[i - 1] < kde_values[i] && kde_values[i] > kde_values[i + 1] {
            maxima.push(i);
        }
    }
    let dim = points.ncols();
    let mut out = Array2::<f64>::zeros((maxima.len(), dim));
    for (ci, &idx) in maxima.iter().enumerate() {
        out.row_mut(ci).assign(&points.row(idx));
    }
    out
}

/// KDE-based initialization: find local maxima of KDE, then farthest-first selection.
pub fn initiate_kde(
    data: &ArrayView2<f64>,
    n_neurons: usize,
    bandwidth: Option<f64>,
) -> Result<Array2<f64>, crate::SomError> {
    let bw = bandwidth.unwrap_or_else(|| bandwidth_estimator(&data.column(0)));
    let kde_vals = kde_multidimensional(data, data, bw);
    let kde_vec: Vec<f64> = kde_vals.iter().copied().collect();
    let local_max = find_local_maxima(&kde_vec, data);
    let max_n = local_max.nrows();
    if max_n < n_neurons {
        return Err(crate::SomError::KdeInsufficientMaxima {
            found: max_n,
            needed: n_neurons,
        });
    }
    // Farthest-first selection among local maxima (O(max_n × n_neurons))
    let mut selected = vec![0usize];
    let mut min_dist: Vec<f64> = (0..max_n)
        .map(|j| {
            let d = &local_max.row(0) - &local_max.row(j);
            d.dot(&d)
        })
        .collect();

    for _ in 1..n_neurons {
        let next = (0..max_n)
            .filter(|i| !selected.contains(i))
            .max_by(|&a, &b| {
                min_dist[a]
                    .partial_cmp(&min_dist[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        selected.push(next);
        for j in 0..max_n {
            let d = &local_max.row(next) - &local_max.row(j);
            let dist = d.dot(&d);
            if dist < min_dist[j] {
                min_dist[j] = dist;
            }
        }
    }
    let dim = local_max.ncols();
    let mut out = Array2::<f64>::zeros((n_neurons, dim));
    for (ci, &si) in selected.iter().enumerate() {
        out.row_mut(ci).assign(&local_max.row(si));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn bandwidth_increases_with_range() {
        let a = ndarray::array![0.0_f64, 1.0, 2.0, 3.0];
        let b = ndarray::array![0.0_f64, 10.0, 20.0, 30.0];
        assert!(bandwidth_estimator(&a.view()) < bandwidth_estimator(&b.view()));
    }

    #[test]
    fn kde_values_positive() {
        let data = Array2::from_shape_fn((20, 2), |(i,j)| i as f64 + j as f64 * 0.1);
        let bw = bandwidth_estimator(&data.column(0));
        let vals = kde_multidimensional(&data.view(), &data.view(), bw);
        assert!(vals.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn local_maxima_detected() {
        let vals: Vec<f64> = (0..11).map(|i| {
            if i == 5 { 1.0 } else { 0.1 }
        }).collect();
        let pts = Array2::from_shape_fn((11, 1), |(i,_)| i as f64);
        let maxima = find_local_maxima(&vals, &pts.view());
        assert_eq!(maxima.nrows(), 1);
    }
}
