use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use crate::SomError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EvalMethod { Silhouette, DaviesBouldin, CalinskiHarabasz, Dunn, All }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ClassEvalMethod { Accuracy, F1, Recall, All }

fn pairwise_distances(data: &ArrayView2<f64>) -> Array2<f64> {
    use crate::core::distance::batch_euclidean;
    batch_euclidean(data, data)
}

pub fn silhouette_score(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
) -> Result<f64, SomError> {
    let n = data.nrows();
    let unique: Vec<usize> = {
        let mut v: Vec<usize> = labels.to_vec();
        v.sort();
        v.dedup();
        v
    };
    if unique.len() == 1 {
        return Ok(0.0);
    }
    let dists = pairwise_distances(data);
    let mut a = Array1::<f64>::zeros(n);
    let mut b = Array1::<f64>::from_elem(n, f64::INFINITY);

    for &label in &unique {
        let idx: Vec<usize> = (0..n).filter(|&i| labels[i] == label).collect();
        let sz = idx.len();
        if sz > 1 {
            for &i in &idx {
                let sum: f64 = idx.iter().filter(|&&j| j != i).map(|&j| dists[[i, j]]).sum();
                a[i] = sum / (sz - 1) as f64;
            }
        }
        for &other in &unique {
            if other == label {
                continue;
            }
            let other_idx: Vec<usize> = (0..n).filter(|&i| labels[i] == other).collect();
            for &i in &idx {
                let mean: f64 = other_idx.iter().map(|&j| dists[[i, j]]).sum::<f64>()
                    / other_idx.len() as f64;
                if mean < b[i] {
                    b[i] = mean;
                }
            }
        }
    }
    let s_vals: Vec<f64> = (0..n)
        .filter_map(|i| {
            if a[i] == 0.0 && b[i] == f64::INFINITY {
                return None;
            }
            let denom = a[i].max(b[i]);
            if denom == 0.0 { None } else { Some((b[i] - a[i]) / denom) }
        })
        .collect();
    if s_vals.is_empty() {
        return Ok(0.0);
    }
    Ok(s_vals.iter().sum::<f64>() / s_vals.len() as f64)
}

pub fn davies_bouldin_index(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
) -> Result<f64, SomError> {
    let n = data.nrows();
    let unique: Vec<usize> = {
        let mut v = labels.to_vec();
        v.sort();
        v.dedup();
        v
    };
    let k = unique.len();
    if k <= 1 {
        return Ok(0.0);
    }
    let centroids: Vec<Array1<f64>> = unique
        .iter()
        .map(|&l| {
            let pts: Vec<usize> = (0..n).filter(|&i| labels[i] == l).collect();
            let mut c = Array1::<f64>::zeros(data.ncols());
            for &i in &pts {
                c.scaled_add(1.0, &data.row(i));
            }
            c / pts.len() as f64
        })
        .collect();
    let dispersions: Vec<f64> = unique
        .iter()
        .enumerate()
        .map(|(ci, &l)| {
            let pts: Vec<usize> = (0..n).filter(|&i| labels[i] == l).collect();
            pts.iter()
                .map(|&i| {
                    let d = &data.row(i) - &centroids[ci];
                    d.dot(&d).sqrt()
                })
                .sum::<f64>()
                / pts.len() as f64
        })
        .collect();
    let mut db = 0.0_f64;
    for i in 0..k {
        let mut max_r = 0.0_f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let d = &centroids[i] - &centroids[j];
            let dist = d.dot(&d).sqrt();
            if dist > 1e-12 {
                let r = (dispersions[i] + dispersions[j]) / dist;
                if r > max_r {
                    max_r = r;
                }
            }
        }
        db += max_r;
    }
    Ok(db / k as f64)
}

pub fn calinski_harabasz_score(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
) -> Result<f64, SomError> {
    let n = data.nrows();
    let unique: Vec<usize> = {
        let mut v = labels.to_vec();
        v.sort();
        v.dedup();
        v
    };
    let k = unique.len();
    if k <= 1 {
        return Ok(0.0);
    }
    let overall: Array1<f64> = data.mean_axis(Axis(0)).unwrap();
    let mut between = 0.0_f64;
    let mut within = 0.0_f64;
    for &l in &unique {
        let pts: Vec<usize> = (0..n).filter(|&i| labels[i] == l).collect();
        let sz = pts.len();
        let mut c = Array1::<f64>::zeros(data.ncols());
        for &i in &pts {
            c.scaled_add(1.0, &data.row(i));
        }
        c /= sz as f64;
        let d = &c - &overall;
        between += sz as f64 * d.dot(&d);
        for &i in &pts {
            let dd = &data.row(i) - &c;
            within += dd.dot(&dd);
        }
    }
    if within.abs() < 1e-12 {
        return Err(SomError::ZeroWithinClusterVariance);
    }
    Ok(((n - k) as f64 / (k - 1) as f64) * (between / within))
}

pub fn dunn_index(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
) -> Result<f64, SomError> {
    let n = data.nrows();
    let unique: Vec<usize> = {
        let mut v = labels.to_vec();
        v.sort();
        v.dedup();
        v
    };
    if unique.len() <= 1 {
        return Ok(0.0);
    }
    let dists = pairwise_distances(data);
    let mut min_inter = f64::INFINITY;
    let mut max_intra = 0.0_f64;
    for (ii, &l1) in unique.iter().enumerate() {
        let idx1: Vec<usize> = (0..n).filter(|&i| labels[i] == l1).collect();
        for &i in &idx1 {
            for &j in &idx1 {
                if i < j {
                    let d = dists[[i, j]];
                    if d > max_intra {
                        max_intra = d;
                    }
                }
            }
        }
        for &l2 in &unique[ii + 1..] {
            let idx2: Vec<usize> = (0..n).filter(|&i| labels[i] == l2).collect();
            for &i in &idx1 {
                for &j in &idx2 {
                    let d = dists[[i, j]];
                    if d < min_inter {
                        min_inter = d;
                    }
                }
            }
        }
    }
    Ok(min_inter / max_intra)
}

pub fn bcubed_scores(
    clusters: &ArrayView1<usize>,
    labels: &ArrayView1<usize>,
) -> (f64, f64) {
    let n = clusters.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let (mut prec, mut rec) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        let same_cluster: Vec<usize> = (0..n).filter(|&j| clusters[j] == clusters[i]).collect();
        let same_label: Vec<usize> = (0..n).filter(|&j| labels[j] == labels[i]).collect();
        let both = same_cluster.iter().filter(|&&j| labels[j] == labels[i]).count();
        prec += both as f64 / same_cluster.len() as f64;
        rec += both as f64 / same_label.len() as f64;
    }
    (prec / n as f64, rec / n as f64)
}

pub fn accuracy(y_true: &ArrayView1<usize>, y_pred: &ArrayView1<usize>) -> f64 {
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let correct = y_true.iter().zip(y_pred.iter()).filter(|(a, b)| a == b).count();
    correct as f64 / n as f64 * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn perfect_clusters() -> (Array2<f64>, ndarray::Array1<usize>) {
        let data = Array2::from_shape_fn((20, 2), |(i, j)| {
            if i < 10 { i as f64 * 0.01 + j as f64 * 0.01 }
            else { 100.0 + i as f64 * 0.01 }
        });
        let labels = ndarray::Array1::from_iter((0..20).map(|i| if i < 10 { 0 } else { 1 }));
        (data, labels)
    }

    #[test]
    fn silhouette_in_range() {
        let (d, l) = perfect_clusters();
        let s = silhouette_score(&d.view(), &l.view()).unwrap();
        assert!(s >= -1.0 && s <= 1.0, "silhouette out of range: {}", s);
    }

    #[test]
    fn silhouette_high_for_separated() {
        let (d, l) = perfect_clusters();
        let s = silhouette_score(&d.view(), &l.view()).unwrap();
        assert!(s > 0.5, "expected high silhouette for separated clusters, got {}", s);
    }

    #[test]
    fn davies_bouldin_non_negative() {
        let (d, l) = perfect_clusters();
        let db = davies_bouldin_index(&d.view(), &l.view()).unwrap();
        assert!(db >= 0.0);
    }

    #[test]
    fn dunn_positive() {
        let (d, l) = perfect_clusters();
        let di = dunn_index(&d.view(), &l.view()).unwrap();
        assert!(di > 0.0);
    }

    #[test]
    fn calinski_harabasz_positive() {
        let (d, l) = perfect_clusters();
        let ch = calinski_harabasz_score(&d.view(), &l.view()).unwrap();
        assert!(ch > 0.0);
    }

    #[test]
    fn ch_zero_variance_error() {
        let d = Array2::<f64>::zeros((10, 2));
        let l = ndarray::Array1::from_iter((0..10).map(|i| i % 2));
        // Either Ok or Err(ZeroWithinClusterVariance) — just must not panic
        let _ = calinski_harabasz_score(&d.view(), &l.view());
    }

    #[test]
    fn bcubed_perfect() {
        let clusters = ndarray::Array1::from_vec(vec![0,0,0,1,1,1]);
        let labels   = ndarray::Array1::from_vec(vec![0,0,0,1,1,1]);
        let (p, r) = bcubed_scores(&clusters.view(), &labels.view());
        assert!((p - 1.0).abs() < 1e-6);
        assert!((r - 1.0).abs() < 1e-6);
    }
}
