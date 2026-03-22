use crate::SomError;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EvalMethod {
    Silhouette,
    DaviesBouldin,
    CalinskiHarabasz,
    Dunn,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ClassEvalMethod {
    Accuracy,
    F1,
    Recall,
    All,
}

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
                let sum: f64 = idx
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| dists[[i, j]])
                    .sum();
                a[i] = sum / (sz - 1) as f64;
            }
        }
        for &other in &unique {
            if other == label {
                continue;
            }
            let other_idx: Vec<usize> = (0..n).filter(|&i| labels[i] == other).collect();
            for &i in &idx {
                let mean: f64 =
                    other_idx.iter().map(|&j| dists[[i, j]]).sum::<f64>() / other_idx.len() as f64;
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
            if denom == 0.0 {
                None
            } else {
                Some((b[i] - a[i]) / denom)
            }
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

pub fn dunn_index(data: &ArrayView2<f64>, labels: &ArrayView1<usize>) -> Result<f64, SomError> {
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

/// Density-Based Clustering Validation (DBCV).
///
/// Evaluates clustering quality by comparing within-cluster density
/// (sparseness) against between-cluster density separation, using mutual
/// reachability distances.  Unlike CH or silhouette, DBCV correctly
/// handles non-convex / non-globular clusters (rings, spirals, etc.)
/// because it measures density connectivity rather than centroid separation.
///
/// Based on Moulavi et al. (2014), "Density-Based Clustering Validation."
///
/// Returns a score in \[-1, 1\] where higher is better.  Complexity: O(n²).
pub fn dbcv_score(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
    core_k: usize,
) -> Result<f64, SomError> {
    let n = data.nrows();
    if n < 2 || core_k == 0 {
        return Ok(0.0);
    }
    let core_k = core_k.min(n - 1);

    let unique: Vec<usize> = {
        let mut v = labels.to_vec();
        v.sort();
        v.dedup();
        v
    };
    if unique.len() <= 1 {
        return Ok(0.0);
    }

    // Step 1: Pairwise Euclidean distances.
    let dists = pairwise_distances(data);

    // Step 2: Core distances (distance to core_k-th nearest neighbour).
    let mut core_dist = vec![0.0f64; n];
    let mut row_dists = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            row_dists[j] = dists[[i, j]];
        }
        row_dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // row_dists[0] == 0 (self), so k-th neighbour is row_dists[core_k].
        core_dist[i] = row_dists[core_k.min(n - 1)];
    }

    // Mutual reachability distance (computed on demand via closure).
    let mr = |i: usize, j: usize| -> f64 {
        core_dist[i].max(core_dist[j]).max(dists[[i, j]])
    };

    // Step 3: For each cluster, build MST via Prim's on mutual reachability,
    //         track max edge (density sparseness = DSC).
    let mut dsc: Vec<f64> = Vec::with_capacity(unique.len());
    for &label in &unique {
        let members: Vec<usize> = (0..n).filter(|&i| labels[i] == label).collect();
        if members.len() <= 1 {
            dsc.push(0.0);
            continue;
        }
        // Prim's MST — dense graph, O(|C|²).
        let m = members.len();
        let mut in_tree = vec![false; m];
        let mut min_edge = vec![f64::INFINITY; m];
        min_edge[0] = 0.0;
        let mut max_mst_edge = 0.0f64;

        for _ in 0..m {
            // Pick the not-in-tree vertex with smallest edge.
            let u = (0..m)
                .filter(|&v| !in_tree[v])
                .min_by(|&a, &b| min_edge[a].partial_cmp(&min_edge[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            in_tree[u] = true;
            if min_edge[u] > max_mst_edge && min_edge[u].is_finite() {
                max_mst_edge = min_edge[u];
            }
            // Update neighbours.
            for v in 0..m {
                if !in_tree[v] {
                    let w = mr(members[u], members[v]);
                    if w < min_edge[v] {
                        min_edge[v] = w;
                    }
                }
            }
        }
        dsc.push(max_mst_edge);
    }

    // Step 4: For each pair of clusters, compute density separation
    //         (min mutual reachability distance between them).
    let k_c = unique.len();
    let mut dspc = vec![vec![f64::INFINITY; k_c]; k_c];
    for ci in 0..k_c {
        let members_i: Vec<usize> = (0..n).filter(|&x| labels[x] == unique[ci]).collect();
        for cj in (ci + 1)..k_c {
            let members_j: Vec<usize> = (0..n).filter(|&x| labels[x] == unique[cj]).collect();
            let mut min_mr = f64::INFINITY;
            for &a in &members_i {
                for &b in &members_j {
                    let d = mr(a, b);
                    if d < min_mr {
                        min_mr = d;
                    }
                }
            }
            dspc[ci][cj] = min_mr;
            dspc[cj][ci] = min_mr;
        }
    }

    // Step 5: Per-cluster validity and weighted average.
    let mut total = 0.0f64;
    let mut total_weight = 0.0f64;
    for ci in 0..k_c {
        let n_c = (0..n).filter(|&x| labels[x] == unique[ci]).count() as f64;
        // Minimum density separation to any other cluster.
        let sep = (0..k_c)
            .filter(|&cj| cj != ci)
            .map(|cj| dspc[ci][cj])
            .fold(f64::INFINITY, f64::min);

        let spar = dsc[ci];
        let denom = sep.max(spar);
        let v_c = if denom < 1e-12 { 0.0 } else { (sep - spar) / denom };
        total += n_c * v_c;
        total_weight += n_c;
    }

    if total_weight < 1e-12 {
        return Ok(0.0);
    }
    Ok(total / total_weight)
}

/// Topology-aware cluster quality metric based on k-nearest-neighbour label
/// consistency.
///
/// For each point, computes the fraction of its `k` nearest neighbours (in
/// Euclidean distance) that share the same cluster label, then averages
/// across all points.  The raw score is **adjusted for chance**: a single
/// cluster trivially scores 1.0, so we subtract the expected score under
/// random (proportional) assignment and normalise to [0, 1].
///
/// **Why this metric?** Standard metrics (silhouette, CH, DB) assume convex
/// / globular clusters.  For non-convex data like concentric rings or
/// spirals, they favour KMeans over density-based methods even when the
/// latter produces a better partition.  k-NN consistency captures *local*
/// neighbourhood preservation: a good clustering keeps nearby points
/// together regardless of global cluster shape.
///
/// Complexity: O(n²) — same as silhouette (pairwise distance computation).
///
/// Returns `Ok(score)` in [0, 1] where higher is better, or 0.0 for
/// degenerate inputs (single cluster, n < 2, etc.).
pub fn knn_consistency_score(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
    k: usize,
) -> Result<f64, SomError> {
    let n = data.nrows();
    if n < 2 || k == 0 {
        return Ok(0.0);
    }
    let k = k.min(n - 1);

    // Unique labels and their sizes (for chance correction).
    let unique: Vec<usize> = {
        let mut v = labels.to_vec();
        v.sort();
        v.dedup();
        v
    };
    if unique.len() <= 1 {
        return Ok(0.0); // single cluster → trivially perfect, adjusted = 0
    }

    // Herfindahl index: expected consistency under random proportional assignment.
    let expected: f64 = unique
        .iter()
        .map(|&l| {
            let cnt = labels.iter().filter(|&&x| x == l).count() as f64;
            (cnt / n as f64).powi(2)
        })
        .sum();

    // Pairwise Euclidean distances (flat vec for cache-friendliness).
    let mut dists = vec![0.0f64; n * n];
    for i in 0..n {
        let ri = data.row(i);
        for j in (i + 1)..n {
            let d = (&ri - &data.row(j)).mapv(|x| x * x).sum().sqrt();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }

    // For each point, find k-NN same-label fraction.
    let mut total_consistency = 0.0f64;
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..n {
        // Partial sort by distance to point i.
        indices.iter_mut().enumerate().for_each(|(idx, v)| *v = idx);
        indices.sort_unstable_by(|&a, &b| {
            dists[i * n + a]
                .partial_cmp(&dists[i * n + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let same = indices
            .iter()
            .filter(|&&j| j != i)
            .take(k)
            .filter(|&&j| labels[j] == labels[i])
            .count();

        total_consistency += same as f64 / k as f64;
    }

    let observed = total_consistency / n as f64;

    // Adjusted = (observed − expected) / (1 − expected), clamped to [0, 1].
    let denom = 1.0 - expected;
    if denom < 1e-12 {
        return Ok(0.0);
    }
    Ok(((observed - expected) / denom).clamp(0.0, 1.0))
}

pub fn bcubed_scores(clusters: &ArrayView1<usize>, labels: &ArrayView1<usize>) -> (f64, f64) {
    let n = clusters.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let (mut prec, mut rec) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        let same_cluster: Vec<usize> = (0..n).filter(|&j| clusters[j] == clusters[i]).collect();
        let same_label: Vec<usize> = (0..n).filter(|&j| labels[j] == labels[i]).collect();
        let both = same_cluster
            .iter()
            .filter(|&&j| labels[j] == labels[i])
            .count();
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
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / n as f64 * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn perfect_clusters() -> (Array2<f64>, ndarray::Array1<usize>) {
        let data = Array2::from_shape_fn((20, 2), |(i, j)| {
            if i < 10 {
                i as f64 * 0.01 + j as f64 * 0.01
            } else {
                100.0 + i as f64 * 0.01
            }
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
        assert!(
            s > 0.5,
            "expected high silhouette for separated clusters, got {}",
            s
        );
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
    fn knn_consistency_high_for_separated() {
        let (d, l) = perfect_clusters();
        let score = knn_consistency_score(&d.view(), &l.view(), 5).unwrap();
        assert!(
            score > 0.5,
            "expected high knn_consistency for separated clusters, got {}",
            score
        );
    }

    #[test]
    fn knn_consistency_single_cluster_zero() {
        let d = Array2::from_shape_fn((10, 2), |(i, j)| i as f64 + j as f64 * 0.1);
        let l = ndarray::Array1::zeros(10);
        let score = knn_consistency_score(&d.view(), &l.view(), 3).unwrap();
        assert!(
            score.abs() < 1e-6,
            "single cluster should give adjusted score 0, got {}",
            score
        );
    }

    #[test]
    fn bcubed_perfect() {
        let clusters = ndarray::Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let labels = ndarray::Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let (p, r) = bcubed_scores(&clusters.view(), &labels.view());
        assert!((p - 1.0).abs() < 1e-6);
        assert!((r - 1.0).abs() < 1e-6);
    }
}
