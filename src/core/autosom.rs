use crate::{
    core::{
        densom::DenSom,
        distance::DistanceFunction,
        evals::{calinski_harabasz_score, silhouette_score},
        kmeans::{KMeansBuilder, KMeansInit},
        som::{InitMethod, SomBuilder},
        SigmaMethod,
    },
    SomError,
};
use ndarray::{Array1, Array2, ArrayView2};
use rand::{SeedableRng, rngs::StdRng, Rng};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmChoice {
    DenSom,
    KMeans,
}

impl std::fmt::Display for AlgorithmChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmChoice::DenSom  => write!(f, "DenSOM"),
            AlgorithmChoice::KMeans => write!(f, "KMeans"),
        }
    }
}

pub struct AutoSomResult {
    /// Final cluster labels. -1 means noise (DenSOM only); KMeans assigns all points ≥ 0.
    pub labels:      Array1<i32>,
    /// Number of clusters found (noise label excluded).
    pub n_clusters:  usize,
    /// Number of clusters detected by the k-selection step (elbow + gap statistic).
    pub k_detected:  usize,
    /// Algorithm that produced the final labels.
    pub algorithm:   AlgorithmChoice,
    /// Fraction of points assigned noise label (-1). Always 0.0 for KMeans.
    pub noise_ratio: f64,
}

/// Builder for the AutoSom automatic clustering pipeline.
pub struct AutoSomBuilder {
    som_m:           usize,
    som_n:           usize,
    dim:             usize,
    learning_rate:   Option<f64>,
    neighbor_radius: f64,
    init_method:     InitMethod,
    distance:        DistanceFunction,
    smooth_sigma:    f64,
    sigma_method:    SigmaMethod,
    /// Number of reference datasets for the gap statistic (used in k-selection). Default: 10.
    gap_n_refs:      usize,
}

impl Default for AutoSomBuilder {
    fn default() -> Self { Self::new() }
}

impl AutoSomBuilder {
    pub fn new() -> Self {
        Self {
            som_m:           10,
            som_n:           10,
            dim:             1,
            learning_rate:   None,
            neighbor_radius: 3.0,
            init_method:     InitMethod::Random,
            distance:        DistanceFunction::Euclidean,
            smooth_sigma:    1.0,
            sigma_method:    SigmaMethod::Silverman,
            gap_n_refs:      10,
        }
    }

    pub fn grid(mut self, m: usize, n: usize) -> Self {
        self.som_m = m;
        self.som_n = n;
        self
    }

    pub fn dim(mut self, d: usize) -> Self {
        self.dim = d;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Result<Self, SomError> {
        if lr > 1.76 {
            return Err(SomError::InvalidLearningRate(lr));
        }
        self.learning_rate = Some(lr);
        Ok(self)
    }

    pub fn neighbor_radius(mut self, r: f64) -> Self {
        self.neighbor_radius = r;
        self
    }

    pub fn init_method(mut self, m: InitMethod) -> Self {
        self.init_method = m;
        self
    }

    pub fn distance(mut self, d: DistanceFunction) -> Self {
        self.distance = d;
        self
    }

    pub fn smooth_sigma(mut self, s: f64) -> Self {
        self.smooth_sigma = s.max(1e-6);
        self
    }

    pub fn sigma_method(mut self, m: SigmaMethod) -> Self {
        self.sigma_method = m;
        self
    }

    /// Set the number of reference datasets for the gap statistic. Clamped to ≥ 1. Default: 10.
    pub fn gap_n_refs(mut self, n: usize) -> Self {
        self.gap_n_refs = n.max(1);
        self
    }

    pub fn build(self) -> AutoSom {
        AutoSom { cfg: self }
    }
}

/// Automatic clustering pipeline.
///
/// 1. Train SOM — learn data topology.
/// 2. Sweep k=2..max_k: cluster neurons with KMeans(k) × N restarts,
///    map data through BMU, score with combined silhouette + CH on data sample.
/// 3. **Run KMeans directly on raw data** with the detected k × N restarts.
///    This is the key: SOM is used only for fast k-detection on the neuron
///    lattice; the final clustering uses unconstrained KMeans centroids on
///    the original data for maximum quality.
/// 4. DenSOM comparison: if density-based clustering produces significantly
///    better silhouette with low noise, prefer it (non-convex shapes).
pub struct AutoSom {
    cfg: AutoSomBuilder,
}

const SIL_SAMPLE: usize = 2000;
const N_RESTARTS: usize = 3;
/// Number of KMeans restarts on the raw data for the final clustering.
const FINAL_RESTARTS: usize = 5;

impl AutoSom {
    pub fn fit_predict(
        &mut self,
        data:       &ArrayView2<f64>,
        epochs:     usize,
        shuffle:    bool,
        batch_size: Option<usize>,
    ) -> Result<AutoSomResult, SomError> {
        let c = &self.cfg;

        // ------------------------------------------------------------------ //
        // Step 1: Train SOM                                                   //
        // ------------------------------------------------------------------ //
        let mut som = {
            let mut b = SomBuilder::new()
                .grid(c.som_m, c.som_n)
                .dim(c.dim)
                .neighbor_radius(c.neighbor_radius)
                .init_method(c.init_method)
                .distance(c.distance);
            if let Some(lr) = c.learning_rate {
                b = b.learning_rate(lr).expect("lr already validated");
            }
            b.build()
        };
        som.fit(data, epochs, shuffle, batch_size)?;

        let bmu_labels = som.predict(data)?;
        let neurons = som.neurons().clone();

        // Build sample for silhouette/CH evaluation on actual data.
        let sample_idx = build_sample_idx(data.nrows(), SIL_SAMPLE);
        let sample_data = subset_rows(data, &sample_idx);

        // ------------------------------------------------------------------ //
        // Step 2: Silhouette + CH sweep on data, KMeans on neurons            //
        // ------------------------------------------------------------------ //
        let mn = c.som_m * c.som_n;
        let max_k = (mn / 2).clamp(2, 50);

        // Collect (sil, ch) for each k to normalize and combine.
        let mut k_scores: Vec<(usize, f64, f64)> = Vec::new();

        for k in 2..=max_k {
            // Multiple restarts: pick the one with best inertia on neurons.
            let mut best_neuron_clusters: Option<Array1<usize>> = None;
            let mut best_inertia = f64::INFINITY;

            for _ in 0..N_RESTARTS {
                let mut km = KMeansBuilder::new()
                    .n_clusters(k)
                    .method(KMeansInit::PlusPlus)
                    .max_iters(100)
                    .build();
                if km.fit(&neurons.view()).is_err() {
                    continue;
                }
                let inertia = km.inertia().unwrap_or(f64::INFINITY);
                if inertia < best_inertia {
                    best_inertia = inertia;
                    best_neuron_clusters = km.predict(&neurons.view()).ok();
                }
            }

            let neuron_clusters = match best_neuron_clusters {
                Some(nc) => nc,
                None => continue,
            };

            let sample_labels = Array1::from_shape_fn(sample_idx.len(), |i| {
                neuron_clusters[bmu_labels[sample_idx[i]]]
            });

            let sil = silhouette_score(&sample_data.view(), &sample_labels.view())
                .unwrap_or(f64::NAN);
            let ch = calinski_harabasz_score(&sample_data.view(), &sample_labels.view())
                .unwrap_or(f64::NAN);

            if sil.is_finite() && ch.is_finite() {
                k_scores.push((k, sil, ch));
            }
        }

        // Find best k using combined normalized score.
        // Silhouette: higher = better (-1 to 1 range, use directly).
        // CH: higher = better, normalize to 0..1 via min-max.
        let best_k = if k_scores.is_empty() {
            2
        } else {
            let ch_min = k_scores.iter().map(|s| s.2).fold(f64::INFINITY, f64::min);
            let ch_max = k_scores.iter().map(|s| s.2).fold(f64::NEG_INFINITY, f64::max);
            let ch_range = (ch_max - ch_min).max(1e-12);

            let sil_min = k_scores.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
            let sil_max = k_scores.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
            let sil_range = (sil_max - sil_min).max(1e-12);

            // Combined = normalized_sil + 0.5 * normalized_ch
            // CH weighted less because it can be noisy, but it helps break ties
            // and favors the correct k for Gaussian mixtures.
            let mut best_score = f64::NEG_INFINITY;
            let mut bk = 2;
            for &(k, sil, ch) in &k_scores {
                let norm_sil = (sil - sil_min) / sil_range;
                let norm_ch = (ch - ch_min) / ch_range;
                let combined = norm_sil + 0.5 * norm_ch;
                if combined > best_score {
                    best_score = combined;
                    bk = k;
                }
            }
            bk
        };

        // ------------------------------------------------------------------ //
        // Step 3: Run KMeans directly on raw data with best_k                 //
        //         Multiple restarts, pick lowest inertia — this is exactly    //
        //         what makes oracle KMeans strong, now with auto-detected k.  //
        // ------------------------------------------------------------------ //
        let km_labels = {
            let mut best_labels: Option<Array1<usize>> = None;
            let mut best_inertia = f64::INFINITY;
            for _ in 0..FINAL_RESTARTS {
                let mut km = KMeansBuilder::new()
                    .n_clusters(best_k)
                    .method(KMeansInit::PlusPlus)
                    .max_iters(300)
                    .build();
                if km.fit(data).is_err() {
                    continue;
                }
                let inertia = km.inertia().unwrap_or(f64::INFINITY);
                if inertia < best_inertia {
                    best_inertia = inertia;
                    best_labels = km.predict(data).ok();
                }
            }
            best_labels.unwrap_or_else(|| {
                // Fallback: single attempt
                som.predict_clustered(data, best_k)
                    .expect("fallback predict_clustered failed")
            })
        };
        let km_sil = {
            let sample_labels = Array1::from_shape_fn(sample_idx.len(), |i| {
                km_labels[sample_idx[i]]
            });
            silhouette_score(&sample_data.view(), &sample_labels.view())
                .unwrap_or(f64::NAN)
        };

        // ------------------------------------------------------------------ //
        // Step 4: DenSOM comparison                                           //
        // ------------------------------------------------------------------ //
        let mut densom = DenSom::from_som(som.clone());
        let adaptive_sigma = compute_adaptive_sigma(data, c.som_m, c.som_n, c.sigma_method);
        densom.set_smooth_sigma(adaptive_sigma);
        densom.refit_density(data)?;
        let ds_result = densom.predict(data)?;

        let ds_sil = {
            let non_noise_sample: Vec<usize> = sample_idx.iter()
                .copied()
                .filter(|&i| ds_result.labels[i] >= 0)
                .collect();
            if non_noise_sample.len() >= 4 {
                let ds_data = subset_rows(data, &non_noise_sample);
                let ds_labels = Array1::from_shape_fn(non_noise_sample.len(), |i| {
                    ds_result.labels[non_noise_sample[i]] as usize
                });
                silhouette_score(&ds_data.view(), &ds_labels.view())
                    .unwrap_or(f64::NAN)
            } else {
                f64::NAN
            }
        };

        // ------------------------------------------------------------------ //
        // Step 5: Pick winner                                                 //
        // ------------------------------------------------------------------ //
        let ds_n = ds_result.n_clusters;
        // DenSOM wins when:
        // - It found >= 2 clusters
        // - Noise is under 20%
        // - Its silhouette is meaningfully better than KMeans
        // OR when KMeans silhouette is very poor (< 0.05, suggesting non-convex data)
        //    and DenSOM has decent silhouette
        let use_densom = ds_sil.is_finite() && ds_n >= 2 && ds_result.noise_ratio < 0.20
            && (ds_sil > km_sil + 0.02
                || (km_sil < 0.05 && ds_sil > 0.1));

        if use_densom {
            Ok(AutoSomResult {
                labels:      ds_result.labels,
                n_clusters:  ds_n,
                k_detected:  best_k,
                algorithm:   AlgorithmChoice::DenSom,
                noise_ratio: ds_result.noise_ratio,
            })
        } else {
            // km_labels is Array1<usize> from raw KMeans
            let labels_i32 = km_labels.mapv(|l| l as i32);
            Ok(AutoSomResult {
                labels:      labels_i32,
                n_clusters:  best_k,
                k_detected:  best_k,
                algorithm:   AlgorithmChoice::KMeans,
                noise_ratio: 0.0,
            })
        }
    }
}

fn build_sample_idx(n: usize, max_n: usize) -> Vec<usize> {
    if n <= max_n {
        return (0..n).collect();
    }
    let step = n / max_n;
    (0..n).step_by(step).take(max_n).collect()
}

fn subset_rows(data: &ArrayView2<f64>, idx: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((idx.len(), data.ncols()), |(i, j)| data[[idx[i], j]])
}

/// Detects the elbow k from an inertia curve using the second-difference (Kneedle) method.
///
/// Returns `(k_elbow, candidates)` where candidates is a sorted, deduplicated Vec
/// of up to 5 k values centered on k_elbow, clamped to `[2, max_k]`.
///
/// Fallbacks (in order):
/// 1. If fewer than 3 inertia points (no interior points): return (2, vec![2]).
/// 2. If max second-difference < 1e-9 (linear curve): set k_elbow = 2.
fn elbow_candidates(
    inertias: &[(usize, f64)], // (k, inertia) pairs, must be sorted by k ascending
    max_k:    usize,
) -> (usize, Vec<usize>) {
    // Need ≥3 points to compute a second difference.
    if inertias.len() < 3 {
        return (2, vec![2]);
    }

    let w_min = inertias.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let w_max = inertias.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let w_range = (w_max - w_min).max(1e-12);

    let w_norm: Vec<f64> = inertias.iter()
        .map(|p| (p.1 - w_min) / w_range)
        .collect();

    // Compute second differences.
    let mut d2: Vec<f64> = Vec::new();
    for i in 1..w_norm.len() - 1 {
        d2.push(w_norm[i - 1] - 2.0 * w_norm[i] + w_norm[i + 1]);
    }

    let k_elbow = if d2.is_empty() {
        2
    } else if d2.len() == 1 {
        // Only one second difference value; use max second derivative
        let best_d2 = d2[0];
        if best_d2 < 1e-9 {
            2
        } else {
            inertias[1].0
        }
    } else {
        // Multiple second differences. Find where the ratio d2[i+1]/d2[i] is smallest.
        // This indicates where the curve transitions from steep (high d2) to flat (low d2).
        let mut best_ratio = f64::INFINITY;
        let mut best_idx = d2.len() - 1;
        let mut found_drop = false;

        for i in 0..d2.len() - 1 {
            if d2[i] > 1e-9 {
                let ratio = d2[i + 1] / d2[i];
                if ratio < best_ratio {
                    best_ratio = ratio;
                    best_idx = i;
                    found_drop = true;
                }
            }
        }

        if !found_drop || best_ratio > 0.8 {
            // No significant drop, linear curve or no clear elbow
            2
        } else {
            // The elbow is at the point where curvature drops.
            // d2[best_idx] is second diff at w_norm[best_idx+1] = inertias[best_idx+1].
            inertias[(best_idx + 1).min(inertias.len() - 1)].0
        }
    };

    // Build candidate window: {k_elbow-2, k_elbow-1, k_elbow, k_elbow+1, k_elbow+2}, clamped, deduplicated.
    // This provides a 5-value window for broader search per spec.
    let mut cands: Vec<usize> = [
        k_elbow.saturating_sub(2).max(2),
        k_elbow.saturating_sub(1).max(2),
        k_elbow,
        (k_elbow + 1).min(max_k),
        (k_elbow + 2).min(max_k),
    ]
    .into_iter()
    .collect();
    cands.sort();
    cands.dedup();

    (k_elbow, cands)
}

/// Compute bandwidth `h` for DenSOM Gaussian smoothing from data statistics.
///
/// Uses Silverman's rule (default) or Scott's rule across all d dimensions,
/// averaging per-dimension std and IQR. Clamps result to `[0.5, max(m,n)/2]`.
fn compute_adaptive_sigma(
    data:   &ArrayView2<f64>,
    m:      usize,
    n:      usize,
    method: SigmaMethod,
) -> f64 {
    let n_samples = data.nrows();
    let d = data.ncols();
    let grid_side = m.max(n) as f64;

    let mut std_sum = 0.0f64;
    let mut iqr_sum = 0.0f64;

    for j in 0..d {
        let mut col: Vec<f64> = data.column(j).to_vec();
        let mean = col.iter().sum::<f64>() / n_samples as f64;
        let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / n_samples as f64;
        std_sum += variance.sqrt();

        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q25_idx = ((n_samples as f64 * 0.25) as usize).min(n_samples - 1);
        let q75_idx = ((n_samples as f64 * 0.75) as usize).min(n_samples - 1);
        iqr_sum += col[q75_idx] - col[q25_idx];
    }

    let sigma_data = (std_sum / d as f64).max(1e-12);
    let iqr_mean   = (iqr_sum / d as f64).max(1e-12);

    let h = match method {
        SigmaMethod::Scott => {
            let exp = -1.0 / (d as f64 + 4.0);
            (n_samples as f64).powf(exp) * sigma_data
        }
        SigmaMethod::Silverman => {
            let iqr_adj = iqr_mean / 1.34;
            0.9 * sigma_data.min(iqr_adj) * (n_samples as f64).powf(-0.2)
        }
    };

    h.clamp(0.5, grid_side / 2.0)
}

/// Gap statistic (Tibshirani et al. 2001) for k selection.
///
/// For each candidate k (sorted ascending), computes:
///   Gap(k) = E[log W_k^ref] - log W_k
///
/// where W_k^ref is within-cluster dispersion on `n_refs` uniformly random
/// reference datasets drawn from the data's bounding box.
///
/// Uses Tibshirani's stopping criterion over the sorted candidate list:
///   best_k = smallest k where Gap(k) ≥ Gap(k_next) - SE(k_next)
///
/// Falls back to k with maximum Gap, or `k_elbow` if all fits fail.
fn gap_statistic(
    sample_data: &ArrayView2<f64>,
    candidates:  &[usize],
    n_refs:      usize,
    k_elbow:     usize,
) -> usize {
    let n = sample_data.nrows();
    let d = sample_data.ncols();

    // Per-dimension bounding box for reference data generation.
    let mut col_min = vec![f64::INFINITY;     d];
    let mut col_max = vec![f64::NEG_INFINITY; d];
    for j in 0..d {
        for i in 0..n {
            let v = sample_data[[i, j]];
            if v < col_min[j] { col_min[j] = v; }
            if v > col_max[j] { col_max[j] = v; }
        }
    }

    // Per-call seed derived from sample size for reproducibility.
    let base_seed = n as u64;
    let mut gap_vals: Vec<(usize, f64, f64)> = Vec::new(); // (k, gap, se)

    for &k in candidates {
        if n < 2 * k { continue; } // too few points for this k

        // Inertia on actual data.
        let mut km = KMeansBuilder::new()
            .n_clusters(k)
            .method(KMeansInit::PlusPlus)
            .max_iters(100)
            .build();
        if km.fit(sample_data).is_err() { continue; }
        let wk = km.inertia().unwrap_or(0.0);
        let log_wk = if wk > 0.0 { wk.ln() } else { continue; };

        // Inertia on n_refs uniform reference datasets.
        let mut ref_log_wks: Vec<f64> = Vec::with_capacity(n_refs);
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(k as u64));

        for _ in 0..n_refs {
            let ref_data = Array2::from_shape_fn((n, d), |(_i, j)| {
                let lo = col_min[j];
                let hi = col_max[j];
                if (hi - lo).abs() < 1e-12 { lo }
                else { lo + rng.random::<f64>() * (hi - lo) }
            });
            let mut km_ref = KMeansBuilder::new()
                .n_clusters(k)
                .method(KMeansInit::PlusPlus)
                .max_iters(100)
                .build();
            if km_ref.fit(&ref_data.view()).is_err() { continue; }
            let w_ref = km_ref.inertia().unwrap_or(0.0);
            if w_ref > 0.0 { ref_log_wks.push(w_ref.ln()); }
        }

        if ref_log_wks.is_empty() { continue; }

        let b        = ref_log_wks.len() as f64;
        let mean_ref = ref_log_wks.iter().sum::<f64>() / b;
        let gap      = mean_ref - log_wk;
        let std_ref  = if b > 1.0 {
            (ref_log_wks.iter()
                .map(|&x| (x - mean_ref).powi(2))
                .sum::<f64>()
                / (b - 1.0)).sqrt()
        } else {
            0.0
        };
        let se = std_ref * (1.0 + 1.0 / b).sqrt();

        gap_vals.push((k, gap, se));
    }

    if gap_vals.is_empty() {
        return k_elbow;
    }

    // Tibshirani criterion: smallest k where Gap(k) ≥ Gap(k_next) - SE(k_next).
    // k_next is the next element in gap_vals (not necessarily k+1 in integers).
    for i in 0..gap_vals.len().saturating_sub(1) {
        let (k, gap_k, _)          = gap_vals[i];
        let (_, gap_next, se_next) = gap_vals[i + 1];
        if gap_k >= gap_next - se_next {
            return k;
        }
    }

    // Fallback: k with maximum Gap(k).
    gap_vals.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(k, _, _)| k)
        .unwrap_or(k_elbow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn silverman_sigma_formula() {
        use crate::core::SigmaMethod;
        // Build a 2D dataset with known statistics
        let mut v = Vec::with_capacity(200);
        for i in 0..100i64 {
            let x = -2.0 + 4.0 * i as f64 / 99.0;
            v.push(x); v.push(x * 0.5);
        }
        let data = Array2::from_shape_vec((100, 2), v).unwrap();
        let h = compute_adaptive_sigma(&data.view(), 10, 10, SigmaMethod::Silverman);
        // Result must be in [0.5, 5.0] (clamped range for 10×10 grid: grid_side/2 = 5)
        assert!(h >= 0.5 && h <= 5.0, "sigma {h} out of clamped range [0.5, 5.0]");
        // Must be deterministic
        let h2 = compute_adaptive_sigma(&data.view(), 10, 10, SigmaMethod::Silverman);
        assert_eq!(h, h2, "sigma computation must be deterministic");
    }

    fn make_blobs(n: usize) -> Array2<f64> {
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n { let t = i as f64 * 0.01; v.push(t); v.push(t); }
        for i in 0..n { let t = i as f64 * 0.01; v.push(8.0 + t); v.push(t); }
        Array2::from_shape_vec((n * 2, 2), v).unwrap()
    }

    #[test]
    fn autosom_blobs_finds_two_clusters() {
        let data = make_blobs(50);
        let mut auto = AutoSomBuilder::new()
            .grid(8, 8)
            .dim(2)
            .init_method(InitMethod::NaiveSharding)
            .build();
        let result = auto
            .fit_predict(&data.view(), 10, false, None)
            .unwrap();
        assert!(result.n_clusters >= 2,
            "two blobs → ≥2 clusters, got {} via {}",
            result.n_clusters, result.algorithm);
    }

    #[test]
    fn autosom_returns_valid_labels() {
        let data = make_blobs(30);
        let mut auto = AutoSomBuilder::new()
            .grid(6, 6)
            .dim(2)
            .build();
        let result = auto
            .fit_predict(&data.view(), 5, false, None)
            .unwrap();
        assert_eq!(result.labels.len(), data.nrows());
        for &l in result.labels.iter() {
            assert!(l >= -1 && (l as usize) < result.n_clusters + 1,
                "label {l} out of range for n_clusters={}", result.n_clusters);
        }
    }

    #[test]
    fn elbow_detect_known_knee() {
        // Inertia drops steeply from k=2 to k=5, then flattens.
        // The knee should be detected at k=5.
        let inertias: Vec<(usize, f64)> = vec![
            (2, 1000.0), (3, 500.0), (4, 200.0), (5, 50.0),
            (6, 45.0), (7, 42.0), (8, 40.0),
        ];
        let (k_elbow, _) = elbow_candidates(&inertias, 8);
        assert_eq!(k_elbow, 5, "knee at k=5, got {k_elbow}");
    }

    #[test]
    fn elbow_detect_linear_fallback() {
        // Perfectly linear inertia — no elbow — should fall back to k=2.
        let inertias: Vec<(usize, f64)> = (2..=8).map(|k| (k, 100.0 - k as f64 * 10.0)).collect();
        let (k_elbow, _) = elbow_candidates(&inertias, 8);
        assert_eq!(k_elbow, 2, "linear curve → fallback k=2, got {k_elbow}");
    }

    #[test]
    fn elbow_candidates_clamped_dedup() {
        // With ±2 window: k_elbow → candidates {k_elbow-2, k_elbow-1, k_elbow, k_elbow+1, k_elbow+2}
        // When clamped/dedup'd, length depends on k_elbow position. The spec requires ≤5 candidates.
        // Edge case: k_elbow = max_k = 6 → window {4, 5, 6, 6, 6} dedup → {4, 5, 6} (len=3 < 5 ✓)
        let max_k = 6usize;
        let inertias: Vec<(usize, f64)> = vec![
            (2, 1000.0), (3, 500.0), (4, 200.0),
            (5, 100.0), (6, 10.0),
        ];
        let (_, cands) = elbow_candidates(&inertias, max_k);
        // Candidates must be in valid range
        assert!(cands.iter().all(|&k| k >= 2 && k <= max_k),
            "all candidates must be in [2, max_k={max_k}], got {:?}", cands);
        // Must be sorted and deduplicated
        let mut deduped = cands.clone();
        deduped.dedup();
        assert_eq!(deduped, cands, "candidates must be sorted with no duplicates");
        // For this curve, window must have length ≤5 (the spec requirement)
        assert!(cands.len() <= 5, "candidate window must have ≤5 elements per spec, got {:?}", cands);
    }

    #[test]
    fn elbow_empty_sweep_fallback() {
        // max_k=2: only one inertia value, second-difference array is empty.
        // Must return k_elbow=2 without panicking.
        let inertias: Vec<(usize, f64)> = vec![(2, 100.0)];
        let (k_elbow, cands) = elbow_candidates(&inertias, 2);
        assert_eq!(k_elbow, 2);
        assert_eq!(cands, vec![2]);
    }

    #[test]
    fn gap_statistic_two_blobs() {
        // Two tight, well-separated blobs at (0,0) and (10,10).
        // Gap statistic with a fixed seed must select k=2.
        let mut v = Vec::with_capacity(200);
        for i in 0..50i64 { v.push(i as f64 * 0.01); v.push(i as f64 * 0.01); }
        for i in 0..50i64 { v.push(10.0 + i as f64 * 0.01); v.push(10.0 + i as f64 * 0.01); }
        let data = Array2::from_shape_vec((100, 2), v).unwrap();
        let candidates = vec![2, 3, 4];
        let best = gap_statistic(&data.view(), &candidates, 10, 2);
        assert_eq!(best, 2, "two blobs → gap must pick k=2, got k={best}");
    }
}
