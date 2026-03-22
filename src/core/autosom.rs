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
    pub labels:      Array1<i32>,
    pub n_clusters:  usize,
    pub k_detected:  usize,
    pub algorithm:   AlgorithmChoice,
    pub noise_ratio: f64,
}

pub struct AutoSomBuilder {
    som_m:           usize,
    som_n:           usize,
    dim:             usize,
    learning_rate:   Option<f64>,
    neighbor_radius: f64,
    init_method:     InitMethod,
    distance:        DistanceFunction,
    smooth_sigma:    f64,
    sigma_method:    SigmaMethod,   // NEW
    gap_n_refs:      usize,         // NEW
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
        let max_k = (mn / 2).min(50).max(2);

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
}
