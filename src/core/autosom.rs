use crate::{
    core::{
        densom::DenSomBuilder,
        distance::DistanceFunction,
        evals::{dbcv_score, silhouette_score},
        kmeans::{KMeansBuilder, KMeansInit},
        som::InitMethod,
    },
    SomError,
};
use ndarray::{Array1, Array2, ArrayView2};

/// Points above this count skip the O(n²) DBCV path and use
/// the faster noise-ratio threshold rule instead.
const DBCV_LIMIT: usize = 5_000;

/// Which algorithm was selected by the automatic pipeline.
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

/// Result returned by `AutoSom::fit_predict`.
pub struct AutoSomResult {
    /// Final cluster labels.  `-1` means noise and only appears when DenSOM
    /// was selected; KMeans always assigns every point.
    pub labels:      Array1<i32>,
    /// Number of clusters in the final assignment (noise label excluded).
    pub n_clusters:  usize,
    /// Auto-detected k from DenSOM density peaks.  Used as the KMeans `k`
    /// when the KMeans path is taken.
    pub k_detected:  usize,
    /// Which algorithm produced the final labels.
    pub algorithm:   AlgorithmChoice,
    /// Fraction of points labelled as noise (0.0 when KMeans is chosen).
    pub noise_ratio: f64,
}

/// Builder for `AutoSom`.  Mirrors `DenSomBuilder`'s API so it can be dropped
/// in as a replacement with zero configuration change.
pub struct AutoSomBuilder {
    som_m:           usize,
    som_n:           usize,
    dim:             usize,
    learning_rate:   Option<f64>,
    neighbor_radius: f64,
    init_method:     InitMethod,
    distance:        DistanceFunction,
    smooth_sigma:    f64,
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

    /// Returns `Err` if `lr > 1.76`, matching `SomBuilder` validation.
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

    /// Gaussian smoothing sigma passed to DenSOM.  Default `1.0`.
    pub fn smooth_sigma(mut self, s: f64) -> Self {
        self.smooth_sigma = s.max(1e-6);
        self
    }

    /// Infallible — all validation happens in individual setter methods.
    pub fn build(self) -> AutoSom {
        AutoSom { cfg: self }
    }
}

/// Automatic algorithm selector that combines DenSOM (for auto-k detection
/// and density-based clustering) with KMeans (for globular high-k datasets).
///
/// # Pipeline
///
/// 1. Train DenSOM → get `k_detected` (density peak count) and `noise_ratio`.
/// 2. Train KMeans with `k = max(2, k_detected)`.
/// 3. Score both using **metric consensus** (DBCV + silhouette):
///    - DenSOM noise points are reassigned to nearest cluster centroid
///      so both methods are compared on full data with 100% assignment.
///    - DBCV captures density-based quality; silhouette captures compactness.
///    - DenSOM is chosen ONLY when **both** metrics prefer it, avoiding
///      false positives from either metric's biases.
/// 4. Return whichever scored higher, along with which algorithm was chosen.
pub struct AutoSom {
    cfg: AutoSomBuilder,
}

impl AutoSom {
    pub fn fit_predict(
        &mut self,
        data:       &ArrayView2<f64>,
        epochs:     usize,
        shuffle:    bool,
        batch_size: Option<usize>,
    ) -> Result<AutoSomResult, SomError> {
        // ------------------------------------------------------------------ //
        // Step 1: DenSOM — auto-detects k and provides topology-aware labels  //
        // ------------------------------------------------------------------ //
        let mut densom = {
            let c = &self.cfg;
            let mut b = DenSomBuilder::new()
                .grid(c.som_m, c.som_n)
                .dim(c.dim)
                .neighbor_radius(c.neighbor_radius)
                .init_method(c.init_method)
                .distance(c.distance)
                .smooth_sigma(c.smooth_sigma);
            if let Some(lr) = c.learning_rate {
                b = b.learning_rate(lr).expect("lr already validated in builder");
            }
            b.build()
        };
        let ds_result = densom.fit_predict(data, epochs, shuffle, batch_size)?;
        let k_detected = ds_result.n_clusters.max(2);

        // ------------------------------------------------------------------ //
        // Step 2: KMeans with the auto-detected k                             //
        // ------------------------------------------------------------------ //
        let mut km = KMeansBuilder::new()
            .n_clusters(k_detected)
            .method(KMeansInit::PlusPlus)
            .max_iters(300)
            .build();
        km.fit(data)?;
        let km_labels = km.predict(data)?;

        // ------------------------------------------------------------------ //
        // Step 3: Algorithm selection via metric consensus                     //
        //                                                                     //
        // No single internal metric reliably distinguishes density-based      //
        // from centroid-based clusterings across all data shapes:             //
        //   • Silhouette / CH favour KMeans on non-convex data                //
        //   • DBCV favours DenSOM even on globular data                       //
        //                                                                     //
        // Consensus approach: DenSOM is chosen ONLY when BOTH a density-     //
        // aware metric (DBCV) AND a shape-agnostic metric (silhouette)        //
        // prefer it.  This avoids false positives in both directions.         //
        //                                                                     //
        //  • n ≤ DBCV_LIMIT: compute both metrics, require consensus.         //
        //  • n > DBCV_LIMIT: O(n²) metrics too expensive — use KMeans.        //
        // ------------------------------------------------------------------ //
        let n = data.nrows();
        let dbcv_k = (n as f64).sqrt().round().max(3.0) as usize;

        let use_densom = if ds_result.n_clusters < 2 {
            false
        } else if n <= DBCV_LIMIT {
            // Reassign noise points to nearest cluster centroid so both
            // methods are scored on identical full data.
            let n_cols = data.ncols();
            let n_clusters = ds_result.n_clusters;

            let mut centroids = Array2::<f64>::zeros((n_clusters, n_cols));
            let mut counts = vec![0usize; n_clusters];
            for i in 0..n {
                let l = ds_result.labels[i];
                if l >= 0 {
                    let c = l as usize;
                    centroids.row_mut(c).scaled_add(1.0, &data.row(i));
                    counts[c] += 1;
                }
            }
            for c in 0..n_clusters {
                if counts[c] > 0 {
                    centroids
                        .row_mut(c)
                        .mapv_inplace(|v| v / counts[c] as f64);
                }
            }

            let ds_full_labels: Array1<usize> = Array1::from_shape_fn(n, |i| {
                let l = ds_result.labels[i];
                if l >= 0 {
                    l as usize
                } else {
                    let pt = data.row(i);
                    let mut best_c = 0;
                    let mut best_d = f64::INFINITY;
                    for c in 0..n_clusters {
                        let d = (&pt - &centroids.row(c))
                            .mapv(|x| x * x)
                            .sum();
                        if d < best_d {
                            best_d = d;
                            best_c = c;
                        }
                    }
                    best_c
                }
            });

            // DBCV: density-aware (favours density-contour clusters).
            let dbcv_ds = dbcv_score(&data.view(), &ds_full_labels.view(), dbcv_k)
                .unwrap_or(f64::NEG_INFINITY);
            let dbcv_km = dbcv_score(data, &km_labels.view(), dbcv_k)
                .unwrap_or(f64::NEG_INFINITY);

            // Silhouette: shape-agnostic (favours compact, well-separated).
            let sil_ds = silhouette_score(&data.view(), &ds_full_labels.view())
                .unwrap_or(-1.0);
            let sil_km = silhouette_score(data, &km_labels.view())
                .unwrap_or(-1.0);

            // Consensus: DenSOM wins only if BOTH metrics prefer it.
            dbcv_ds > dbcv_km && sil_ds > sil_km
        } else {
            // For large datasets, O(n²) metrics are too expensive.
            // Default to KMeans — the safer choice for most data.
            false
        };

        // ------------------------------------------------------------------ //
        // Step 4: Return winner                                                //
        // ------------------------------------------------------------------ //
        if use_densom {
            Ok(AutoSomResult {
                labels:      ds_result.labels,
                n_clusters:  ds_result.n_clusters,
                k_detected,
                algorithm:   AlgorithmChoice::DenSom,
                noise_ratio: ds_result.noise_ratio,
            })
        } else {
            let labels_i32 = km_labels.mapv(|l| l as i32);
            Ok(AutoSomResult {
                labels:      labels_i32,
                n_clusters:  k_detected,
                k_detected,
                algorithm:   AlgorithmChoice::KMeans,
                noise_ratio: 0.0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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
        assert_eq!(result.n_clusters, 2,
            "two blobs → 2 clusters, got {} via {}",
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
        // All labels must be >= -1 (noise) and < n_clusters
        for &l in result.labels.iter() {
            assert!(l >= -1 && (l as usize) < result.n_clusters + 1,
                "label {l} out of range for n_clusters={}", result.n_clusters);
        }
    }
}
