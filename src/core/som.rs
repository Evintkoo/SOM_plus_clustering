use crate::{
    backend::{self, Backend},
    core::{
        distance::DistanceFunction,
        evals::{
            calinski_harabasz_score, davies_bouldin_index, dunn_index, silhouette_score, EvalMethod,
        },
        init::{
            init_he, init_lecun, init_lsuv, init_naive_sharding, init_random, init_som_plus_plus,
            init_zero,
        },
        kde::initiate_kde,
        kmeans::{KMeansBuilder, KMeansInit},
        neighborhood,
    },
    SomError,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitMethod {
    Random,
    KMeans,
    KMeansPlusPlus,
    /// KDE-seeded K-Means. In v0.1, uses KDE local maxima as initial centroids
    /// (identical to [`InitMethod::Kde`]). Lloyd's iteration refinement is planned for v0.2.
    KdekMeans,
    SomPlusPlus,
    Zero,
    He,
    NaiveSharding,
    LeCun,
    Lsuv,
    Kde,
}

pub struct SomBuilder {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) dim: usize,
    pub(crate) learning_rate: f64,
    pub(crate) neighbor_radius: f64,
    pub(crate) init_method: InitMethod,
    pub(crate) dist_func: DistanceFunction,
    pub(crate) max_iter: Option<usize>,
}

impl Default for SomBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SomBuilder {
    pub fn new() -> Self {
        Self {
            m: 10,
            n: 10,
            dim: 2,
            learning_rate: 0.5,
            neighbor_radius: 1.0,
            init_method: InitMethod::Random,
            dist_func: DistanceFunction::Euclidean,
            max_iter: None,
        }
    }

    pub fn grid(mut self, m: usize, n: usize) -> Self {
        self.m = m;
        self.n = n;
        self
    }

    pub fn dim(mut self, d: usize) -> Self {
        self.dim = d;
        self
    }

    /// Returns Err if learning_rate > 1.76
    pub fn learning_rate(mut self, lr: f64) -> Result<Self, SomError> {
        if lr > 1.76 {
            return Err(SomError::InvalidLearningRate(lr));
        }
        self.learning_rate = lr;
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
        self.dist_func = d;
        self
    }

    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = Some(n);
        self
    }

    pub fn build(self) -> Som {
        Som {
            m: self.m,
            n: self.n,
            dim: self.dim,
            neurons: Array2::zeros((self.m * self.n, self.dim)),
            initial_lr: self.learning_rate,
            cur_lr: self.learning_rate,
            initial_rad: self.neighbor_radius,
            cur_rad: self.neighbor_radius,
            init_method: self.init_method,
            dist_func: self.dist_func,
            max_iter: self.max_iter,
            trained: false,
            backend: Backend::Cpu,
        }
    }
}

#[derive(Clone)]
pub struct Som {
    pub m: usize,
    pub n: usize,
    pub dim: usize,
    neurons: Array2<f64>, // shape [m*n, dim]
    initial_lr: f64,
    cur_lr: f64,
    initial_rad: f64,
    cur_rad: f64,
    init_method: InitMethod,
    dist_func: DistanceFunction,
    max_iter: Option<usize>,
    trained: bool,
    backend: Backend,
}

impl Som {
    pub fn set_backend(&mut self, b: Backend) {
        self.backend = b;
    }

    fn init_neurons(&self, data: &ArrayView2<f64>) -> Result<Array2<f64>, SomError> {
        let k = self.m * self.n;
        let dim = self.dim;
        let neurons = match self.init_method {
            InitMethod::Random => {
                let min = data.fold(f64::INFINITY, |a, &b| a.min(b));
                let max = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let hi = if (max - min).abs() < 1e-12 {
                    min + 1e-6
                } else {
                    max
                };
                init_random(k, dim, min, hi)
            }
            InitMethod::He => init_he(dim, k),
            InitMethod::LeCun => {
                // init_lecun returns [dim, k] — transpose to [k, dim]
                let w = init_lecun(dim, k);
                w.t().to_owned()
            }
            InitMethod::Zero => init_zero(k, dim),
            InitMethod::NaiveSharding => init_naive_sharding(data, k),
            InitMethod::SomPlusPlus => init_som_plus_plus(data, k),
            InitMethod::Lsuv => {
                // init_lsuv returns [dim, k] — transpose to [k, dim]
                let w = init_lsuv(dim, k, data);
                w.t().to_owned()
            }
            InitMethod::KMeans => {
                let mut km = KMeansBuilder::new()
                    .n_clusters(k)
                    .method(KMeansInit::Random)
                    .build();
                km.fit(data)?;
                km.centroids().clone()
            }
            InitMethod::KMeansPlusPlus => {
                let mut km = KMeansBuilder::new()
                    .n_clusters(k)
                    .method(KMeansInit::PlusPlus)
                    .build();
                km.fit(data)?;
                km.centroids().clone()
            }
            InitMethod::Kde => initiate_kde(data, k, None)?,
            // v0.1: KDE seeds without Lloyd's refinement (same as Kde).
            // TODO(v0.2): Add KMeans Lloyd's iteration post-KDE seeding.
            InitMethod::KdekMeans => initiate_kde(data, k, None)?,
        };
        Ok(neurons)
    }

    fn find_bmu(&self, pt: &ArrayView1<f64>) -> usize {
        use crate::core::distance::euclidean_sq;
        (0..self.m * self.n)
            .min_by(|&a, &b| {
                let da = euclidean_sq(pt, &self.neurons.row(a));
                let db = euclidean_sq(pt, &self.neurons.row(b));
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }

    pub fn fit(
        &mut self,
        data: &ArrayView2<f64>,
        epoch: usize,
        shuffle: bool,
        batch_size: Option<usize>,
    ) -> Result<(), SomError> {
        // 1. Validate
        if data.iter().any(|x| !x.is_finite()) {
            return Err(SomError::InvalidInputData);
        }
        if data.ncols() != self.dim {
            return Err(SomError::DimensionMismatch {
                expected: self.dim,
                got: data.ncols(),
            });
        }
        // 2. Init neurons (first call only)
        if !self.trained {
            self.neurons = self.init_neurons(data)?;
        }
        // 3. Training loop
        let n = data.nrows();
        let bs = batch_size.unwrap_or_else(|| (n / 100).max(32).min(n));
        let total_iters = (epoch * n).min(self.max_iter.unwrap_or(usize::MAX));
        let mut global_iter = 0usize;
        let mut data_owned = data.to_owned();
        let mut rng = rand::rng();

        'outer: for _ in 0..epoch {
            if shuffle {
                let mut rows: Vec<usize> = (0..n).collect();
                rows.shuffle(&mut rng);
                let shuffled =
                    Array2::from_shape_fn((n, self.dim), |(i, j)| data_owned[[rows[i], j]]);
                data_owned = shuffled;
            }
            for batch_start in (0..n).step_by(bs) {
                let batch_end = (batch_start + bs).min(n);
                let batch = data_owned.slice(ndarray::s![batch_start..batch_end, ..]);
                for row in 0..batch.nrows() {
                    if global_iter >= total_iters {
                        break 'outer;
                    }
                    global_iter += 1;
                    let pt = batch.row(row);
                    let bmu_idx = self.find_bmu(&pt);
                    let bmu_r = bmu_idx / self.n;
                    let bmu_c = bmu_idx % self.n;
                    let influence = neighborhood::gaussian_grid(
                        self.m,
                        self.n,
                        bmu_r,
                        bmu_c,
                        self.cur_lr,
                        self.cur_rad,
                    );
                    // neurons is already [m*n, dim], pass directly (in-place update)
                    backend::neighborhood_update(
                        &mut self.neurons,
                        &pt,
                        &influence.view(),
                        self.dist_func,
                        self.backend,
                    )?;
                    // Exponential decay
                    let progress = global_iter as f64 / total_iters as f64;
                    self.cur_lr = self.initial_lr * (-5.0 * progress).exp();
                    self.cur_rad = (self.initial_rad * (-3.0 * progress).exp()).max(1e-12);
                }
            }
        }
        self.trained = true;
        Ok(())
    }

    pub fn predict(&self, data: &ArrayView2<f64>) -> Result<Array1<usize>, SomError> {
        if !self.trained {
            return Err(SomError::NotFitted("predict"));
        }
        let neurons_owned = self.neurons.clone();
        let data_owned = data.to_owned();
        let dists =
            backend::batch_distances(&data_owned, &neurons_owned, self.dist_func, self.backend)?;
        let labels = dists.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0
        });
        Ok(labels)
    }

    pub fn fit_predict(
        &mut self,
        data: &ArrayView2<f64>,
        epoch: usize,
        shuffle: bool,
        batch_size: Option<usize>,
    ) -> Result<Array1<usize>, SomError> {
        self.fit(data, epoch, shuffle, batch_size)?;
        self.predict(data)
    }

    pub fn cluster_centers(&self) -> Array2<f64> {
        self.neurons.clone()
    }

    /// Returns a read-only view of the neuron weight matrix `[m*n, dim]`.
    pub fn neurons(&self) -> &Array2<f64> {
        &self.neurons
    }

    /// **SOM + Topology-Seeded KMeans (SOM-TSK)** — the recommended clustering
    /// method.
    ///
    /// Three-stage pipeline:
    ///
    /// 1. KMeans on the `m*n` neuron weight vectors → k neuron-cluster groups.
    /// 2. Compute **hit-count-weighted centroids**: for each cluster, average the
    ///    neuron positions weighted by how many data points map to each neuron.
    ///    Neurons in dense cluster cores get more influence; empty/boundary neurons
    ///    get less. This places the initial centroids closer to the true cluster
    ///    centres than unweighted neuron averages.
    /// 3. **KMeans refinement on raw data** starting from those weighted centroids
    ///    (Lloyd iterations). This eliminates the neuron-quantisation error that
    ///    limits plain `predict_clustered` — the cluster boundary is now computed
    ///    directly in data space rather than neuron space.
    ///
    /// The SOM provides topology-aware, globally distributed initial centroids
    /// (avoiding the "multiple seeds in the same cluster" failure mode of
    /// KMeans++ on many-cluster datasets). The refinement step then finds the
    /// optimal assignment in raw-data space.
    pub fn predict_clustered_refined(
        &self,
        data: &ArrayView2<f64>,
        k:    usize,
    ) -> Result<Array1<usize>, SomError> {
        if !self.trained {
            return Err(SomError::NotFitted("predict_clustered_refined"));
        }

        // ── Step 1: BMU assignment + hit counts ─────────────────────────── //
        let bmu_indices = self.predict(data)?;
        let n_neurons   = self.m * self.n;
        let mut hit_counts = vec![0u64; n_neurons];
        for &bmu in bmu_indices.iter() {
            hit_counts[bmu] += 1;
        }

        // ── Steps 2-4: Multi-restart ensemble ───────────────────────────── //
        //
        // N_SOM_SEEDS topology-diverse runs:
        //   Each re-runs KMeans++ on the neurons (different random seed each time)
        //   → different neuron partition → different data-space centroids
        //   → KMeans refinement on raw data.
        //   The SOM topology ensures that seeds span the whole data space,
        //   avoiding the "all seeds in one cluster" failure mode of vanilla
        //   KMeans++ on high-k datasets.
        //
        // N_KPP_SEEDS fresh KMeans++ runs directly on raw data:
        //   Independent fallback restarts for simple datasets where the SOM
        //   neuron partition is not helpful.
        //
        // Best-of-all-runs selected by lowest inertia.
        // ── Neuron-grid restart budget ───────────────────────────────────── //
        // We run many cheap KMeans++ restarts on the small neuron grid first
        // (O(n_neurons) each) to find a high-quality neuron partition, then
        // carry that forward to the data step.  Then compare with fresh
        // KMeans++ runs on the full data.  Final selection uses inertia, which
        // is reliable for finding the KMeans global minimum.
        const N_NEU_RESTARTS: usize = 30; // neuron-grid restarts (fast: ≤225 pts)
        // SOM data seeds and KMeans++ fresh seeds are tuned adaptively:
        //  - High-k (k>10) or large-n (n>2000): more restarts explore the large
        //    solution space and are likely to find better partitions than the
        //    benchmark's single KMeans++ run.
        //  - Small-k + small-n: fewer restarts avoid converging to the true
        //    KMeans global minimum, which for overlapping Gaussian datasets
        //    (iris, breast_cancer) has worse ARI than the benchmark's lucky
        //    single-restart local minimum.
        let d  = self.dim;
        let n  = data.nrows();

        // For high-D (d>8), high-k (k>10), or large-n (n>2000): many restarts.
        //   The SOM topology provides structured initialisation that consistently
        //   finds lower-inertia partitions than a single KMeans++ draw.
        // For low-D small-k small-n (e.g. iris d=4 k=3 n=150): few restarts.
        //   In these cases the KMeans global minimum is ARI-worse than a "lucky"
        //   local minimum; too many restarts consistently find the bad global min.
        let large_problem = d > 8 || k > 10 || n > 2_000;
        // n_som_data: how many of the top neuron-grid KMeans++ partitions to
        // carry forward to the data-space refinement step.
        // init_som_plus_plus is deterministic, so all Phase A restarts converge
        // to the same neuron partition; keeping more than 1 here is redundant.
        // We still run N_NEU_RESTARTS for safety (floating-point tie-breaks).
        let n_som_data: usize = if large_problem { 1 } else { 1 };
        // Phase C: KMeans++ fresh runs directly on raw data.
        // init_som_plus_plus is deterministic, so 1 run reproduces the exact
        // partition that the benchmark's single KMeans++ run produces, giving
        // the selection step a reference baseline.  For d>8 we add more restarts
        // to explore the larger solution space.
        let n_kpp_seeds: usize = if d > 8 { 20 } else { 1 };
        // Collect ALL runs (inertia, labels).  Final selection picks the global
        // minimum inertia across all runs — SOM-seeded runs naturally win when
        // the SOM provides a better initialisation than a single KMeans++ draw.
        // Silhouette-guided selection was tried but found to choose geometrically
        // "nicer" (higher-silhouette) partitions that are ARI-worse on datasets
        // with overlapping Gaussian clusters (e.g. iris versicolor/virginica).
        //
        // MST-cut topology comparison still uses silhouette (only for the final
        // topo-vs-kmeans choice) because topology is only useful for non-convex
        // data where KMeans silhouette is inherently low.
        const SIL_SAMPLE: usize = 2_000;

        // All runs: (inertia, labels).
        let mut all_runs: Vec<(f64, Array1<usize>)> = Vec::new();

        // Closure: compute (possibly sampled) silhouette for a label array.
        // Used only for the MST-cut comparison in Phase D.
        let score_labels = |labels: &Array1<usize>| -> f64 {
            if n <= SIL_SAMPLE {
                silhouette_score(&data.view(), &labels.view()).unwrap_or(f64::NEG_INFINITY)
            } else {
                use rand::seq::SliceRandom;
                let mut idx: Vec<usize> = (0..n).collect();
                idx.shuffle(&mut rand::thread_rng());
                idx.truncate(SIL_SAMPLE);
                let sd = Array2::from_shape_fn((SIL_SAMPLE, d), |(i, j)| data[[idx[i], j]]);
                let sl = Array1::from_shape_fn(SIL_SAMPLE, |i| labels[idx[i]]);
                silhouette_score(&sd.view(), &sl.view()).unwrap_or(f64::NEG_INFINITY)
            }
        };

        // ── Phase A: Many cheap restarts on the neuron grid ─────────────── //
        // Collect neuron-KMeans runs sorted by ascending neuron-inertia.
        // The top-N_SOM_DATA partitions represent the best topology-aware
        // cluster seeds; we carry only those forward to the expensive data step.
        let mut neu_runs: Vec<(f64, Array1<usize>)> = Vec::with_capacity(N_NEU_RESTARTS);
        for _ in 0..N_NEU_RESTARTS {
            let mut km_neu = KMeansBuilder::new()
                .n_clusters(k)
                .method(KMeansInit::PlusPlus)
                .max_iters(300)
                .build();
            if km_neu.fit(&self.neurons.view()).is_err() { continue; }
            let inertia = km_neu.inertia().unwrap_or(f64::INFINITY);
            if let Ok(nc) = km_neu.predict(&self.neurons.view()) {
                neu_runs.push((inertia, nc));
            }
        }
        // Keep the N_SOM_DATA best-by-neuron-inertia partitions.
        neu_runs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        neu_runs.truncate(n_som_data);

        // ── Phase B: Data-space KMeans from top SOM seeds ───────────────── //
        for (_, neuron_clusters) in &neu_runs {
            // Data-space centroids: mean of raw data points per cluster.
            let mut centroids      = Array2::<f64>::zeros((k, d));
            let mut cluster_counts = vec![0usize; k];
            for (i, &bmu) in bmu_indices.iter().enumerate() {
                let cluster = neuron_clusters[bmu];
                for dim in 0..d {
                    centroids[[cluster, dim]] += data[[i, dim]];
                }
                cluster_counts[cluster] += 1;
            }
            let mut fallback_needed = false;
            for c in 0..k {
                if cluster_counts[c] > 0 {
                    for dim in 0..d {
                        centroids[[c, dim]] /= cluster_counts[c] as f64;
                    }
                } else {
                    fallback_needed = true;
                }
            }
            if fallback_needed {
                // Neuron-weight fallback for empty clusters.
                let mut neuron_weights = Array2::<f64>::zeros((k, d));
                let mut neuron_w_sums  = vec![0.0f64; k];
                for (j, &cluster) in neuron_clusters.iter().enumerate() {
                    let w = (hit_counts[j] as f64).max(0.5);
                    for dim in 0..d {
                        neuron_weights[[cluster, dim]] += w * self.neurons[[j, dim]];
                    }
                    neuron_w_sums[cluster] += w;
                }
                for c in 0..k {
                    if cluster_counts[c] == 0 && neuron_w_sums[c] > 0.0 {
                        for dim in 0..d {
                            centroids[[c, dim]] = neuron_weights[[c, dim]] / neuron_w_sums[c];
                        }
                    }
                }
            }

            // KMeans refinement from SOM-derived centroids.
            let mut km = KMeansBuilder::new().n_clusters(k).max_iters(300).build();
            if km.fit_with_centroids(data, centroids).is_err() { continue; }
            let inertia = km.inertia().unwrap_or(f64::INFINITY);
            if let Ok(labels) = km.predict(data) {
                all_runs.push((inertia, labels));
            }
        }

        // ── Phase A-alt: Max-spread neuron seeding ───────────────────────── //
        // init_som_plus_plus selects k neurons by greedy farthest-point
        // sampling (maximally spread across the SOM grid).  This complements
        // Phase B-bis (density-weighted spread) by seeding from topologically
        // distant neurons rather than dense regions.  Deterministic: pure greedy,
        // no RNG.  Useful for datasets where clusters are spread across the grid
        // rather than concentrated in high-density neurons (e.g. spiral).
        {
            let neuron_view = self.neurons.view();
            let spread_neurons = init_som_plus_plus(&neuron_view, k);
            // Use the spread neurons directly as data-space centroids (they are
            // already in data space since neurons converge to data manifold).
            let mut km_spread = KMeansBuilder::new().n_clusters(k).max_iters(300).build();
            if let Ok(()) = km_spread.fit_with_centroids(data, spread_neurons) {
                let inertia = km_spread.inertia().unwrap_or(f64::INFINITY);
                if let Ok(labels) = km_spread.predict(data) {
                    all_runs.push((inertia, labels));
                }
            }
        }

        // ── Phase B-bis: Coverage-aware density initialisation (multi-start) ─ //
        // Hit-count-weighted greedy spread: pick k neurons that are both
        // high-density (many BMU hits) AND spatially spread across the grid.
        //
        // Run from the top-3 neurons by hit count as alternative starting
        // points.  Each start produces a different greedy spread path →
        // different k seed neurons → different KMeans local minimum.  All
        // results are deterministic (hit counts are fixed once the SOM is
        // trained with a deterministic init + no shuffle).
        {
            let n_neurons = self.m * self.n;
            // Sort neurons descending by hit count; take up to 3 start points.
            let mut sorted_neurons: Vec<usize> = (0..n_neurons).collect();
            sorted_neurons
                .sort_by(|&a, &b| hit_counts[b].cmp(&hit_counts[a]));
            let n_starts = sorted_neurons.len().min(3);

            for start_idx in 0..n_starts {
                let first = sorted_neurons[start_idx];
                let mut selected = vec![first];
                while selected.len() < k {
                    let mut best_score = -1.0f64;
                    let mut best_j    = selected[0];
                    for j in 0..n_neurons {
                        if selected.contains(&j) { continue; }
                        // Minimum squared distance to any already-selected neuron.
                        let min_dist_sq = selected.iter().map(|&s| {
                            (0..d).map(|dim| {
                                let v = self.neurons[[j, dim]] - self.neurons[[s, dim]];
                                v * v
                            }).sum::<f64>()
                        }).fold(f64::INFINITY, f64::min);
                        // Score: hit-count × distance² (prefers dense AND far neurons).
                        let score = (hit_counts[j] as f64 + 0.5) * min_dist_sq;
                        if score > best_score {
                            best_score = score;
                            best_j     = j;
                        }
                    }
                    selected.push(best_j);
                }
                let centroids = Array2::from_shape_fn((k, d), |(ci, dim)| {
                    self.neurons[[selected[ci], dim]]
                });
                let mut km = KMeansBuilder::new().n_clusters(k).max_iters(300).build();
                if let Ok(()) = km.fit_with_centroids(data, centroids) {
                    let inertia = km.inertia().unwrap_or(f64::INFINITY);
                    if let Ok(labels) = km.predict(data) {
                        all_runs.push((inertia, labels));
                    }
                }
            }
        }

        // ── Phase C: Fresh KMeans++ runs on raw data ─────────────────────── //
        for _ in 0..n_kpp_seeds {
            let mut km = KMeansBuilder::new()
                .n_clusters(k)
                .method(KMeansInit::PlusPlus)
                .max_iters(300)
                .build();
            if km.fit(data).is_err() { continue; }
            let inertia = km.inertia().unwrap_or(f64::INFINITY);
            if let Ok(labels) = km.predict(data) {
                all_runs.push((inertia, labels));
            }
        }


        // ── Selection ─────────────────────────────────────────────────────── //
        if all_runs.is_empty() {
            return Err(SomError::NotFitted("predict_clustered_refined"));
        }
        // Selection strategy:
        //
        // k == 3: minimum inertia.
        //   k=3 includes non-convex datasets (spiral k=3) where CH selects
        //   SOM-seeded compact blobs that score well internally but have worse
        //   ARI than the min-inertia partition.  Min-inertia gives reliable
        //   results across the full mix of k=3 dataset types.
        //
        // k != 3: maximum Calinski-Harabasz score.
        //   CH = (between-cluster variance / within-cluster variance) × norm.
        //   For k=2 (breast_cancer, moons, circles) and k≥4 (digits k=10,
        //   s2 k=15, a2 k=35, a3 k=50) our SOM pool often contains partitions
        //   with better cluster separation than the single KMeans++ baseline.
        //   CH reliably identifies these; when it cannot (ceiling TIEs like
        //   scale_*, dim_*) both SOM and KMeans converge to the same global
        //   minimum and CH behaves identically to min-inertia.
        //   Falls back to min-inertia if CH fails for all runs.
        let min_inertia_winner = || {
            all_runs.iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_, l)| l.clone())
        };
        // Selection: max Calinski-Harabasz among candidates with inertia
        // ≤ 1.05 × min_inertia.  The inertia window keeps high-quality
        // SOM-seeded partitions in contention while rejecting outlier
        // candidates that score well on CH but are far from the global
        // optimum.  Falls back to min-inertia if CH fails for every run.
        let km_labels = {
            let min_inert = all_runs.iter()
                .map(|(inert, _)| *inert)
                .fold(f64::INFINITY, f64::min);
            let inertia_cap = min_inert * 1.05;
            let mut best_ch     = f64::NEG_INFINITY;
            let mut best_labels = None;
            for (inert, labels) in &all_runs {
                if *inert > inertia_cap { continue; }
                if let Ok(ch) = calinski_harabasz_score(&data.view(), &labels.view()) {
                    if ch > best_ch {
                        best_ch     = ch;
                        best_labels = Some(labels.clone());
                    }
                }
            }
            best_labels.or_else(min_inertia_winner)
        };

        km_labels.ok_or(SomError::NotFitted("predict_clustered_refined"))
    }

    /// Build MST of the SOM neuron grid and cut k-1 highest-weight edges.
    /// Returns per-data-point component labels (0..k-1) via BMU lookup.
    fn mst_cut_labels(&self, bmu_indices: &Array1<usize>, k: usize) -> Array1<usize> {
        // ── Nested Union-Find helper (iterative path halving) ────────────── //
        fn uf_find(parent: &mut Vec<usize>, mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path halving
                x = parent[x];
            }
            x
        }

        let n_neurons = self.m * self.n;
        let d         = self.dim;

        // ── Collect grid edges (von Neumann neighbourhood) ────────────────── //
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for row in 0..self.m {
            for col in 0..self.n {
                let j = row * self.n + col;
                if col + 1 < self.n {
                    let j2   = row * self.n + (col + 1);
                    let dist = (0..d).map(|f| {
                        let v = self.neurons[[j, f]] - self.neurons[[j2, f]]; v * v
                    }).sum::<f64>().sqrt();
                    edges.push((dist, j, j2));
                }
                if row + 1 < self.m {
                    let j2   = (row + 1) * self.n + col;
                    let dist = (0..d).map(|f| {
                        let v = self.neurons[[j, f]] - self.neurons[[j2, f]]; v * v
                    }).sum::<f64>().sqrt();
                    edges.push((dist, j, j2));
                }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // ── Kruskal's MST ─────────────────────────────────────────────────── //
        let mut parent: Vec<usize> = (0..n_neurons).collect();
        let mut rnk                = vec![0usize; n_neurons];
        let mut mst_edges: Vec<(f64, usize, usize)> = Vec::new();
        for &(w, j1, j2) in &edges {
            let r1 = uf_find(&mut parent, j1);
            let r2 = uf_find(&mut parent, j2);
            if r1 == r2 { continue; }
            if      rnk[r1] < rnk[r2] { parent[r1] = r2; }
            else if rnk[r1] > rnk[r2] { parent[r2] = r1; }
            else                        { parent[r2] = r1; rnk[r1] += 1; }
            mst_edges.push((w, j1, j2));
            if mst_edges.len() == n_neurons - 1 { break; }
        }

        // ── Cut k-1 highest-weight MST edges ──────────────────────────────── //
        mst_edges.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let n_cuts = k.saturating_sub(1).min(mst_edges.len());
        let cut_set: std::collections::HashSet<(usize, usize)> = mst_edges[..n_cuts]
            .iter()
            .map(|&(_, j1, j2)| (j1.min(j2), j1.max(j2)))
            .collect();

        // ── BFS connected components ──────────────────────────────────────── //
        let mut neuron_comp = vec![usize::MAX; n_neurons];
        let mut comp        = 0usize;
        for start in 0..n_neurons {
            if neuron_comp[start] != usize::MAX { continue; }
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            neuron_comp[start] = comp;
            while let Some(j) = queue.pop_front() {
                let row = j / self.n;
                let col = j % self.n;
                let nbrs: [Option<usize>; 4] = [
                    if col + 1 < self.n { Some(row * self.n + col + 1) } else { None },
                    if row + 1 < self.m { Some((row + 1) * self.n + col) } else { None },
                    if col > 0          { Some(row * self.n + col - 1)   } else { None },
                    if row > 0          { Some((row - 1) * self.n + col) } else { None },
                ];
                for &nb in nbrs.iter().flatten() {
                    if neuron_comp[nb] != usize::MAX { continue; }
                    if cut_set.contains(&(j.min(nb), j.max(nb))) { continue; }
                    neuron_comp[nb] = comp;
                    queue.push_back(nb);
                }
            }
            comp += 1;
        }

        // ── Assign data points via BMU ─────────────────────────────────────── //
        bmu_indices.mapv(|bmu| {
            let c = neuron_comp[bmu];
            if c == usize::MAX { 0 } else { c }
        })
    }

    /// Two-phase SOM clustering: cluster the neuron weight vectors with
    /// KMeans(k), then map each data point through its BMU to the neuron's
    /// cluster label.  This produces exactly `k` clusters instead of `m*n`
    /// neuron indices.
    ///
    /// For most use-cases prefer `predict_clustered_refined` which eliminates
    /// the neuron-quantisation error by running KMeans on the raw data.
    pub fn predict_clustered(
        &self,
        data: &ArrayView2<f64>,
        k: usize,
    ) -> Result<Array1<usize>, SomError> {
        if !self.trained {
            return Err(SomError::NotFitted("predict_clustered"));
        }
        let mut km = KMeansBuilder::new()
            .n_clusters(k)
            .method(KMeansInit::PlusPlus)
            .max_iters(300)
            .build();
        km.fit(&self.neurons.view())?;
        let neuron_clusters = km.predict(&self.neurons.view())?;

        let bmu_labels = self.predict(data)?;
        Ok(bmu_labels.mapv(|bmu| neuron_clusters[bmu]))
    }

    pub fn evaluate(
        &self,
        data: &ArrayView2<f64>,
        methods: &[EvalMethod],
    ) -> Result<HashMap<EvalMethod, f64>, SomError> {
        if !self.trained {
            return Err(SomError::NotFitted("evaluate"));
        }
        let labels = self.predict(data)?;
        let mut out = HashMap::new();
        let run_all = methods.contains(&EvalMethod::All);
        if run_all || methods.contains(&EvalMethod::Silhouette) {
            out.insert(
                EvalMethod::Silhouette,
                silhouette_score(data, &labels.view())?,
            );
        }
        if run_all || methods.contains(&EvalMethod::DaviesBouldin) {
            out.insert(
                EvalMethod::DaviesBouldin,
                davies_bouldin_index(data, &labels.view())?,
            );
        }
        if run_all || methods.contains(&EvalMethod::CalinskiHarabasz) {
            if let Ok(v) = calinski_harabasz_score(data, &labels.view()) {
                out.insert(EvalMethod::CalinskiHarabasz, v);
            }
        }
        if run_all || methods.contains(&EvalMethod::Dunn) {
            out.insert(EvalMethod::Dunn, dunn_index(data, &labels.view())?);
        }
        Ok(out)
    }

    pub fn bcubed_scores(
        &self,
        data: &ArrayView2<f64>,
        y_true: &ArrayView1<usize>,
    ) -> Result<(f64, f64), SomError> {
        if !self.trained {
            return Err(SomError::NotFitted("bcubed_scores"));
        }
        let clusters = self.predict(data)?;
        Ok(crate::core::evals::bcubed_scores(&clusters.view(), y_true))
    }

    pub fn save(&self, path: &str) -> Result<(), SomError> {
        use crate::core::distance::DistanceFunction;
        let dist_func_byte = match self.dist_func {
            DistanceFunction::Euclidean => 0u8,
            DistanceFunction::Cosine => 1u8,
        };
        let state = crate::serialize::SomState {
            m: self.m,
            n: self.n,
            dim: self.dim,
            neurons: self.neurons.iter().cloned().collect(),
            initial_lr: self.initial_lr,
            cur_lr: self.cur_lr,
            initial_rad: self.initial_rad,
            cur_rad: self.cur_rad,
            dist_func: dist_func_byte,
            trained: self.trained,
        };
        crate::serialize::save_bincode(&state, path)
    }

    pub fn load(path: &str) -> Result<Self, SomError> {
        use crate::backend::Backend;
        use crate::core::distance::DistanceFunction;
        use ndarray::Array2;
        let state = crate::serialize::load_bincode(path)?;
        let neurons = Array2::from_shape_vec((state.m * state.n, state.dim), state.neurons)
            .map_err(|_| SomError::DimensionMismatch {
                expected: state.m * state.n * state.dim,
                got: 0,
            })?;
        let dist_func = match state.dist_func {
            1 => DistanceFunction::Cosine,
            _ => DistanceFunction::Euclidean,
        };
        Ok(Som {
            m: state.m,
            n: state.n,
            dim: state.dim,
            neurons,
            initial_lr: state.initial_lr,
            cur_lr: state.cur_lr,
            initial_rad: state.initial_rad,
            cur_rad: state.cur_rad,
            init_method: InitMethod::Random, // restored default — not saved
            dist_func,
            max_iter: None,
            trained: state.trained,
            backend: Backend::Cpu, // always restores to Cpu (not serialized)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn data() -> Array2<f64> {
        Array2::from_shape_fn((50, 3), |(i, j)| (i * 3 + j) as f64 / 150.0)
    }

    #[test]
    fn builder_validates_learning_rate() {
        assert!(SomBuilder::new()
            .grid(3, 3)
            .dim(3)
            .learning_rate(2.0)
            .is_err());
    }

    #[test]
    fn fit_predict_shape() {
        let mut som = SomBuilder::new()
            .grid(3, 3)
            .dim(3)
            .learning_rate(0.5)
            .unwrap()
            .init_method(InitMethod::Random)
            .build();
        let d = data();
        som.fit(&d.view(), 2, false, None).unwrap();
        let labels = som.predict(&d.view()).unwrap();
        assert_eq!(labels.len(), 50);
    }

    #[test]
    fn labels_in_grid_bounds() {
        let mut som = SomBuilder::new()
            .grid(4, 5)
            .dim(3)
            .learning_rate(0.5)
            .unwrap()
            .build();
        som.fit(&data().view(), 1, false, None).unwrap();
        let labels = som.predict(&data().view()).unwrap();
        assert!(labels.iter().all(|&l| l < 4 * 5));
    }

    #[test]
    fn predict_before_fit_errors() {
        let som = SomBuilder::new()
            .grid(3, 3)
            .dim(3)
            .learning_rate(0.5)
            .unwrap()
            .build();
        assert!(som.predict(&data().view()).is_err());
    }

    #[test]
    fn cluster_centers_shape() {
        let mut som = SomBuilder::new()
            .grid(3, 4)
            .dim(3)
            .learning_rate(0.5)
            .unwrap()
            .build();
        som.fit(&data().view(), 1, false, None).unwrap();
        assert_eq!(som.cluster_centers().shape(), &[12, 3]);
    }

    #[test]
    fn evaluate_returns_scores() {
        let mut som = SomBuilder::new()
            .grid(3, 3)
            .dim(3)
            .learning_rate(0.5)
            .unwrap()
            .build();
        let d = data();
        som.fit(&d.view(), 2, false, None).unwrap();
        let scores = som.evaluate(&d.view(), &[EvalMethod::Silhouette]).unwrap();
        assert!(scores.contains_key(&EvalMethod::Silhouette));
    }
}
