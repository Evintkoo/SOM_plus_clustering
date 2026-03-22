use crate::{
    core::{
        neighborhood,
        som::{InitMethod, Som, SomBuilder},
        distance::DistanceFunction,
    },
    SomError,
};
use ndarray::{Array1, ArrayView2};
use std::collections::VecDeque;

pub struct DenSomResult {
    pub labels:      Array1<i32>,
    pub density:     Array1<f64>,
    pub n_clusters:  usize,
    pub noise_ratio: f64,
}

pub struct DenSomBuilder {
    som_builder:  SomBuilder,
    m:            usize,
    n:            usize,
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

/// Gaussian-smooth the flat BMU hit map over the m×n neuron grid.
/// Uses neighborhood::gaussian(dist_sq, lr=1.0, radius=sigma).
fn smooth_hits(hits: &[usize], m: usize, n: usize, sigma: f64) -> Array1<f64> {
    let sigma = sigma.max(1e-6);
    let mn = m * n;
    let mut out = Array1::<f64>::zeros(mn);
    for i in 0..mn {
        let ri = (i / n) as f64;
        let ci = (i % n) as f64;
        let mut acc = 0.0f64;
        for (j, &hit) in hits.iter().enumerate().take(mn) {
            let rj = (j / n) as f64;
            let cj = (j % n) as f64;
            let dr = ri - rj;
            let dc = ci - cj;
            let dist_sq = dr * dr + dc * dc;
            acc += neighborhood::gaussian(dist_sq, 1.0, sigma) * hit as f64;
        }
        out[i] = acc;
    }
    out
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
    for (t, &h) in hist.iter().enumerate().take(BINS) {
        w0 += h as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }
        sum0 += t as f64 * h as f64;
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

/// BFS connected-components on core neurons with 4-neighbour connectivity.
/// Returns cluster IDs in row-major order; noise neurons receive -1.
fn connected_components(core_mask: &[bool], m: usize, n: usize) -> Array1<i32> {
    let mn = m * n;
    let mut labels = Array1::<i32>::from_elem(mn, -1i32);
    let mut cluster_id = 0i32;

    for start in 0..mn {
        if !core_mask[start] || labels[start] != -1 {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        labels[start] = cluster_id;
        while let Some(cur) = queue.pop_front() {
            let r = cur / n;
            let c = cur % n;
            let neighbours: [Option<usize>; 4] = [
                if r > 0     { Some((r - 1) * n + c) } else { None },
                if r + 1 < m { Some((r + 1) * n + c) } else { None },
                if c > 0     { Some(r * n + c - 1)   } else { None },
                if c + 1 < n { Some(r * n + c + 1)   } else { None },
            ];
            for nb in neighbours.into_iter().flatten() {
                if core_mask[nb] && labels[nb] == -1 {
                    labels[nb] = cluster_id;
                    queue.push_back(nb);
                }
            }
        }
        cluster_id += 1;
    }
    labels
}

/// Topographic watershed cluster extraction.
///
/// **Phase 1 — find density peaks:** a core neuron is a strict local maximum
/// when its smooth_density is strictly greater than that of every core
/// 4-neighbour. These peaks are the cluster seeds. If the density is perfectly
/// flat among all core neurons (no strict maxima), the single highest-density
/// core neuron is used as the unique seed.
///
/// **Phase 2 — BFS flood-fill:** from all seeds simultaneously (highest-density
/// seeds seeded into the queue first), expand to every unassigned core
/// neighbour. First-arrival assigns the cluster. This partitions the core
/// region into exactly as many clusters as there are density peaks — entirely
/// determined by the activation map topology, with no hard-coded cluster count.
fn watershed_components(smooth_density: &[f64], core_mask: &[bool], m: usize, n: usize) -> Array1<i32> {
    let mn = m * n;
    let mut labels = Array1::<i32>::from_elem(mn, -1i32);
    let mut cluster_id = 0i32;

    // Phase 1: collect strict local maxima among core neurons.
    let mut maxima: Vec<(usize, f64)> = Vec::new();
    for i in 0..mn {
        if !core_mask[i] {
            continue;
        }
        let r = i / n;
        let c = i % n;
        let val = smooth_density[i];
        let is_strict_max = [
            if r > 0     { Some((r - 1) * n + c) } else { None },
            if r + 1 < m { Some((r + 1) * n + c) } else { None },
            if c > 0     { Some(r * n + c - 1)   } else { None },
            if c + 1 < n { Some(r * n + c + 1)   } else { None },
        ]
        .into_iter()
        .flatten()
        .filter(|&nb| core_mask[nb])
        .all(|nb| smooth_density[nb] < val);

        if is_strict_max {
            maxima.push((i, val));
        }
    }

    // Fallback: flat core region has no strict maxima — use a single seed at
    // the highest-density core neuron so the whole region forms one cluster.
    if maxima.is_empty() {
        let seed = (0..mn)
            .filter(|&i| core_mask[i])
            .max_by(|&a, &b| {
                smooth_density[a]
                    .partial_cmp(&smooth_density[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        match seed {
            Some(s) => maxima.push((s, smooth_density[s])),
            None    => return labels, // no core neurons at all
        }
    }

    // Phase 2: BFS flood-fill. Seed the queue in descending density order so
    // higher peaks expand first, producing stable basin boundaries.
    maxima.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut queue = VecDeque::new();
    for (i, _) in &maxima {
        labels[*i] = cluster_id;
        queue.push_back(*i);
        cluster_id += 1;
    }

    while let Some(cur) = queue.pop_front() {
        let r = cur / n;
        let c = cur % n;
        let cur_label = labels[cur];
        let neighbours: [Option<usize>; 4] = [
            if r > 0     { Some((r - 1) * n + c) } else { None },
            if r + 1 < m { Some((r + 1) * n + c) } else { None },
            if c > 0     { Some(r * n + c - 1)   } else { None },
            if c + 1 < n { Some(r * n + c + 1)   } else { None },
        ];
        for nb in neighbours.into_iter().flatten() {
            if core_mask[nb] && labels[nb] == -1 {
                labels[nb] = cur_label;
                queue.push_back(nb);
            }
        }
    }
    labels
}

impl Default for DenSomBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DenSomBuilder {
    pub fn new() -> Self {
        Self {
            som_builder: SomBuilder::new(),
            m: 10,
            n: 10,
            smooth_sigma: 1.0,
        }
    }

    pub fn grid(mut self, m: usize, n: usize) -> Self {
        self.som_builder = self.som_builder.grid(m, n);
        self.m = m;
        self.n = n;
        self
    }

    pub fn dim(mut self, d: usize) -> Self {
        self.som_builder = self.som_builder.dim(d);
        self
    }

    /// Returns `Err` if `lr > 1.76`, matching `SomBuilder` validation.
    pub fn learning_rate(mut self, lr: f64) -> Result<Self, SomError> {
        self.som_builder = self.som_builder.learning_rate(lr)?;
        Ok(self)
    }

    pub fn neighbor_radius(mut self, r: f64) -> Self {
        self.som_builder = self.som_builder.neighbor_radius(r);
        self
    }

    pub fn init_method(mut self, m: InitMethod) -> Self {
        self.som_builder = self.som_builder.init_method(m);
        self
    }

    pub fn distance(mut self, d: DistanceFunction) -> Self {
        self.som_builder = self.som_builder.distance(d);
        self
    }

    /// Gaussian smoothing sigma on the neuron grid. Default `1.0`. Clamped to `>= 1e-6`.
    pub fn smooth_sigma(mut self, s: f64) -> Self {
        self.smooth_sigma = s.max(1e-6);
        self
    }

    /// Infallible — all validation happens in individual setter methods.
    pub fn build(self) -> DenSom {
        let mn = self.m * self.n;
        DenSom {
            som:            self.som_builder.build(),
            smooth_sigma:   self.smooth_sigma,
            bmu_hits:       Array1::zeros(mn),
            smooth_density: Array1::zeros(mn),
            cluster_map:    Array1::from_elem(mn, -1i32),
            n_clusters:     0,
            fitted:         false,
        }
    }
}

impl DenSom {
    /// Wrap an already-trained `Som`. Sets `fitted = true` immediately.
    /// Call `refit_density` to build the density map from training data.
    pub fn from_som(som: Som) -> Self {
        let mn = som.m * som.n;
        DenSom {
            som,
            smooth_sigma:   1.0,
            bmu_hits:       Array1::zeros(mn),
            smooth_density: Array1::zeros(mn),
            cluster_map:    Array1::from_elem(mn, -1i32),
            n_clusters:     0,
            fitted:         true,
        }
    }

    fn finalize(&mut self) {
        let m = self.som.m;
        let n = self.som.n;

        // Adaptive sigma: scale the user's smooth_sigma by the grid's geometric
        // mean side length divided by 2π (the bandwidth-resolution trade-off of a
        // Gaussian filter on a discrete lattice). This makes smooth_sigma=1.0 mean
        // "one natural frequency unit of the grid" regardless of grid dimensions,
        // preserving peak separability across grid sizes.
        let effective_sigma = self.smooth_sigma
            * (m as f64 * n as f64).sqrt()
            / (2.0 * std::f64::consts::PI);

        self.smooth_density = smooth_hits(
            self.bmu_hits.as_slice().unwrap(),
            m,
            n,
            effective_sigma,
        );

        let max_d = self.smooth_density.iter().cloned().fold(0.0f64, f64::max);
        if max_d == 0.0 {
            // No data reached any neuron — treat all as one core component.
            let core_mask = vec![true; m * n];
            self.cluster_map = connected_components(&core_mask, m, n);
            self.n_clusters = 1;
            return;
        }

        let vals: Vec<f64> = self.smooth_density.iter().cloned().collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let std_dev = (vals.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / vals.len() as f64)
            .sqrt();

        let core_mask: Vec<bool> = if std_dev < 1e-9 {
            // Flat density — all core; connected_components gives 1 cluster.
            vec![true; m * n]
        } else {
            let tau = otsu(&vals);
            vals.iter().map(|&v| v >= tau).collect()
        };

        // Watershed extracts one cluster per density peak inside the core region,
        // partitioning by basin of attraction. Connected_components is only used
        // for the flat-density edge case where every neuron is equally core.
        self.cluster_map = if std_dev < 1e-9 {
            connected_components(&core_mask, m, n)
        } else {
            watershed_components(&vals, &core_mask, m, n)
        };

        self.n_clusters = {
            let max_id = self.cluster_map.iter().cloned().max().unwrap_or(-1);
            if max_id < 0 { 0 } else { (max_id + 1) as usize }
        };
    }

    pub fn fit(
        &mut self,
        data: &ArrayView2<f64>,
        epoch: usize,
        shuffle: bool,
        batch_size: Option<usize>,
    ) -> Result<(), SomError> {
        if data.nrows() < 2 {
            return Err(SomError::InsufficientData { n: data.nrows() });
        }
        // Reset hit counts for this training run
        self.bmu_hits.fill(0);
        self.som.fit(data, epoch, shuffle, batch_size)?;
        // Count BMU hits from final trained weights
        let bmu_labels = self.som.predict(data)?;
        for &b in bmu_labels.iter() {
            self.bmu_hits[b] += 1;
        }
        self.finalize();
        self.fitted = true;
        Ok(())
    }

    pub fn refit_density(&mut self, data: &ArrayView2<f64>) -> Result<(), SomError> {
        if data.ncols() != self.som.dim {
            return Err(SomError::DimensionMismatch {
                expected: self.som.dim,
                got: data.ncols(),
            });
        }
        if data.nrows() < 2 {
            return Err(SomError::InsufficientData { n: data.nrows() });
        }
        self.bmu_hits.fill(0);
        let bmu_labels = self.som.predict(data)?;
        for &b in bmu_labels.iter() {
            self.bmu_hits[b] += 1;
        }
        self.finalize();
        self.fitted = true;
        Ok(())
    }

    pub fn predict(&self, data: &ArrayView2<f64>) -> Result<DenSomResult, SomError> {
        if !self.fitted {
            return Err(SomError::NotFitted("predict"));
        }
        let bmu_labels = self.som.predict(data)?;
        let n = data.nrows();
        let max_density = self
            .smooth_density
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        let mut labels  = Array1::<i32>::zeros(n);
        let mut density = Array1::<f64>::zeros(n);
        let mut noise_count = 0usize;

        for (i, &b) in bmu_labels.iter().enumerate() {
            labels[i] = self.cluster_map[b];
            density[i] = if max_density > 0.0 {
                self.smooth_density[b] / max_density
            } else {
                0.0
            };
            if labels[i] < 0 {
                noise_count += 1;
            }
        }

        Ok(DenSomResult {
            labels,
            density,
            n_clusters:  self.n_clusters,
            noise_ratio: noise_count as f64 / n as f64,
        })
    }

    pub fn fit_predict(
        &mut self,
        data: &ArrayView2<f64>,
        epoch: usize,
        shuffle: bool,
        batch_size: Option<usize>,
    ) -> Result<DenSomResult, SomError> {
        self.fit(data, epoch, shuffle, batch_size)?;
        self.predict(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn from_som_is_fitted() {
        use crate::core::som::SomBuilder;
        // from_som should set fitted=true immediately
        let som = SomBuilder::new().grid(3, 3).dim(2).build();
        let densom = DenSom::from_som(som);
        assert!(densom.fitted, "from_som must set fitted=true");
        assert_eq!(densom.n_clusters, 0, "no finalize called yet — n_clusters=0");
    }

    #[test]
    fn builder_defaults() {
        let densom = DenSomBuilder::new().grid(4, 4).dim(2).build();
        assert!(!densom.fitted, "new standalone DenSom starts unfitted");
        assert_eq!(densom.smooth_sigma, 1.0);
    }

    #[test]
    fn builder_smooth_sigma_clamped() {
        let densom = DenSomBuilder::new().grid(2, 2).dim(2).smooth_sigma(0.0).build();
        assert!(densom.smooth_sigma >= 1e-6, "sigma must be clamped to at least 1e-6");
    }

    #[test]
    fn otsu_two_class() {
        // Two well-separated classes: 10 values near 0.0, 10 near 1.0
        let mut vals: Vec<f64> = (0..10).map(|i| i as f64 * 0.01).collect();
        vals.extend((0..10).map(|i| 1.0 - i as f64 * 0.01));
        let tau = otsu(&vals);
        assert!(tau > 0.05, "threshold {tau} should be above the low class");
        assert!(tau < 0.95, "threshold {tau} should be below the high class");
    }

    #[test]
    fn smooth_hits_peak_preserved() {
        // 3×3 grid, single hot neuron at centre (index 4)
        let mut hits = [0usize; 9];
        hits[4] = 100;
        let smooth = smooth_hits(&hits, 3, 3, 1.0);
        // Centre must still be the maximum after smoothing
        let max_idx = smooth
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 4, "peak should stay at centre neuron");
    }

    #[test]
    fn connected_components_two_islands() {
        // 5×5 grid: top-left 2×2 block core, bottom-right 2×2 block core, rest noise
        // Neurons: row-major index = r*5 + c
        // Core: (0,0)=0 (0,1)=1 (1,0)=5 (1,1)=6  AND  (3,3)=18 (3,4)=19 (4,3)=23 (4,4)=24
        let mut core = [false; 25];
        for &i in &[0usize, 1, 5, 6, 18, 19, 23, 24] {
            core[i] = true;
        }
        let labels = connected_components(&core, 5, 5);
        // Two distinct non-negative cluster IDs
        let id_a = labels[0];
        let id_b = labels[18];
        assert!(id_a >= 0, "top-left block should be a core cluster");
        assert!(id_b >= 0, "bottom-right block should be a core cluster");
        assert_ne!(id_a, id_b, "two blocks must be different clusters");
        // Noise neurons get -1
        assert_eq!(labels[12], -1, "centre neuron should be noise");
        // Count distinct non-noise cluster IDs
        let mut ids: Vec<i32> = labels.iter().filter(|&&v| v >= 0).cloned().collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 2, "expected exactly 2 clusters, got {}", ids.len());
    }

    #[test]
    fn flat_activation_all_core() {
        // Flat hits (all same value, std ≈ 0) → pre-check fires → all neurons core → n_clusters ≥ 1
        let hits = vec![5usize; 4]; // 2×2 grid, uniform
        let smooth = smooth_hits(&hits, 2, 2, 1.0);
        let vals: Vec<f64> = smooth.iter().cloned().collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let std_dev = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
        assert!(std_dev < 1e-9, "uniform hits should produce near-zero std after smoothing");
        // All neurons should be core (pre-check fires)
        let core_mask = vec![true; 4];
        let labels = connected_components(&core_mask, 2, 2);
        let n_clusters = {
            let max_id = labels.iter().cloned().max().unwrap_or(-1);
            if max_id < 0 { 0 } else { (max_id + 1) as usize }
        };
        assert!(n_clusters >= 1, "flat activation should produce at least 1 cluster");
    }

    #[test]
    fn zero_hits_all_density_zero() {
        // All zeros → max == 0.0 guard → density scores all 0.0
        let hits = vec![0usize; 9];
        let smooth = smooth_hits(&hits, 3, 3, 1.0);
        let max_d = smooth.iter().cloned().fold(0.0f64, f64::max);
        assert_eq!(max_d, 0.0, "all-zero hits → max density must be 0.0");
    }

    #[test]
    fn fit_populates_clusters() {
        // Two tight blobs: 20 pts near (0,0) and 20 pts near (5,5)
        let mut data_vec: Vec<f64> = Vec::new();
        for i in 0..20 {
            data_vec.push(i as f64 * 0.01);
            data_vec.push(i as f64 * 0.01);
        }
        for i in 0..20 {
            data_vec.push(5.0 + i as f64 * 0.01);
            data_vec.push(5.0 + i as f64 * 0.01);
        }
        let data = Array2::from_shape_vec((40, 2), data_vec).unwrap();
        let mut densom = DenSomBuilder::new()
            .grid(5, 5)
            .dim(2)
            .build();
        densom.fit(&data.view(), 5, false, None).unwrap();
        assert!(densom.fitted);
        assert!(densom.n_clusters >= 1, "should find at least 1 cluster");
        assert!(densom.bmu_hits.iter().sum::<usize>() == 40,
                "all 40 points must be accounted for in bmu_hits");
    }

    #[test]
    fn refit_density_dimension_mismatch() {
        use crate::core::som::SomBuilder;
        let som = SomBuilder::new().grid(3, 3).dim(2).build();
        let mut densom = DenSom::from_som(som);
        // data with dim=3 should fail
        let bad_data = Array2::<f64>::zeros((10, 3));
        let result = densom.refit_density(&bad_data.view());
        assert!(matches!(result, Err(SomError::DimensionMismatch { .. })));
    }

    #[test]
    fn fit_insufficient_data_error() {
        let data = Array2::<f64>::zeros((1, 2)); // only 1 point
        let mut densom = DenSomBuilder::new().grid(3, 3).dim(2).build();
        let result = densom.fit(&data.view(), 1, false, None);
        assert!(matches!(result, Err(SomError::InsufficientData { .. })));
    }

    #[test]
    fn predict_returns_correct_fields() {
        let mut data_vec: Vec<f64> = Vec::new();
        for i in 0..30 { data_vec.push(i as f64 * 0.01); data_vec.push(0.0); }
        for i in 0..30 { data_vec.push(10.0 + i as f64 * 0.01); data_vec.push(0.0); }
        let data = Array2::from_shape_vec((60, 2), data_vec).unwrap();
        let mut densom = DenSomBuilder::new().grid(6, 6).dim(2).build();
        let result = densom.fit_predict(&data.view(), 5, false, None).unwrap();
        assert_eq!(result.labels.len(), 60);
        assert_eq!(result.density.len(), 60);
        assert!(result.density.iter().all(|&d| d >= 0.0 && d <= 1.0),
                "all density scores must be in [0,1]");
        assert!(result.noise_ratio >= 0.0 && result.noise_ratio <= 1.0);
    }

    #[test]
    fn predict_not_fitted_error() {
        let densom = DenSomBuilder::new().grid(3, 3).dim(2).build();
        let data = Array2::<f64>::zeros((5, 2));
        let result = densom.predict(&data.view());
        assert!(matches!(result, Err(SomError::NotFitted(_))));
    }

    #[test]
    fn watershed_two_peaks() {
        // 5×5 grid: two isolated density peaks at corners (0,0) and (4,4).
        // All neurons are core (above a floor). Watershed should assign every
        // neuron to one of the two peaks, producing exactly 2 clusters.
        let mn = 25;
        let mut density = vec![0.1f64; mn]; // low background — all core
        density[0]  = 1.0; // peak at (row=0, col=0)
        density[24] = 1.0; // peak at (row=4, col=4)
        let core_mask = vec![true; mn];
        let labels = watershed_components(&density, &core_mask, 5, 5);

        // Both peak neurons must be in different clusters.
        assert!(labels[0]  >= 0, "peak (0,0) should be core");
        assert!(labels[24] >= 0, "peak (4,4) should be core");
        assert_ne!(labels[0], labels[24], "two peaks must be different clusters");

        // Count distinct cluster IDs.
        let mut ids: Vec<i32> = labels.iter().filter(|&&v| v >= 0).cloned().collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 2, "expected exactly 2 watershed clusters, got {}", ids.len());

        // No noise neurons (all core).
        assert!(labels.iter().all(|&v| v >= 0), "all neurons are core — none should be noise");
    }

    #[test]
    fn watershed_single_peak_all_core() {
        // When there is only one density peak, watershed must produce 1 cluster.
        let mn = 9;
        let mut density = vec![0.1f64; mn];
        density[4] = 1.0; // single peak at centre of 3×3
        let core_mask = vec![true; mn];
        let labels = watershed_components(&density, &core_mask, 3, 3);
        let mut ids: Vec<i32> = labels.iter().filter(|&&v| v >= 0).cloned().collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 1, "single peak → exactly 1 cluster, got {}", ids.len());
    }
}
