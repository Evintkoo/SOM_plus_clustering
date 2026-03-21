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
use rand::seq::SliceRandom;
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
