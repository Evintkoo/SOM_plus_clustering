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
