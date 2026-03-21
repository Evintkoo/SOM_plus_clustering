pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

use ndarray::Array2;
use crate::{SomError, core::distance::DistanceFunction};

/// Which compute backend to use for distance and update kernels.
/// NOT serialized — always restores to Cpu on deserialization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
}

pub(crate) fn batch_distances(
    data: &Array2<f64>,
    neurons: &Array2<f64>,
    dist_fn: DistanceFunction,
    backend: Backend,
) -> Result<Array2<f64>, SomError> {
    match backend {
        Backend::Cpu => Ok(cpu::batch_distances(data, neurons, dist_fn)),
        #[cfg(feature = "cuda")]
        Backend::Cuda => cuda::batch_distances(data, neurons, dist_fn),
        #[cfg(feature = "metal")]
        Backend::Metal => metal::batch_distances(data, neurons, dist_fn),
    }
}

pub(crate) fn neighborhood_update(
    neurons: &mut Array2<f64>,
    data_point: &ndarray::ArrayView1<f64>,
    influence: &ndarray::ArrayView2<f64>,
    dist_fn: DistanceFunction,
    backend: Backend,
) -> Result<(), SomError> {
    match backend {
        Backend::Cpu => {
            cpu::neighborhood_update(neurons, data_point, influence, dist_fn);
            Ok(())
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda => cuda::neighborhood_update(neurons, data_point, influence, dist_fn),
        #[cfg(feature = "metal")]
        Backend::Metal => metal::neighborhood_update(neurons, data_point, influence, dist_fn),
    }
}
