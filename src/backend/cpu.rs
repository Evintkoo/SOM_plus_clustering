use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use crate::core::distance::{batch_euclidean, batch_cosine, DistanceFunction};

/// Returns a distance matrix of shape [n_samples, n_neurons].
pub fn batch_distances(
    data: &Array2<f64>,
    neurons: &Array2<f64>,
    dist_fn: DistanceFunction,
) -> Array2<f64> {
    match dist_fn {
        DistanceFunction::Euclidean => batch_euclidean(&data.view(), &neurons.view()),
        DistanceFunction::Cosine => batch_cosine(&data.view(), &neurons.view()),
    }
}

/// Update neurons in-place: neurons[i] += h[i] * (data_point - neurons[i]).
/// neurons: shape [m*n, dim], influence: shape [m, n] (flattened row-major).
pub fn neighborhood_update(
    neurons: &mut Array2<f64>,
    data_point: &ArrayView1<f64>,
    influence: &ArrayView2<f64>,
    dist_fn: DistanceFunction,
) {
    let flat_h: Vec<f64> = influence.iter().cloned().collect();
    neurons
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(flat_h.par_iter())
        .for_each(|(mut neuron, &h)| match dist_fn {
            DistanceFunction::Euclidean => {
                let diff = data_point - &neuron;
                neuron.scaled_add(h, &diff);
            }
            DistanceFunction::Cosine => {
                let nn = neuron.dot(&neuron).sqrt().max(1e-12);
                let xn = data_point.dot(data_point).sqrt().max(1e-12);
                let norm_n = &neuron / nn;
                let norm_x = data_point / xn;
                let dot = norm_n.dot(&norm_x);
                let dir = &norm_n * dot - &norm_n;
                neuron.scaled_add(h * nn, &dir);
            }
        });
}
