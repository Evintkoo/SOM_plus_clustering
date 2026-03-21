use crate::core::distance::{batch_cosine, batch_euclidean, DistanceFunction};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

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
    assert_eq!(
        flat_h.len(),
        neurons.nrows(),
        "influence grid size ({}) must equal neuron count ({})",
        flat_h.len(),
        neurons.nrows()
    );

    // Precompute normalized data point for cosine (avoid recomputing per neuron)
    let xn = data_point.dot(data_point).sqrt().max(1e-12);
    let norm_x = data_point / xn;

    neurons
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(flat_h.par_iter())
        .for_each(|(mut neuron, &h)| match dist_fn {
            DistanceFunction::Euclidean => {
                neuron.zip_mut_with(data_point, |w, &x| *w += h * (x - *w));
            }
            DistanceFunction::Cosine => {
                let nn = neuron.dot(&neuron).sqrt().max(1e-12);
                let norm_n = &neuron / nn;
                let dot = norm_n.dot(&norm_x);
                let dir = &norm_x - &norm_n * dot;
                neuron.scaled_add(h * nn, &dir);
            }
        });
}
