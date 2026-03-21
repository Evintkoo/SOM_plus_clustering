#![cfg(feature = "cuda")]

// NOTE: This backend converts f64 inputs to f32 for GPU computation (a ~7-digit
// precision reduction). For distance computation this is generally acceptable.
// For neighborhood_update, accumulated f32→f64 round-trips across training steps
// mean the GPU training path is not numerically identical to the CPU path.
// Precision-sensitive applications should use the CPU backend.

use crate::{core::distance::DistanceFunction, SomError};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use ndarray::{Array2, ArrayView1, ArrayView2};
use once_cell::sync::Lazy;
use std::sync::Arc;

static PTX_EUCLIDEAN: &str = include_str!(concat!(env!("OUT_DIR"), "/euclidean_distances.ptx"));
static PTX_COSINE: &str = include_str!(concat!(env!("OUT_DIR"), "/cosine_distances.ptx"));
static PTX_NEIGHBORHOOD: &str = include_str!(concat!(env!("OUT_DIR"), "/neighborhood_update.ptx"));

static CUDA_DEVICE: Lazy<Result<Arc<CudaDevice>, String>> =
    Lazy::new(|| CudaDevice::new(0).map_err(|e| e.to_string()));

fn get_device() -> Result<Arc<CudaDevice>, SomError> {
    CUDA_DEVICE
        .as_ref()
        .map(|d| d.clone())
        .map_err(|e| SomError::BackendUnavailable(e.clone()))
}

pub fn batch_distances(
    data: &Array2<f64>,
    neurons: &Array2<f64>,
    dist_fn: DistanceFunction,
) -> Result<Array2<f64>, SomError> {
    let dev = get_device()?;
    let n = data.nrows();
    let k = neurons.nrows();
    let dim = data.ncols();

    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let neu_f32: Vec<f32> = neurons.iter().map(|&x| x as f32).collect();

    let d_data = dev
        .htod_sync_copy(&data_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_neurons = dev
        .htod_sync_copy(&neu_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let mut d_out: CudaSlice<f32> = dev
        .alloc_zeros(n * k)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    let (ptx_src, module_name, fn_name) = match dist_fn {
        DistanceFunction::Cosine => (PTX_COSINE, "cosine", "batch_cosine"),
        _ => (PTX_EUCLIDEAN, "euclidean", "batch_euclidean"),
    };

    dev.load_ptx(ptx_src.into(), module_name, &[fn_name])
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let func = dev
        .get_func(module_name, fn_name)
        .ok_or_else(|| SomError::BackendUnavailable("kernel not found".into()))?;

    let threads_x = 16u32;
    let threads_y = 16u32;
    let grid_x = (n as u32 + threads_x - 1) / threads_x;
    let grid_y = (k as u32 + threads_y - 1) / threads_y;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (threads_x, threads_y, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &d_data, &d_neurons, &mut d_out, n as i32, k as i32, dim as i32,
            ),
        )
    }
    .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    let out_f32 = dev
        .dtoh_sync_copy(&d_out)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    Array2::from_shape_vec((n, k), out_f32.iter().map(|&x| x as f64).collect())
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))
}

pub fn neighborhood_update(
    neurons: &mut Array2<f64>,
    data_point: &ArrayView1<f64>,
    influence: &ArrayView2<f64>,
    _dist_fn: DistanceFunction,
) -> Result<(), SomError> {
    let dev = get_device()?;
    let mn = neurons.nrows();
    let dim = neurons.ncols();

    let expected_len = mn;
    let actual_len = influence.nrows() * influence.ncols();
    if actual_len != expected_len {
        return Err(SomError::BackendUnavailable(format!(
            "influence shape mismatch: expected {expected_len}, got {actual_len}"
        )));
    }

    let neu_f32: Vec<f32> = neurons.iter().map(|&x| x as f32).collect();
    let pt_f32: Vec<f32> = data_point.iter().map(|&x| x as f32).collect();
    let inf_f32: Vec<f32> = influence.iter().map(|&x| x as f32).collect();

    let mut d_neurons = dev
        .htod_sync_copy(&neu_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_pt = dev
        .htod_sync_copy(&pt_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_inf = dev
        .htod_sync_copy(&inf_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    dev.load_ptx(
        PTX_NEIGHBORHOOD.into(),
        "neighborhood",
        &["neighborhood_update"],
    )
    .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let func = dev
        .get_func("neighborhood", "neighborhood_update")
        .ok_or_else(|| SomError::BackendUnavailable("kernel not found".into()))?;

    let cfg = LaunchConfig::for_num_elems(mn as u32);
    unsafe { func.launch(cfg, (&mut d_neurons, &d_pt, &d_inf, mn as i32, dim as i32)) }
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    let out = dev
        .dtoh_sync_copy(&d_neurons)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    neurons
        .as_slice_mut()
        .ok_or_else(|| SomError::BackendUnavailable("neurons not contiguous".into()))?
        .iter_mut()
        .zip(out.iter())
        .for_each(|(w, &v)| *w = v as f64);

    Ok(())
}
