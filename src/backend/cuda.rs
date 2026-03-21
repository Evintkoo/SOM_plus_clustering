#![cfg(feature = "cuda")]

use ndarray::{Array2, ArrayView1, ArrayView2};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use crate::{SomError, core::distance::DistanceFunction};
use std::sync::Arc;

static PTX_EUCLIDEAN: &str = include_str!("shaders/euclidean_distances.ptx");
static PTX_COSINE: &str = include_str!("shaders/cosine_distances.ptx");
static PTX_NEIGHBORHOOD: &str = include_str!("shaders/neighborhood_update.ptx");

fn get_device() -> Result<Arc<CudaDevice>, SomError> {
    CudaDevice::new(0).map_err(|e| SomError::BackendUnavailable(e.to_string()))
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

    let d_data = dev.htod_sync_copy(&data_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_neurons = dev.htod_sync_copy(&neu_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let mut d_out: CudaSlice<f32> = dev.alloc_zeros(n * k)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    let (ptx_src, module_name, fn_name) = match dist_fn {
        DistanceFunction::Cosine => (PTX_COSINE, "cosine", "batch_cosine"),
        _ => (PTX_EUCLIDEAN, "euclidean", "batch_euclidean"),
    };

    dev.load_ptx(ptx_src.into(), module_name, &[fn_name])
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let func = dev.get_func(module_name, fn_name)
        .ok_or_else(|| SomError::BackendUnavailable("kernel not found".into()))?;

    let cfg = LaunchConfig::for_num_elems((n * k) as u32);
    unsafe {
        func.launch(cfg, (&d_data, &d_neurons, &mut d_out, n as i32, k as i32, dim as i32))
    }
    .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    let out_f32 = dev.dtoh_sync_copy(&d_out)
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

    let neu_f32: Vec<f32> = neurons.iter().map(|&x| x as f32).collect();
    let pt_f32: Vec<f32> = data_point.iter().map(|&x| x as f32).collect();
    let inf_f32: Vec<f32> = influence.iter().map(|&x| x as f32).collect();

    let mut d_neurons = dev.htod_sync_copy(&neu_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_pt = dev.htod_sync_copy(&pt_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_inf = dev.htod_sync_copy(&inf_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    dev.load_ptx(PTX_NEIGHBORHOOD.into(), "neighborhood", &["neighborhood_update"])
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let func = dev.get_func("neighborhood", "neighborhood_update")
        .ok_or_else(|| SomError::BackendUnavailable("kernel not found".into()))?;

    let cfg = LaunchConfig::for_num_elems(mn as u32);
    unsafe {
        func.launch(cfg, (&mut d_neurons, &d_pt, &d_inf, mn as i32, dim as i32))
    }
    .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    let out = dev.dtoh_sync_copy(&d_neurons)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;

    neurons
        .as_slice_mut()
        .ok_or_else(|| SomError::BackendUnavailable("neurons not contiguous".into()))?
        .iter_mut()
        .zip(out.iter())
        .for_each(|(w, &v)| *w = v as f64);

    Ok(())
}
