#![cfg(feature = "metal")]
// NOTE: This backend converts f64 inputs to f32 for GPU computation (a ~7-digit
// precision reduction). The batch_distances GPU path is the hot path that dominates
// training time (BMU search). The neighborhood_update falls back to CPU for v0.1.
// The GPU training path is not numerically identical to the CPU path.

use ndarray::{Array2, ArrayView1, ArrayView2};
use metal::{Device, MTLResourceOptions, MTLSize};
use once_cell::sync::Lazy;
use crate::{SomError, core::distance::DistanceFunction};

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/som_kernels.metallib"));

static METAL_DEVICE: Lazy<Result<Device, String>> = Lazy::new(|| {
    Device::system_default()
        .ok_or_else(|| "No Metal device found".to_string())
});

fn get_device() -> Result<&'static Device, SomError> {
    METAL_DEVICE.as_ref()
        .map_err(|e| SomError::BackendUnavailable(e.clone()))
}

pub fn batch_distances(
    data: &Array2<f64>,
    neurons: &Array2<f64>,
    dist_fn: DistanceFunction,
) -> Result<Array2<f64>, SomError> {
    let dev = get_device()?;

    let lib = dev.new_library_with_data(METALLIB)
        .map_err(|e| SomError::BackendUnavailable(e))?;

    let fn_name = match dist_fn {
        DistanceFunction::Cosine => "batch_cosine",
        _ => "batch_euclidean",
    };
    let kernel = lib.get_function(fn_name, None)
        .map_err(|e| SomError::BackendUnavailable(e))?;
    let pipeline = dev.new_compute_pipeline_state_with_function(&*kernel)
        .map_err(|e| SomError::BackendUnavailable(e))?;

    let n = data.nrows();
    let k = neurons.nrows();
    let dim = data.ncols();
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let neu_f32: Vec<f32> = neurons.iter().map(|&x| x as f32).collect();
    let out_len = n * k;

    let buf_data = dev.new_buffer_with_data(
        data_f32.as_ptr() as *const _,
        (data_f32.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_neurons = dev.new_buffer_with_data(
        neu_f32.as_ptr() as *const _,
        (neu_f32.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_out = dev.new_buffer(
        (out_len * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let n_i = n as i32;
    let k_i = k as i32;
    let dim_i = dim as i32;
    let buf_n = dev.new_buffer_with_data(
        &n_i as *const i32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_k = dev.new_buffer_with_data(
        &k_i as *const i32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_d = dev.new_buffer_with_data(
        &dim_i as *const i32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = dev.new_command_queue();
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_data), 0);
    enc.set_buffer(1, Some(&buf_neurons), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_buffer(3, Some(&buf_n), 0);
    enc.set_buffer(4, Some(&buf_k), 0);
    enc.set_buffer(5, Some(&buf_d), 0);

    let thread_group_size = MTLSize { width: 16, height: 16, depth: 1 };
    let grid_size = MTLSize {
        width: (n as u64 + 15) / 16,
        height: (k as u64 + 15) / 16,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid_size, thread_group_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, out_len) };
    Array2::from_shape_vec((n, k), slice.iter().map(|&x| x as f64).collect())
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))
}

pub fn neighborhood_update(
    neurons: &mut Array2<f64>,
    data_point: &ArrayView1<f64>,
    influence: &ArrayView2<f64>,
    dist_fn: DistanceFunction,
) -> Result<(), SomError> {
    // v0.1 intentional simplification: fall back to CPU rayon.
    // batch_distances (BMU search) dominates training time; the update is fast on CPU.
    // TODO(v0.2): Wire neighborhood_update.metal here.
    use crate::backend::cpu;
    cpu::neighborhood_update(neurons, data_point, influence, dist_fn);
    Ok(())
}
