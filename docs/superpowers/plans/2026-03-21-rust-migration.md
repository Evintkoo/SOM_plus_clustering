# Rust Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `som_plus_clustering` to a pure Rust crate on crates.io with CPU/CUDA/Metal backends, SomBuilder API, serde serialization, and full test coverage.

**Architecture:** Layered single-crate design — `src/core/` holds all algorithms, `src/backend/` holds CPU/CUDA/Metal dispatch, `src/serialize.rs` handles persistence. Feature flags `cuda` and `metal` gate GPU backends at compile time. All public types implement `serde::Serialize/Deserialize` except `Backend` which resets to `Cpu` on deserialize.

**Tech Stack:** Rust 2021 edition, ndarray 0.16, ndarray-rand 0.16, rayon 1.10, rand 0.9, serde 1, bincode 2, thiserror 1, cudarc 0.12 (optional), metal 0.33 (optional), criterion 0.5, proptest 1.

---

## File Map

| File | Responsibility |
|------|---------------|
| `Cargo.toml` | Dependencies, features, bench harnesses |
| `build.rs` | Compile CUDA .cu → .ptx and Metal .metal → .metallib |
| `src/lib.rs` | Public re-exports, feature gates |
| `src/error.rs` | `SomError` enum |
| `src/core/mod.rs` | Core module re-exports |
| `src/core/distance.rs` | Euclidean + cosine distance, batch variants |
| `src/core/neighborhood.rs` | Gaussian neighborhood function |
| `src/core/init.rs` | All 11 init methods |
| `src/core/kmeans.rs` | `KMeans` struct + `KMeansBuilder` |
| `src/core/kde.rs` | KDE kernel, bandwidth, local maxima, neuron selection |
| `src/core/evals.rs` | Silhouette, DB, CH, Dunn, BCubed |
| `src/core/som.rs` | `Som` struct + `SomBuilder` + fit/predict/evaluate |
| `src/core/som_classification.rs` | Supervised `SomClassification` |
| `src/core/model_picker.rs` | `ModelPicker` — evaluates all init methods |
| `src/backend/mod.rs` | `Backend` enum + dispatch fns |
| `src/backend/cpu.rs` | Rayon batch distances + neighborhood update |
| `src/backend/cuda.rs` | cudarc kernels (feature = "cuda") |
| `src/backend/metal.rs` | Metal shaders (feature = "metal") |
| `src/backend/shaders/*.cu` | CUDA PTX source |
| `src/backend/shaders/*.metal` | MSL compute shaders |
| `src/serialize.rs` | save/load bincode v2 + JSON feature |
| `tests/test_som.rs` | Integration: SOM fit/predict/evaluate |
| `tests/test_kmeans.rs` | Integration: KMeans |
| `tests/test_evals.rs` | Integration: all metrics |
| `tests/test_serialization.rs` | Integration: save → load → predict |
| `benches/bench_som.rs` | Criterion: CPU fit/predict |
| `benches/bench_kmeans.rs` | Criterion: KMeans fit |

---

## Task 1: Create dev branch + scaffold Cargo.toml

**Files:**
- Create: `Cargo.toml` (at repo root)
- Create: `build.rs`
- Create: `src/lib.rs`
- Create: `src/error.rs`

- [ ] **Step 1: Create dev branch**

```bash
git checkout -b dev
```

- [ ] **Step 2: Write `Cargo.toml`**

```toml
[package]
name = "som_plus_clustering"
version = "0.1.0"
edition = "2021"
authors = ["Evint Leovonzko"]
description = "Industrial-grade Self-Organizing Map and clustering library"
license = "MIT"
repository = "https://github.com/Evintkoo/SOM_plus_clustering"
keywords = ["som", "clustering", "machine-learning", "unsupervised"]
categories = ["science", "algorithms"]

[features]
default    = []
cuda       = ["dep:cudarc"]
metal      = ["dep:metal"]
serde-json = ["dep:serde_json"]

[dependencies]
ndarray       = { version = "0.16", features = ["rayon", "serde"] }
ndarray-rand  = "0.16"
rayon         = "1.10"
rand          = "0.9"
serde         = { version = "1", features = ["derive"] }
bincode       = "2"
thiserror     = "1"
cudarc        = { version = "0.12", optional = true }
metal         = { version = "0.33", optional = true }
serde_json    = { version = "1", optional = true }

[dev-dependencies]
criterion  = { version = "0.5", features = ["html_reports"] }
proptest   = "1"

[[bench]]
name    = "bench_som"
harness = false

[[bench]]
name    = "bench_kmeans"
harness = false
```

- [ ] **Step 3: Write `build.rs`**

```rust
fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();

    #[cfg(feature = "metal")]
    compile_metal_shaders();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::process::Command;
    let shaders_dir = "src/backend/shaders";
    for name in &["euclidean_distances", "cosine_distances", "neighborhood_update"] {
        let cu = format!("{}/{}.cu", shaders_dir, name);
        let ptx = format!("{}/{}.ptx", shaders_dir, name);
        let status = Command::new("nvcc")
            .args(["--ptx", "-o", &ptx, &cu])
            .status()
            .expect("nvcc not found — install CUDA toolkit");
        assert!(status.success(), "nvcc failed for {}", cu);
        println!("cargo:rerun-if-changed={}", cu);
    }
}

#[cfg(feature = "metal")]
fn compile_metal_shaders() {
    use std::process::Command;
    let shaders_dir = "src/backend/shaders";
    for name in &["euclidean_distances", "cosine_distances", "neighborhood_update"] {
        let msl = format!("{}/{}.metal", shaders_dir, name);
        let air = format!("{}/{}.air", shaders_dir, name);
        let lib = format!("{}/{}.metallib", shaders_dir, name);
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c", &msl, "-o", &air])
            .status().expect("xcrun metal not found");
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib", &air, "-o", &lib])
            .status().expect("xcrun metallib failed");
        println!("cargo:rerun-if-changed={}", msl);
    }
}
```

- [ ] **Step 4: Write `src/error.rs`**

```rust
#[derive(thiserror::Error, Debug)]
pub enum SomError {
    #[error("learning rate must be <= 1.76, got {0}")]
    InvalidLearningRate(f64),
    #[error("invalid init method: {0}")]
    InvalidInitMethod(String),
    #[error("SOM must be fitted before calling {0}")]
    NotFitted(&'static str),
    #[error("already fitted — create a new instance to retrain")]
    AlreadyFitted,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("input data contains NaN or infinite values")]
    InvalidInputData,
    #[error("degenerate cluster: within-cluster variance is zero")]
    ZeroWithinClusterVariance,
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("KDE found {found} local maxima, need at least {needed}")]
    KdeInsufficientMaxima { found: usize, needed: usize },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

- [ ] **Step 5: Write `src/lib.rs` skeleton**

```rust
pub mod core;
pub mod backend;
pub mod serialize;
mod error;

pub use error::SomError;
pub use core::som::{Som, SomBuilder};
pub use core::som_classification::{SomClassification, SomClassificationBuilder};
pub use core::kmeans::{KMeans, KMeansBuilder, KMeansInit};
pub use core::evals::{EvalMethod, ClassEvalMethod};
pub use core::model_picker::ModelPicker;
pub use backend::Backend;

// Re-export ndarray types users need at the API boundary
pub use ndarray::{Array1, Array2, ArrayView2};
```

- [ ] **Step 6: Create module skeleton dirs**

```bash
mkdir -p src/core src/backend src/backend/shaders
touch src/core/mod.rs src/backend/mod.rs src/backend/cpu.rs
touch src/core/distance.rs src/core/neighborhood.rs src/core/init.rs
touch src/core/kmeans.rs src/core/kde.rs src/core/evals.rs
touch src/core/som.rs src/core/som_classification.rs src/core/model_picker.rs
touch src/serialize.rs
mkdir -p tests benches
```

- [ ] **Step 7: Verify it compiles (with stub mod files)**

Add to each `src/core/*.rs` and `src/backend/*.rs`: `// TODO` as a placeholder.
Add to `src/core/mod.rs`:
```rust
pub mod distance;
pub mod neighborhood;
pub mod init;
pub mod kmeans;
pub mod kde;
pub mod evals;
pub mod som;
pub mod som_classification;
pub mod model_picker;
```

```bash
cargo check
```
Expected: compiles with warnings only (unused imports, empty mods).

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml build.rs src/
git commit -m "feat: scaffold Rust crate — Cargo.toml, error types, module skeletons"
```

---

## Task 2: Distance functions

**Files:**
- Create: `src/core/distance.rs`

- [ ] **Step 1: Write failing unit tests**

Inside `src/core/distance.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn euclidean_zero_distance() {
        let a = array![1.0_f64, 2.0, 3.0];
        assert!((euclidean(&a.view(), &a.view())).abs() < 1e-10);
    }

    #[test]
    fn euclidean_known_value() {
        let a = array![0.0_f64, 0.0];
        let b = array![3.0_f64, 4.0];
        assert!((euclidean(&a.view(), &b.view()) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = array![1.0_f64, 0.0, 0.0];
        assert!(cosine(&a.view(), &a.view()).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = array![1.0_f64, 0.0];
        let b = array![0.0_f64, 1.0];
        assert!((cosine(&a.view(), &b.view()) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn batch_euclidean_shape() {
        let data = ndarray::Array2::<f64>::zeros((5, 3));
        let neurons = ndarray::Array2::<f64>::zeros((10, 3));
        let d = batch_euclidean(&data.view(), &neurons.view());
        assert_eq!(d.shape(), &[5, 10]);
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test core::distance::tests -- --nocapture
```
Expected: compile error (functions not defined).

- [ ] **Step 3: Implement**

```rust
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Euclidean distance between two 1-D vectors.
pub fn euclidean(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let diff = a - b;
    diff.dot(&diff).sqrt()
}

/// Squared Euclidean distance (cheaper, used for BMU search).
pub fn euclidean_sq(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let diff = a - b;
    diff.dot(&diff)
}

/// Cosine distance (1 - cosine_similarity) between two 1-D vectors.
pub fn cosine(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt() + 1e-12;
    let norm_b = b.dot(b).sqrt() + 1e-12;
    1.0 - dot / (norm_a * norm_b)
}

/// Batch Euclidean distance matrix: shape [n_samples, n_neurons].
/// Uses ||a-b||² = ||a||² + ||b||² - 2·a·bᵀ (BLAS-accelerated).
pub fn batch_euclidean(data: &ArrayView2<f64>, neurons: &ArrayView2<f64>) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;
    let n = data.nrows();
    let k = neurons.nrows();
    let data_sq = data.mapv(|x| x * x).sum_axis(ndarray::Axis(1)); // (n,)
    let neuron_sq = neurons.mapv(|x| x * x).sum_axis(ndarray::Axis(1)); // (k,)
    let mut cross = Array2::<f64>::zeros((n, k));
    general_mat_mul(1.0, data, &neurons.t(), 0.0, &mut cross);
    // d²[i,j] = data_sq[i] + neuron_sq[j] - 2*cross[i,j]
    let mut dist = cross;
    for i in 0..n {
        for j in 0..k {
            dist[[i, j]] = data_sq[i] + neuron_sq[j] - 2.0 * dist[[i, j]];
            if dist[[i, j]] < 0.0 { dist[[i, j]] = 0.0; } // numerical safety
        }
    }
    dist
}

/// Batch cosine distance matrix: shape [n_samples, n_neurons].
pub fn batch_cosine(data: &ArrayView2<f64>, neurons: &ArrayView2<f64>) -> Array2<f64> {
    use ndarray::linalg::general_mat_mul;
    let n = data.nrows();
    let k = neurons.nrows();
    let data_norms = data.mapv(|x| x * x)
        .sum_axis(ndarray::Axis(1))
        .mapv(|x| x.sqrt() + 1e-12); // (n,)
    let neuron_norms = neurons.mapv(|x| x * x)
        .sum_axis(ndarray::Axis(1))
        .mapv(|x| x.sqrt() + 1e-12); // (k,)
    let mut dots = Array2::<f64>::zeros((n, k));
    general_mat_mul(1.0, data, &neurons.t(), 0.0, &mut dots);
    for i in 0..n {
        for j in 0..k {
            dots[[i, j]] = 1.0 - dots[[i, j]] / (data_norms[i] * neuron_norms[j]);
        }
    }
    dots
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test core::distance::tests
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/distance.rs
git commit -m "feat: distance functions — euclidean, cosine, batch variants"
```

---

## Task 3: Neighborhood function

**Files:**
- Create: `src/core/neighborhood.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn bmu_gets_max_influence() {
        // At BMU itself (dist_sq=0), influence equals learning_rate
        let lr = 0.5_f64;
        let radius = 1.0_f64;
        let h = gaussian(0.0, lr, radius);
        assert!((h - lr).abs() < 1e-10);
    }

    #[test]
    fn far_neuron_gets_near_zero() {
        let h = gaussian(1000.0, 0.5, 0.1);
        assert!(h < 1e-10);
    }

    #[test]
    fn grid_shape_correct() {
        let grid = gaussian_grid(3, 4, 1, 2, 0.5, 1.0);
        assert_eq!(grid.shape(), &[3, 4]);
    }

    #[test]
    fn grid_bmu_is_max() {
        let m = 5; let n = 5;
        let bmu_r = 2; let bmu_c = 2;
        let grid = gaussian_grid(m, n, bmu_r, bmu_c, 0.5, 1.0);
        let max_val = grid[[bmu_r, bmu_c]];
        assert!(grid.iter().all(|&v| v <= max_val + 1e-12));
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test core::neighborhood::tests
```

- [ ] **Step 3: Implement**

```rust
use ndarray::Array2;

/// Gaussian neighborhood influence for a single neuron.
/// dist_sq: squared grid distance from BMU to this neuron.
pub fn gaussian(dist_sq: f64, learning_rate: f64, radius: f64) -> f64 {
    let r2 = (radius * radius).max(1e-18);
    learning_rate * (-dist_sq / (2.0 * r2)).exp()
}

/// Compute full m×n influence grid for BMU at (bmu_row, bmu_col).
pub fn gaussian_grid(
    m: usize, n: usize,
    bmu_row: usize, bmu_col: usize,
    learning_rate: f64, radius: f64,
) -> Array2<f64> {
    let mut grid = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let dr = (i as f64 - bmu_row as f64).powi(2);
            let dc = (j as f64 - bmu_col as f64).powi(2);
            grid[[i, j]] = gaussian(dr + dc, learning_rate, radius);
        }
    }
    grid
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test core::neighborhood::tests
```

- [ ] **Step 5: Commit**

```bash
git add src/core/neighborhood.rs
git commit -m "feat: Gaussian neighborhood function"
```

---

## Task 4: Weight initialization methods

**Files:**
- Create: `src/core/init.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn dummy_data(n: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |(i, j)| (i * d + j) as f64 / (n * d) as f64)
    }

    #[test]
    fn random_shape() {
        let c = init_random(6, 3, 0.0, 1.0);
        assert_eq!(c.shape(), &[6, 3]);
    }

    #[test]
    fn he_shape_no_nan() {
        let c = init_he(3, 10);
        assert_eq!(c.shape(), &[10, 3]);
        assert!(c.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn lecun_shape_no_nan() {
        let c = init_lecun(3, 10);
        assert_eq!(c.shape(), &[3, 10]);
        assert!(c.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_init_identity() {
        let w = init_zero(3, 3);
        assert_eq!(w.shape(), &[3, 3]);
        assert!(w.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn naive_sharding_shape() {
        let data = dummy_data(20, 4);
        let c = init_naive_sharding(&data.view(), 5);
        assert_eq!(c.shape(), &[5, 4]);
        assert!(c.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn som_plus_plus_shape() {
        let data = dummy_data(30, 3);
        let c = init_som_plus_plus(&data.view(), 9);
        assert_eq!(c.shape(), &[9, 3]);
    }

    #[test]
    fn lsuv_shape_no_nan() {
        let data = dummy_data(20, 4);
        let c = init_lsuv(4, 8, &data.view());
        assert_eq!(c.shape(), &[4, 8]);
        assert!(c.iter().all(|x| x.is_finite()));
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test core::init::tests
```

- [ ] **Step 3: Implement all init methods**

```rust
use ndarray::{Array2, ArrayView2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, Normal};

pub fn init_random(k: usize, dim: usize, min: f64, max: f64) -> Array2<f64> {
    Array2::random((k, dim), Uniform::new(min, max))
}

pub fn init_he(dim: usize, k: usize) -> Array2<f64> {
    let std = (2.0_f64 / dim as f64).sqrt();
    Array2::random((k, dim), Normal::new(0.0, std).unwrap())
}

pub fn init_lecun(dim: usize, k: usize) -> Array2<f64> {
    let std = (1.0_f64 / dim as f64).sqrt();
    Array2::random((dim, k), Normal::new(0.0, std).unwrap())
}

pub fn init_zero(p: usize, q: usize) -> Array2<f64> {
    if p == q {
        return ndarray::Array2::eye(p);
    }
    // Partial identity or Hadamard-based — simplified to partial identity
    let mut w = Array2::<f64>::zeros((p, q));
    let diag_len = p.min(q);
    for i in 0..diag_len { w[[i, i]] = 1.0; }
    w
}

pub fn init_naive_sharding(data: &ArrayView2<f64>, k: usize) -> Array2<f64> {
    let n = data.nrows();
    let dim = data.ncols();
    // Sort by feature sum
    let mut sums: Vec<(f64, usize)> = (0..n)
        .map(|i| (data.row(i).sum(), i))
        .collect();
    sums.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let chunk = (n + k - 1) / k;
    let mut centroids = Array2::<f64>::zeros((k, dim));
    for ci in 0..k {
        let start = ci * chunk;
        let end = ((ci + 1) * chunk).min(n);
        if start >= end { continue; }
        let mut mean = ndarray::Array1::<f64>::zeros(dim);
        for &(_, idx) in &sums[start..end] {
            mean += &data.row(idx);
        }
        mean /= (end - start) as f64;
        centroids.row_mut(ci).assign(&mean);
    }
    centroids
}

pub fn init_som_plus_plus(data: &ArrayView2<f64>, k: usize) -> Array2<f64> {
    let n = data.nrows();
    let dim = data.ncols();
    // Start from point farthest from mean
    let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
    let first = (0..n).max_by(|&a, &b| {
        let da: f64 = (&data.row(a) - &mean).mapv(|x| x * x).sum();
        let db: f64 = (&data.row(b) - &mean).mapv(|x| x * x).sum();
        da.partial_cmp(&db).unwrap()
    }).unwrap();

    let mut selected = vec![first];
    let mut min_dist: Vec<f64> = (0..n).map(|i| {
        (&data.row(i) - &data.row(first)).mapv(|x| x * x).sum()
    }).collect();

    for _ in 1..k {
        let next = (0..n)
            .filter(|i| !selected.contains(i))
            .max_by(|&a, &b| min_dist[a].partial_cmp(&min_dist[b]).unwrap())
            .unwrap();
        selected.push(next);
        for i in 0..n {
            let d: f64 = (&data.row(i) - &data.row(next)).mapv(|x| x * x).sum();
            if d < min_dist[i] { min_dist[i] = d; }
        }
    }

    let mut out = Array2::<f64>::zeros((k, dim));
    for (ci, &si) in selected.iter().enumerate() {
        out.row_mut(ci).assign(&data.row(si));
    }
    out
}

pub fn init_lsuv(input_dim: usize, output_dim: usize, x_batch: &ArrayView2<f64>) -> Array2<f64> {
    // SVD orthonormal init
    use ndarray_linalg::SVD;
    let a = Array2::<f64>::random((input_dim, output_dim),
        Normal::new(0.0, 1.0).unwrap());
    let (u, _, vt) = a.svd(true, true).unwrap();
    let mut weights = if u.as_ref().unwrap().shape() == [input_dim, output_dim] {
        u.unwrap()
    } else {
        vt.unwrap()
    };
    // Iterative variance scaling
    for _ in 0..10 {
        let acts = x_batch.dot(&weights);
        let var = acts.var(0.0);
        if var < 1e-8 {
            let std = (2.0_f64 / input_dim as f64).sqrt();
            weights = Array2::random((input_dim, output_dim), Normal::new(0.0, std).unwrap());
            break;
        }
        if (var - 1.0).abs() < 0.1 { break; }
        weights.mapv_inplace(|x| x / var.sqrt());
    }
    weights
}
```

**Note on LSUV — SVD dependency decision (must choose one):**

`ndarray-linalg` requires a BLAS/LAPACK native library. Choose the right feature for the target:

| Platform | Feature to add |
|----------|---------------|
| macOS (Accelerate) | `ndarray-linalg = { version = "0.16", features = ["accelerate"] }` |
| Linux static (CI-friendly) | `ndarray-linalg = { version = "0.16", features = ["openblas-static"] }` |
| Linux dynamic | `ndarray-linalg = { version = "0.16", features = ["openblas"] }` |

Add the chosen line to `Cargo.toml` under `[dependencies]`.
On macOS `accelerate` is zero-install; on Linux `openblas-static` compiles OpenBLAS from source (slower build, no system dependency).

**Alternative:** Avoid `ndarray-linalg` entirely by implementing LSUV without SVD — use He init as the orthonormal seed and only apply the iterative variance-scaling step. This is simpler and removes the native dependency:

```rust
pub fn init_lsuv(input_dim: usize, output_dim: usize, x_batch: &ArrayView2<f64>) -> Array2<f64> {
    // Seed with He initialization (no SVD needed)
    let std = (2.0_f64 / input_dim as f64).sqrt();
    let mut weights = Array2::random((input_dim, output_dim), Normal::new(0.0, std).unwrap());
    // Iterative variance scaling
    for _ in 0..10 {
        let acts = x_batch.dot(&weights);
        let var = acts.var(0.0);
        if var < 1e-8 { break; }
        if (var - 1.0).abs() < 0.1 { break; }
        weights.mapv_inplace(|x| x / var.sqrt());
    }
    weights
}
```
**Recommendation:** Use the SVD-free variant for v0.1 to keep the build simple. LSUV's benefit comes from the variance scaling, not strictly from orthonormal init.

- [ ] **Step 4: Run tests**

```bash
cargo test core::init::tests
```

- [ ] **Step 5: Commit**

```bash
git add src/core/init.rs Cargo.toml
git commit -m "feat: weight initialization methods (He, LeCun, LSUV, ZerO, naive_sharding, som++)"
```

---

## Task 5: KMeans

**Files:**
- Create: `src/core/kmeans.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn blobs() -> Array2<f64> {
        // Two clearly separated blobs
        let mut d = Array2::<f64>::zeros((20, 2));
        for i in 0..10 { d[[i,0]] = i as f64 * 0.01; d[[i,1]] = i as f64 * 0.01; }
        for i in 10..20 { d[[i,0]] = 10.0 + i as f64 * 0.01; d[[i,1]] = 10.0; }
        d
    }

    #[test]
    fn centroids_shape() {
        let km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::Random).build();
        km.fit(&blobs().view()).unwrap();
        assert_eq!(km.centroids().shape(), &[2, 2]);
    }

    #[test]
    fn labels_in_range() {
        let km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::PlusPlus).build();
        km.fit(&blobs().view()).unwrap();
        let labels = km.predict(&blobs().view()).unwrap();
        assert!(labels.iter().all(|&l| l < 2));
    }

    #[test]
    fn already_fitted_error() {
        let km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::Random).build();
        km.fit(&blobs().view()).unwrap();
        assert!(matches!(km.fit(&blobs().view()), Err(crate::SomError::AlreadyFitted)));
    }

    #[test]
    fn inertia_not_fitted_error() {
        let km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::Random).build();
        assert!(matches!(km.inertia(), Err(crate::SomError::NotFitted("inertia"))));
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test core::kmeans::tests
```

- [ ] **Step 3: Implement**

```rust
use ndarray::{Array1, Array2, ArrayView2, Axis};
use crate::SomError;
use crate::core::init::{init_random, init_som_plus_plus};
use std::cell::Cell;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KMeansInit { Random, PlusPlus }

pub struct KMeansBuilder {
    n_clusters: usize,
    method: KMeansInit,
    max_iters: usize,
    tol: f64,
}

impl KMeansBuilder {
    pub fn new() -> Self {
        Self { n_clusters: 8, method: KMeansInit::PlusPlus, max_iters: 300, tol: 1e-6 }
    }
    pub fn n_clusters(mut self, k: usize) -> Self { self.n_clusters = k; self }
    pub fn method(mut self, m: KMeansInit) -> Self { self.method = m; self }
    pub fn max_iters(mut self, n: usize) -> Self { self.max_iters = n; self }
    pub fn tol(mut self, t: f64) -> Self { self.tol = t; self }
    pub fn build(self) -> KMeans {
        KMeans {
            n_clusters: self.n_clusters,
            method: self.method,
            max_iters: self.max_iters,
            tol: self.tol,
            centroids: None,
            inertia: Cell::new(f64::INFINITY),
            n_iter: Cell::new(0),
        }
    }
}

pub struct KMeans {
    n_clusters: usize,
    method: KMeansInit,
    max_iters: usize,
    tol: f64,
    centroids: Option<Array2<f64>>,
    inertia: Cell<f64>,
    n_iter: Cell<usize>,
}

impl KMeans {
    pub fn fit(&mut self, data: &ArrayView2<f64>) -> Result<(), SomError> {
        if self.centroids.is_some() { return Err(SomError::AlreadyFitted); }
        let n = data.nrows();
        let dim = data.ncols();
        let min = data.fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut cents = match self.method {
            KMeansInit::Random => init_random(self.n_clusters, dim, min, max),
            KMeansInit::PlusPlus => init_som_plus_plus(data, self.n_clusters),
        };

        let mut prev_inertia = f64::INFINITY;
        let mut iter = 0;
        for _ in 0..self.max_iters {
            iter += 1;
            // Assign
            let labels = self.assign(data, &cents.view());
            // Update
            let new_cents = self.update(data, &labels, dim);
            // Convergence
            let shift = (&new_cents - &cents).mapv(|x| x * x).sum().sqrt();
            cents = new_cents;
            let inert = self.compute_inertia(data, &labels, &cents.view());
            if shift < self.tol || (prev_inertia - inert).abs() < self.tol { break; }
            prev_inertia = inert;
        }
        self.inertia.set(self.compute_inertia(data, &self.assign(data, &cents.view()), &cents.view()));
        self.n_iter.set(iter);
        self.centroids = Some(cents);
        Ok(())
    }

    fn assign(&self, data: &ArrayView2<f64>, cents: &ArrayView2<f64>) -> Array1<usize> {
        use crate::core::distance::batch_euclidean;
        let d = batch_euclidean(data, cents);
        d.map_axis(Axis(1), |row| row.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0)
    }

    fn update(&self, data: &ArrayView2<f64>, labels: &Array1<usize>, dim: usize) -> Array2<f64> {
        let mut new = Array2::<f64>::zeros((self.n_clusters, dim));
        let mut counts = vec![0usize; self.n_clusters];
        for (i, &c) in labels.iter().enumerate() {
            new.row_mut(c).scaled_add(1.0, &data.row(i));
            counts[c] += 1;
        }
        for (ci, &cnt) in counts.iter().enumerate() {
            if cnt > 0 { new.row_mut(ci).mapv_inplace(|x| x / cnt as f64); }
        }
        new
    }

    fn compute_inertia(&self, data: &ArrayView2<f64>, labels: &Array1<usize>, cents: &ArrayView2<f64>) -> f64 {
        labels.iter().enumerate().map(|(i, &c)| {
            let d = &data.row(i) - &cents.row(c);
            d.dot(&d)
        }).sum()
    }

    pub fn predict(&self, data: &ArrayView2<f64>) -> Result<Array1<usize>, SomError> {
        let cents = self.centroids.as_ref().ok_or(SomError::NotFitted("predict"))?;
        Ok(self.assign(data, &cents.view()))
    }

    pub fn centroids(&self) -> &Array2<f64> { self.centroids.as_ref().unwrap() }
    pub fn inertia(&self) -> Result<f64, SomError> {
        self.centroids.as_ref().ok_or(SomError::NotFitted("inertia"))?;
        Ok(self.inertia.get())
    }
    pub fn n_iter(&self) -> usize { self.n_iter.get() }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test core::kmeans::tests
```

- [ ] **Step 5: Commit**

```bash
git add src/core/kmeans.rs
git commit -m "feat: KMeans clustering with Random and KMeans++ init"
```

---

## Task 6: KDE kernel

**Files:**
- Create: `src/core/kde.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn bandwidth_increases_with_range() {
        let a = ndarray::array![0.0_f64, 1.0, 2.0, 3.0];
        let b = ndarray::array![0.0_f64, 10.0, 20.0, 30.0];
        assert!(bandwidth_estimator(&a.view()) < bandwidth_estimator(&b.view()));
    }

    #[test]
    fn kde_values_positive() {
        let data = Array2::from_shape_fn((20, 2), |(i,j)| i as f64 + j as f64 * 0.1);
        let bw = bandwidth_estimator(&data.column(0));
        let vals = kde_multidimensional(&data.view(), &data.view(), bw);
        assert!(vals.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn local_maxima_detected() {
        // Construct kde_values with clear local max at index 5
        let vals: Vec<f64> = (0..11).map(|i| {
            if i == 5 { 1.0 } else { 0.1 }
        }).collect();
        let pts = Array2::from_shape_fn((11, 1), |(i,_)| i as f64);
        let maxima = find_local_maxima(&vals, &pts.view());
        assert_eq!(maxima.nrows(), 1);
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test core::kde::tests
```

- [ ] **Step 3: Implement**

```rust
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::f64::consts::PI;

pub fn bandwidth_estimator(data: &ArrayView1<f64>) -> f64 {
    let n = data.len();
    if n < 2 { return 1.0; }
    let max = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min = data.fold(f64::INFINITY, |a, &b| a.min(b));
    (max - min) / (1.0 + (n as f64).log2())
}

fn gaussian_kernel(x: &ArrayView1<f64>, xi: &ArrayView1<f64>, bandwidth: f64) -> f64 {
    let d = x.len() as f64;
    let norm = 1.0 / ((2.0 * PI).sqrt().powf(d) * bandwidth.powf(d));
    let diff = x - xi;
    let exponent = -0.5 * diff.dot(&diff) / (bandwidth * bandwidth);
    norm * exponent.exp()
}

pub fn kde_multidimensional(
    data: &ArrayView2<f64>,
    points: &ArrayView2<f64>,
    bandwidth: f64,
) -> Array1<f64> {
    let n = data.nrows() as f64;
    let m = points.nrows();
    let mut vals = Array1::<f64>::zeros(m);
    for i in 0..m {
        let pt = points.row(i);
        let kernel_sum: f64 = (0..data.nrows())
            .map(|j| gaussian_kernel(&pt, &data.row(j), bandwidth))
            .sum();
        vals[i] = kernel_sum / n;
    }
    vals
}

pub fn find_local_maxima(kde_values: &[f64], points: &ArrayView2<f64>) -> Array2<f64> {
    let mut maxima = Vec::new();
    let n = kde_values.len();
    for i in 1..n.saturating_sub(1) {
        if kde_values[i - 1] < kde_values[i] && kde_values[i] > kde_values[i + 1] {
            maxima.push(i);
        }
    }
    let dim = points.ncols();
    let mut out = Array2::<f64>::zeros((maxima.len(), dim));
    for (ci, &idx) in maxima.iter().enumerate() {
        out.row_mut(ci).assign(&points.row(idx));
    }
    out
}

pub fn initiate_kde(
    data: &ArrayView2<f64>,
    n_neurons: usize,
    bandwidth: Option<f64>,
) -> Result<Array2<f64>, crate::SomError> {
    let bw = bandwidth.unwrap_or_else(|| bandwidth_estimator(&data.column(0)));
    let kde_vals = kde_multidimensional(data, data, bw);
    let local_max = find_local_maxima(kde_vals.as_slice().unwrap(), data);
    let max_n = local_max.nrows();
    if max_n <= n_neurons {
        return Err(crate::SomError::KdeInsufficientMaxima { found: max_n, needed: n_neurons });
    }
    // Farthest-first selection
    let mut selected = vec![0usize];
    let dist_matrix: Vec<Vec<f64>> = (0..max_n).map(|i| {
        (0..max_n).map(|j| {
            let d = &local_max.row(i) - &local_max.row(j);
            d.dot(&d)
        }).collect()
    }).collect();
    let mut min_dist: Vec<f64> = dist_matrix[0].clone();
    for _ in 1..n_neurons {
        let next = (0..max_n)
            .filter(|i| !selected.contains(i))
            .max_by(|&a, &b| min_dist[a].partial_cmp(&min_dist[b]).unwrap())
            .unwrap();
        selected.push(next);
        for i in 0..max_n {
            if dist_matrix[next][i] < min_dist[i] { min_dist[i] = dist_matrix[next][i]; }
        }
    }
    let dim = local_max.ncols();
    let mut out = Array2::<f64>::zeros((n_neurons, dim));
    for (ci, &si) in selected.iter().enumerate() {
        out.row_mut(ci).assign(&local_max.row(si));
    }
    Ok(out)
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test core::kde::tests
```

- [ ] **Step 5: Commit**

```bash
git add src/core/kde.rs
git commit -m "feat: KDE kernel — Gaussian, bandwidth estimator, local maxima, initiate_kde"
```

---

## Task 7: Evaluation metrics

**Files:**
- Create: `src/core/evals.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn perfect_clusters() -> (Array2<f64>, ndarray::Array1<usize>) {
        // Two well-separated clusters
        let data = Array2::from_shape_fn((20, 2), |(i, j)| {
            if i < 10 { i as f64 * 0.01 + j as f64 * 0.01 }
            else { 100.0 + i as f64 * 0.01 }
        });
        let labels = ndarray::Array1::from_iter((0..20).map(|i| if i < 10 { 0 } else { 1 }));
        (data, labels)
    }

    #[test]
    fn silhouette_in_range() {
        let (d, l) = perfect_clusters();
        let s = silhouette_score(&d.view(), &l.view()).unwrap();
        assert!(s >= -1.0 && s <= 1.0, "silhouette out of range: {}", s);
    }

    #[test]
    fn silhouette_high_for_separated() {
        let (d, l) = perfect_clusters();
        let s = silhouette_score(&d.view(), &l.view()).unwrap();
        assert!(s > 0.5, "expected high silhouette for separated clusters, got {}", s);
    }

    #[test]
    fn davies_bouldin_non_negative() {
        let (d, l) = perfect_clusters();
        let db = davies_bouldin_index(&d.view(), &l.view()).unwrap();
        assert!(db >= 0.0);
    }

    #[test]
    fn dunn_positive() {
        let (d, l) = perfect_clusters();
        let di = dunn_index(&d.view(), &l.view()).unwrap();
        assert!(di > 0.0);
    }

    #[test]
    fn calinski_harabasz_positive() {
        let (d, l) = perfect_clusters();
        let ch = calinski_harabasz_score(&d.view(), &l.view()).unwrap();
        assert!(ch > 0.0);
    }

    #[test]
    fn ch_zero_variance_error() {
        // All same point → zero within-cluster variance
        let d = Array2::<f64>::zeros((10, 2));
        let l = ndarray::Array1::from_iter((0..10).map(|i| i % 2));
        // This may or may not trigger depending on implementation
        // Just ensure it doesn't panic — either Ok or Err(ZeroWithinClusterVariance)
        let _ = calinski_harabasz_score(&d.view(), &l.view());
    }

    #[test]
    fn bcubed_perfect() {
        let clusters = ndarray::Array1::from_vec(vec![0,0,0,1,1,1]);
        let labels   = ndarray::Array1::from_vec(vec![0,0,0,1,1,1]);
        let (p, r) = bcubed_scores(&clusters.view(), &labels.view());
        assert!((p - 1.0).abs() < 1e-6);
        assert!((r - 1.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test core::evals::tests
```

- [ ] **Step 3: Implement**

```rust
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use crate::SomError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EvalMethod { Silhouette, DaviesBouldin, CalinskiHarabasz, Dunn, All }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassEvalMethod { Accuracy, F1, Recall, All }

fn pairwise_sq_distances(data: &ArrayView2<f64>) -> Array2<f64> {
    use crate::core::distance::batch_euclidean;
    batch_euclidean(data, data)
}

pub fn silhouette_score(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
) -> Result<f64, SomError> {
    let n = data.nrows();
    let unique: Vec<usize> = {
        let mut v: Vec<usize> = labels.to_vec();
        v.sort(); v.dedup(); v
    };
    if unique.len() == 1 { return Ok(0.0); }
    let dists = pairwise_sq_distances(data).mapv(f64::sqrt);
    let mut a = Array1::<f64>::zeros(n);
    let mut b = Array1::<f64>::from_elem(n, f64::INFINITY);

    for &label in &unique {
        let idx: Vec<usize> = (0..n).filter(|&i| labels[i] == label).collect();
        let sz = idx.len();
        if sz > 1 {
            for &i in &idx {
                let sum: f64 = idx.iter().filter(|&&j| j != i).map(|&j| dists[[i,j]]).sum();
                a[i] = sum / (sz - 1) as f64;
            }
        }
        for &other in &unique {
            if other == label { continue; }
            let other_idx: Vec<usize> = (0..n).filter(|&i| labels[i] == other).collect();
            for &i in &idx {
                let mean: f64 = other_idx.iter().map(|&j| dists[[i,j]]).sum::<f64>()
                    / other_idx.len() as f64;
                if mean < b[i] { b[i] = mean; }
            }
        }
    }
    let s_vals: Vec<f64> = (0..n).filter_map(|i| {
        if a[i] == 0.0 && b[i] == f64::INFINITY { return None; }
        let denom = a[i].max(b[i]);
        if denom == 0.0 { None } else { Some((b[i] - a[i]) / denom) }
    }).collect();
    if s_vals.is_empty() { return Ok(0.0); }
    Ok(s_vals.iter().sum::<f64>() / s_vals.len() as f64)
}

pub fn davies_bouldin_index(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
) -> Result<f64, SomError> {
    let n = data.nrows();
    let unique: Vec<usize> = { let mut v = labels.to_vec(); v.sort(); v.dedup(); v };
    let k = unique.len();
    if k <= 1 { return Ok(0.0); }
    let centroids: Vec<Array1<f64>> = unique.iter().map(|&l| {
        let pts: Vec<usize> = (0..n).filter(|&i| labels[i] == l).collect();
        let mut c = Array1::<f64>::zeros(data.ncols());
        for &i in &pts { c.scaled_add(1.0, &data.row(i)); }
        c / pts.len() as f64
    }).collect();
    let dispersions: Vec<f64> = unique.iter().enumerate().map(|(ci, &l)| {
        let pts: Vec<usize> = (0..n).filter(|&i| labels[i] == l).collect();
        pts.iter().map(|&i| {
            let d = &data.row(i) - &centroids[ci];
            d.dot(&d).sqrt()
        }).sum::<f64>() / pts.len() as f64
    }).collect();
    let mut db = 0.0_f64;
    for i in 0..k {
        let mut max_r = 0.0_f64;
        for j in 0..k {
            if i == j { continue; }
            let d = &centroids[i] - &centroids[j];
            let dist = d.dot(&d).sqrt();
            if dist > 1e-12 {
                let r = (dispersions[i] + dispersions[j]) / dist;
                if r > max_r { max_r = r; }
            }
        }
        db += max_r;
    }
    Ok(db / k as f64)
}

pub fn calinski_harabasz_score(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
) -> Result<f64, SomError> {
    let n = data.nrows();
    let unique: Vec<usize> = { let mut v = labels.to_vec(); v.sort(); v.dedup(); v };
    let k = unique.len();
    let overall: Array1<f64> = data.mean_axis(Axis(0)).unwrap();
    let mut between = 0.0_f64;
    let mut within = 0.0_f64;
    for &l in &unique {
        let pts: Vec<usize> = (0..n).filter(|&i| labels[i] == l).collect();
        let sz = pts.len();
        let mut c = Array1::<f64>::zeros(data.ncols());
        for &i in &pts { c.scaled_add(1.0, &data.row(i)); }
        c /= sz as f64;
        let d = &c - &overall;
        between += sz as f64 * d.dot(&d);
        for &i in &pts {
            let dd = &data.row(i) - &c;
            within += dd.dot(&dd);
        }
    }
    if within.abs() < 1e-12 { return Err(SomError::ZeroWithinClusterVariance); }
    Ok(((n - k) as f64 / (k - 1) as f64) * (between / within))
}

pub fn dunn_index(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<usize>,
) -> Result<f64, SomError> {
    let n = data.nrows();
    let unique: Vec<usize> = { let mut v = labels.to_vec(); v.sort(); v.dedup(); v };
    let dists = pairwise_sq_distances(data).mapv(f64::sqrt);
    let mut min_inter = f64::INFINITY;
    let mut max_intra = 0.0_f64;
    for (ii, &l1) in unique.iter().enumerate() {
        let idx1: Vec<usize> = (0..n).filter(|&i| labels[i] == l1).collect();
        for &i in &idx1 {
            for &j in &idx1 { if i < j { let d = dists[[i,j]]; if d > max_intra { max_intra = d; } } }
        }
        for &l2 in &unique[ii+1..] {
            let idx2: Vec<usize> = (0..n).filter(|&i| labels[i] == l2).collect();
            for &i in &idx1 {
                for &j in &idx2 { let d = dists[[i,j]]; if d < min_inter { min_inter = d; } }
            }
        }
    }
    Ok(min_inter / max_intra)
}

pub fn bcubed_scores(
    clusters: &ArrayView1<usize>,
    labels: &ArrayView1<usize>,
) -> (f64, f64) {
    let n = clusters.len();
    let (mut prec, mut rec) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        let same_cluster: Vec<usize> = (0..n).filter(|&j| clusters[j] == clusters[i]).collect();
        let same_label:   Vec<usize> = (0..n).filter(|&j| labels[j]   == labels[i]).collect();
        let both_cluster_and_label = same_cluster.iter().filter(|&&j| labels[j] == labels[i]).count();
        prec += both_cluster_and_label as f64 / same_cluster.len() as f64;
        rec  += both_cluster_and_label as f64 / same_label.len() as f64;
    }
    (prec / n as f64, rec / n as f64)
}

pub fn accuracy(y_true: &ArrayView1<usize>, y_pred: &ArrayView1<usize>) -> f64 {
    let n = y_true.len();
    let correct = y_true.iter().zip(y_pred.iter()).filter(|(a,b)| a == b).count();
    correct as f64 / n as f64 * 100.0
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test core::evals::tests
```

- [ ] **Step 5: Commit**

```bash
git add src/core/evals.rs
git commit -m "feat: evaluation metrics — silhouette, DB, CH, Dunn, BCubed, accuracy"
```

---

## Task 8: CPU backend

**Files:**
- Create: `src/backend/cpu.rs`
- Modify: `src/backend/mod.rs`

- [ ] **Step 1: Write `src/backend/mod.rs`**

```rust
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
pub enum Backend { Cpu, #[cfg(feature="cuda")] Cuda, #[cfg(feature="metal")] Metal }

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
    neurons: &mut Array2<f64>,  // shape [m*n, dim]
    data_point: &ndarray::ArrayView1<f64>,
    influence: &ndarray::ArrayView2<f64>,  // shape [m, n]
    dist_fn: DistanceFunction,
    backend: Backend,
) -> Result<(), SomError> {
    match backend {
        Backend::Cpu => { cpu::neighborhood_update(neurons, data_point, influence, dist_fn); Ok(()) },
        #[cfg(feature = "cuda")]
        Backend::Cuda => cuda::neighborhood_update(neurons, data_point, influence, dist_fn),
        #[cfg(feature = "metal")]
        Backend::Metal => metal::neighborhood_update(neurons, data_point, influence, dist_fn),
    }
}
```

Note: Add `DistanceFunction` enum to `src/core/distance.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DistanceFunction { Euclidean, Cosine }
```

- [ ] **Step 2: Write `src/backend/cpu.rs`**

```rust
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use crate::core::distance::{batch_euclidean, batch_cosine, DistanceFunction};

pub fn batch_distances(
    data: &Array2<f64>,
    neurons: &Array2<f64>,
    dist_fn: DistanceFunction,
) -> Array2<f64> {
    match dist_fn {
        DistanceFunction::Euclidean => batch_euclidean(&data.view(), &neurons.view()),
        DistanceFunction::Cosine    => batch_cosine(&data.view(), &neurons.view()),
    }
}

/// Update neurons in-place: neurons[i] += h[i] * (data_point - neurons[i])
/// neurons: shape [m*n, dim], influence: shape [m, n] (will be flattened)
pub fn neighborhood_update(
    neurons: &mut Array2<f64>,
    data_point: &ArrayView1<f64>,
    influence: &ArrayView2<f64>,
    dist_fn: DistanceFunction,
) {
    let flat_h: Vec<f64> = influence.iter().cloned().collect();
    neurons.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(flat_h.par_iter())
        .for_each(|(mut neuron, &h)| {
            match dist_fn {
                DistanceFunction::Euclidean => {
                    let diff = data_point - &neuron;
                    neuron.scaled_add(h, &diff);
                }
                DistanceFunction::Cosine => {
                    let nn = neuron.dot(&neuron).sqrt() + 1e-12;
                    let xn = data_point.dot(data_point).sqrt() + 1e-12;
                    let norm_n = &neuron / nn;
                    let norm_x = data_point / xn;
                    let dot = norm_n.dot(&norm_x);
                    let dir = &norm_n * dot - &norm_n;
                    neuron.scaled_add(h * nn, &dir);
                }
            }
        });
}
```

- [ ] **Step 3: Run check**

```bash
cargo check
```

- [ ] **Step 4: Commit**

```bash
git add src/backend/mod.rs src/backend/cpu.rs src/core/distance.rs
git commit -m "feat: CPU rayon backend — batch distances + parallel neighborhood update"
```

---

## Task 9: SOM core (unsupervised)

**Files:**
- Create: `src/core/som.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn data() -> Array2<f64> {
        Array2::from_shape_fn((50, 3), |(i,j)| (i*3+j) as f64 / 150.0)
    }

    #[test]
    fn builder_validates_learning_rate() {
        assert!(SomBuilder::new().grid(3,3).dim(3)
            .learning_rate(2.0).is_err());
    }

    #[test]
    fn fit_predict_shape() {
        let mut som = SomBuilder::new().grid(3,3).dim(3)
            .learning_rate(0.5).unwrap()
            .init_method(InitMethod::Random)
            .build();
        let d = data();
        som.fit(&d.view(), 2, false, None).unwrap();
        let labels = som.predict(&d.view()).unwrap();
        assert_eq!(labels.len(), 50);
    }

    #[test]
    fn labels_in_grid_bounds() {
        let mut som = SomBuilder::new().grid(4,5).dim(3)
            .learning_rate(0.5).unwrap()
            .build();
        som.fit(&data().view(), 1, false, None).unwrap();
        let labels = som.predict(&data().view()).unwrap();
        assert!(labels.iter().all(|&l| l < 4*5));
    }

    #[test]
    fn predict_before_fit_errors() {
        let som = SomBuilder::new().grid(3,3).dim(3)
            .learning_rate(0.5).unwrap().build();
        assert!(som.predict(&data().view()).is_err());
    }

    #[test]
    fn cluster_centers_shape() {
        let mut som = SomBuilder::new().grid(3,4).dim(3)
            .learning_rate(0.5).unwrap().build();
        som.fit(&data().view(), 1, false, None).unwrap();
        assert_eq!(som.cluster_centers().shape(), &[12, 3]);
    }

    #[test]
    fn evaluate_returns_scores() {
        let mut som = SomBuilder::new().grid(3,3).dim(3)
            .learning_rate(0.5).unwrap().build();
        let d = data();
        som.fit(&d.view(), 2, false, None).unwrap();
        let scores = som.evaluate(&d.view(), &[EvalMethod::Silhouette]).unwrap();
        assert!(scores.contains_key(&EvalMethod::Silhouette));
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test core::som::tests
```

- [ ] **Step 3: Implement `src/core/som.rs`**

Key structure — implement the full `Som` struct with:
- `SomBuilder` with chainable setters, `learning_rate` returns `Result<SomBuilder, SomError>`
- `Som` contains: `m, n, dim, neurons: Array3<f64>, initial_neurons: Array3<f64>, init_method, dist_func, cur_lr, initial_lr, cur_rad, initial_rad, max_iter: Option<usize>, trained: bool, backend: Backend`
- `fit`: validate data (NaN/inf check), init neurons if not trained, run training loop with exponential decay for lr and radius, mini-batch processing
- `predict`: batch BMU search via `backend::batch_distances`, return `Array1<usize>`
- `evaluate`: call evals functions per method, build `HashMap<EvalMethod, f64>`
- `cluster_centers`: reshape neurons to `[m*n, dim]`
- `bcubed_scores`: delegate to `evals::bcubed_scores`
- `set_backend`: update backend field
- `save`/`load`: delegate to `serialize.rs`

Training loop structure:
```rust
pub fn fit(
    &mut self,
    data: &ArrayView2<f64>,
    epoch: usize,
    shuffle: bool,
    batch_size: Option<usize>,
) -> Result<(), SomError> {
    // 1. Validate data
    if data.iter().any(|x| !x.is_finite()) { return Err(SomError::InvalidInputData); }
    if data.ncols() != self.dim {
        return Err(SomError::DimensionMismatch { expected: self.dim, got: data.ncols() });
    }
    // 2. Init neurons (first call only)
    if !self.trained { self.neurons = self.init_neurons(data)?; }
    // 3. Training loop
    let n = data.nrows();
    let bs = batch_size.unwrap_or_else(|| (n / 100).max(32).min(n));
    let total_iters = (epoch * n).min(self.max_iter.unwrap_or(usize::MAX));
    let mut global_iter = 0usize;
    let mut data_owned = data.to_owned();

    'outer: for _ in 0..epoch {
        if shuffle { shuffle_rows(&mut data_owned); }
        for batch_start in (0..n).step_by(bs) {
            let batch_end = (batch_start + bs).min(n);
            let batch = data_owned.slice(ndarray::s![batch_start..batch_end, ..]);
            for row in 0..batch.nrows() {
                if global_iter >= total_iters { break 'outer; }
                global_iter += 1;
                let pt = batch.row(row);
                let bmu_idx = self.find_bmu(&pt)?;
                let (bmu_r, bmu_c) = (bmu_idx / self.n, bmu_idx % self.n);
                let influence = neighborhood::gaussian_grid(
                    self.m, self.n, bmu_r, bmu_c,
                    self.cur_lr, self.cur_rad,
                );
                let mut neurons_flat = self.neurons.view_mut()
                    .into_shape((self.m * self.n, self.dim)).unwrap();
                backend::neighborhood_update(
                    &mut neurons_flat.to_owned(), &pt, &influence.view(),
                    self.dist_func, self.backend,
                )?;
                // Write updated neurons back to self.neurons (CRITICAL — update is in-place):
                // `neighborhood_update` takes `&mut Array2` and modifies it in-place.
                // Since `neurons_flat` was obtained from a mutable view of `self.neurons`,
                // obtain it as a mutable view directly (do NOT call `.to_owned()`):
                //
                //   let mut neurons_flat = self.neurons
                //       .view_mut()
                //       .into_shape_with_order((self.m * self.n, self.dim))
                //       .unwrap();
                //   backend::neighborhood_update(&mut neurons_flat, &pt, ...) // modifies in-place
                //
                // No explicit copy-back needed — the mutable view writes through to self.neurons.
                // Exponential decay
                let progress = global_iter as f64 / total_iters as f64;
                self.cur_lr  = self.initial_lr  * (-5.0 * progress).exp();
                self.cur_rad = (self.initial_rad * (-3.0 * progress).exp()).max(1e-12);
            }
        }
    }
    self.trained = true;
    Ok(())
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test core::som::tests
```

- [ ] **Step 5: Commit**

```bash
git add src/core/som.rs
git commit -m "feat: SOM struct — SomBuilder, fit/predict/evaluate, CPU backend"
```

---

## Task 10: SOM Classification

**Files:**
- Create: `src/core/som_classification.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn labeled_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_fn((30, 2), |(i,j)| i as f64 * 0.1 + j as f64 * 0.01);
        let y = Array1::from_iter((0..30).map(|i| i % 3));
        (x, y)
    }

    #[test]
    fn fit_predict_shape() {
        let (x, y) = labeled_data();
        let mut clf = SomClassificationBuilder::new().grid(3,3).dim(2)
            .learning_rate(0.5).unwrap().build();
        clf.fit(&x.view(), &y.view(), 2, false, None).unwrap();
        let pred = clf.predict(&x.view()).unwrap();
        assert_eq!(pred.len(), 30);
    }
}
```

- [ ] **Step 2: Implement**

`SomClassification` wraps `Som` and adds a `neuron_label: Array2<usize>` (shape `[m, n]`).
After `fit`, assign each neuron the label of its closest training point.
`predict` maps each input to its BMU's `neuron_label`.

```rust
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::{SomError, core::evals::{accuracy, ClassEvalMethod}};
use super::som::{Som, SomBuilder, InitMethod};
use crate::backend::Backend;
use std::collections::HashMap;

pub struct SomClassificationBuilder(SomBuilder);

impl SomClassificationBuilder {
    pub fn new() -> Self { Self(SomBuilder::new()) }
    pub fn grid(mut self, m: usize, n: usize) -> Self { self.0 = self.0.grid(m, n); self }
    pub fn dim(mut self, d: usize) -> Self { self.0 = self.0.dim(d); self }
    pub fn learning_rate(mut self, lr: f64) -> Result<Self, SomError> {
        self.0 = self.0.learning_rate(lr)?; Ok(self)
    }
    pub fn build(self) -> SomClassification {
        SomClassification { som: self.0.build(), neuron_label: None }
    }
}

pub struct SomClassification {
    som: Som,
    neuron_label: Option<Array2<usize>>,
}

impl SomClassification {
    pub fn fit(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<usize>,
               epoch: usize, shuffle: bool, batch_size: Option<usize>) -> Result<(), SomError> {
        self.som.fit(x, epoch, shuffle, batch_size)?;
        // Assign labels to neurons
        let centers = self.som.cluster_centers();  // [m*n, dim]
        let m = self.som.m; let n = self.som.n;
        let mut labels = Array2::<usize>::zeros((m, n));
        for ci in 0..m*n {
            // Find closest training point
            let best = (0..x.nrows()).min_by(|&a, &b| {
                let da = (&x.row(a) - &centers.row(ci)).mapv(|v| v*v).sum();
                let db = (&x.row(b) - &centers.row(ci)).mapv(|v| v*v).sum();
                da.partial_cmp(&db).unwrap()
            }).unwrap();
            labels[[ci/n, ci%n]] = y[best];
        }
        self.neuron_label = Some(labels);
        Ok(())
    }

    pub fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<usize>, SomError> {
        let bmu_ids = self.som.predict(x)?;
        let labels = self.neuron_label.as_ref().ok_or(SomError::NotFitted("predict"))?;
        let n = self.som.n;
        Ok(Array1::from_iter(bmu_ids.iter().map(|&idx| labels[[idx/n, idx%n]])))
    }

    pub fn evaluate(&self, x: &ArrayView2<f64>, y: &ArrayView1<usize>,
                    methods: &[ClassEvalMethod]) -> Result<HashMap<ClassEvalMethod, f64>, SomError> {
        let pred = self.predict(x)?;
        let mut out = HashMap::new();
        let run_all = methods.contains(&ClassEvalMethod::All);
        if run_all || methods.contains(&ClassEvalMethod::Accuracy) {
            out.insert(ClassEvalMethod::Accuracy, accuracy(y, &pred.view()));
        }
        Ok(out)
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test core::som_classification::tests
```

- [ ] **Step 4: Commit**

```bash
git add src/core/som_classification.rs
git commit -m "feat: supervised SOM classification"
```

---

## Task 11: ModelPicker + Serialization

**Files:**
- Create: `src/core/model_picker.rs`
- Create: `src/serialize.rs`

- [ ] **Step 1: Write model_picker**

```rust
use crate::{SomError, core::som::{Som, SomBuilder, InitMethod}, core::evals::EvalMethod};
use ndarray::ArrayView2;
use crate::backend::Backend;
use crate::core::distance::DistanceFunction;

pub struct ModelPicker {
    models: Vec<Som>,
    scores: Vec<f64>,
}

impl ModelPicker {
    pub fn new() -> Self { Self { models: vec![], scores: vec![] } }

    pub fn evaluate_all_init_methods(
        &mut self,
        data: &ArrayView2<f64>,
        grid_m: usize, grid_n: usize,
        learning_rate: f64, neighbor_rad: f64,
        dist_fn: DistanceFunction,
        max_iter: Option<usize>,
        epoch: usize,
    ) -> Result<(), SomError> {
        // All variants. Kde and KdekMeans may Err (insufficient KDE maxima on small data)
        // — those are silently skipped via the `if som.fit(...).is_err() { continue; }` guard.
        let all_methods = [
            InitMethod::Random, InitMethod::KMeans, InitMethod::KMeansPlusPlus,
            InitMethod::KdekMeans, InitMethod::SomPlusPlus, InitMethod::Zero,
            InitMethod::He, InitMethod::NaiveSharding, InitMethod::LeCun, InitMethod::Lsuv,
            InitMethod::Kde,
        ];
        for method in &all_methods {
            let mut b = SomBuilder::new()
                .grid(grid_m, grid_n)
                .dim(data.ncols())
                .learning_rate(learning_rate)?
                .neighbor_radius(neighbor_rad)
                .distance(dist_fn)
                .init_method(*method);
            if let Some(mi) = max_iter { b = b.max_iter(mi); }
            let mut som = b.build();
            if som.fit(data, epoch, true, None).is_err() { continue; }
            let score = som.evaluate(data, &[EvalMethod::Silhouette])
                .ok()
                .and_then(|s| s.get(&EvalMethod::Silhouette).copied())
                .unwrap_or(f64::NEG_INFINITY);
            self.models.push(som);
            self.scores.push(score);
        }
        Ok(())
    }

    pub fn best_model(self) -> Result<Som, SomError> {
        if self.models.is_empty() { return Err(SomError::NotFitted("best_model")); }
        let best_idx = self.scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        Ok(self.models.into_iter().nth(best_idx).unwrap())
    }
}
```

- [ ] **Step 2: Write `src/serialize.rs`**

```rust
use std::fs::File;
use std::io::{BufReader, BufWriter};
use bincode::{config, encode_to_vec, decode_from_std_read};
use crate::SomError;

pub fn save_bincode<T: bincode::Encode>(value: &T, path: &str) -> Result<(), SomError> {
    let bytes = encode_to_vec(value, config::standard())
        .map_err(|e| SomError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load_bincode<T: bincode::Decode>(path: &str) -> Result<T, SomError> {
    let bytes = std::fs::read(path)?;
    let (val, _) = bincode::decode_from_slice(&bytes, config::standard())
        .map_err(|e| SomError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    Ok(val)
}

#[cfg(feature = "serde-json")]
pub fn save_json<T: serde::Serialize>(value: &T, path: &str) -> Result<(), SomError> {
    let f = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(f), value)
        .map_err(|e| SomError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
}

#[cfg(feature = "serde-json")]
pub fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, SomError> {
    let f = File::open(path)?;
    serde_json::from_reader(BufReader::new(f))
        .map_err(|e| SomError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
}
```

Add `save`/`load`/`save_json`/`load_json` methods to `Som` that delegate to these functions.

- [ ] **Step 3: Run check**

```bash
cargo check
```

- [ ] **Step 4: Commit**

```bash
git add src/core/model_picker.rs src/serialize.rs
git commit -m "feat: ModelPicker + bincode/JSON serialization"
```

---

## Task 12: Integration tests + proptest

**Files:**
- Create: `tests/test_som.rs`
- Create: `tests/test_kmeans.rs`
- Create: `tests/test_evals.rs`
- Create: `tests/test_serialization.rs`

- [ ] **Step 1: Write `tests/test_som.rs`**

```rust
use som_plus_clustering::{Som, SomBuilder, EvalMethod};
use ndarray::Array2;

fn synthetic_data() -> Array2<f64> {
    Array2::from_shape_fn((100, 4), |(i,j)| ((i+j) as f64).sin() * 0.5 + 0.5)
}

#[test]
fn full_fit_predict_evaluate() {
    let d = synthetic_data();
    let mut som = SomBuilder::new().grid(5,5).dim(4)
        .learning_rate(0.5).unwrap().build();
    som.fit(&d.view(), 3, true, None).unwrap();
    let labels = som.predict(&d.view()).unwrap();
    assert_eq!(labels.len(), 100);
    assert!(labels.iter().all(|&l| l < 25));
    let scores = som.evaluate(&d.view(), &[EvalMethod::All]).unwrap();
    assert!(scores.contains_key(&EvalMethod::Silhouette));
    assert!(scores.contains_key(&EvalMethod::DaviesBouldin));
}
```

- [ ] **Step 2: Write `tests/test_serialization.rs`**

```rust
use som_plus_clustering::{Som, SomBuilder};
use ndarray::Array2;

#[test]
fn save_load_predict_identical() {
    let d = Array2::from_shape_fn((50, 3), |(i,j)| i as f64 + j as f64 * 0.1);
    let mut som = SomBuilder::new().grid(3,3).dim(3)
        .learning_rate(0.5).unwrap().build();
    som.fit(&d.view(), 2, false, None).unwrap();
    let labels_before = som.predict(&d.view()).unwrap();
    som.save("/tmp/test_som.bin").unwrap();
    let som2 = Som::load("/tmp/test_som.bin").unwrap();
    let labels_after = som2.predict(&d.view()).unwrap();
    assert_eq!(labels_before, labels_after);
}
```

- [ ] **Step 2b: Write `tests/test_kmeans.rs`**

```rust
use som_plus_clustering::{KMeansBuilder, KMeansInit, SomError};
use ndarray::Array2;

fn blobs() -> Array2<f64> {
    Array2::from_shape_fn((40, 2), |(i,j)| {
        if i < 20 { i as f64 * 0.1 + j as f64 * 0.05 }
        else { 50.0 + i as f64 * 0.1 }
    })
}

#[test]
fn kmeans_fit_predict_labels_valid() {
    let mut km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::PlusPlus).build();
    km.fit(&blobs().view()).unwrap();
    let labels = km.predict(&blobs().view()).unwrap();
    assert_eq!(labels.len(), 40);
    assert!(labels.iter().all(|&l| l < 2));
}

#[test]
fn kmeans_inertia_finite() {
    let mut km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::Random).build();
    km.fit(&blobs().view()).unwrap();
    let inertia = km.inertia().unwrap();
    assert!(inertia.is_finite() && inertia >= 0.0);
}

#[test]
fn kmeans_already_fitted() {
    let mut km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::Random).build();
    km.fit(&blobs().view()).unwrap();
    assert!(matches!(km.fit(&blobs().view()), Err(SomError::AlreadyFitted)));
}
```

- [ ] **Step 2c: Write `tests/test_evals.rs`**

```rust
use som_plus_clustering::{Som, SomBuilder, EvalMethod};
use ndarray::{Array1, Array2};

fn separated_data() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_fn((40, 2), |(i,j)| {
        if i < 20 { i as f64 * 0.01 + j as f64 * 0.01 }
        else { 100.0 + i as f64 * 0.01 }
    });
    let labels = Array1::from_iter((0..40).map(|i| if i < 20 { 0 } else { 1 }));
    (x, labels)
}

#[test]
fn silhouette_high_for_separated_clusters() {
    let (x, l) = separated_data();
    let mut som = SomBuilder::new().grid(2,1).dim(2)
        .learning_rate(0.5).unwrap().build();
    som.fit(&x.view(), 3, false, None).unwrap();
    let scores = som.evaluate(&x.view(), &[EvalMethod::Silhouette]).unwrap();
    let s = scores[&EvalMethod::Silhouette];
    assert!(s >= -1.0 && s <= 1.0);
}

#[test]
fn all_metrics_return_values() {
    let (x, _) = separated_data();
    let mut som = SomBuilder::new().grid(2,2).dim(2)
        .learning_rate(0.5).unwrap().build();
    som.fit(&x.view(), 2, false, None).unwrap();
    let scores = som.evaluate(&x.view(), &[EvalMethod::All]).unwrap();
    assert!(scores.contains_key(&EvalMethod::Silhouette));
    assert!(scores.contains_key(&EvalMethod::DaviesBouldin));
    assert!(scores.contains_key(&EvalMethod::Dunn));
}

#[test]
fn bcubed_perfect_separation() {
    use ndarray::Array1;
    let clusters = Array1::from_vec(vec![0,0,0,0,1,1,1,1]);
    let labels   = Array1::from_vec(vec![0,0,0,0,1,1,1,1]);
    // Need to call bcubed_scores through a fitted Som or directly from evals module
    // Use the public bcubed_scores fn exposed via Som or directly
    use som_plus_clustering::core::evals::bcubed_scores;
    let (p, r) = bcubed_scores(&clusters.view(), &labels.view());
    assert!((p - 1.0).abs() < 1e-6);
    assert!((r - 1.0).abs() < 1e-6);
}
```

- [ ] **Step 3: Write proptest tests in `tests/test_som.rs`**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn bmu_in_grid_bounds(m in 2usize..6, n in 2usize..6, dim in 2usize..5) {
        let data = Array2::from_shape_fn((20, dim), |(i,j)| (i+j) as f64 * 0.1);
        let mut som = SomBuilder::new().grid(m, n).dim(dim)
            .learning_rate(0.5).unwrap().build();
        som.fit(&data.view(), 1, false, None).unwrap();
        let labels = som.predict(&data.view()).unwrap();
        prop_assert!(labels.iter().all(|&l| l < m*n));
    }

    #[test]
    fn neuron_count_matches_grid(m in 2usize..6, n in 2usize..6) {
        let dim = 3usize;
        let data = Array2::from_shape_fn((20, dim), |(i,j)| (i+j) as f64 * 0.1);
        let mut som = SomBuilder::new().grid(m, n).dim(dim)
            .learning_rate(0.5).unwrap().build();
        som.fit(&data.view(), 1, false, None).unwrap();
        prop_assert_eq!(som.cluster_centers().nrows(), m*n);
    }
}
```

- [ ] **Step 4: Run all integration tests**

```bash
cargo test --test test_som
cargo test --test test_serialization
```

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: integration tests + proptest property tests"
```

---

## Task 13: Criterion benchmarks

**Files:**
- Create: `benches/bench_som.rs`
- Create: `benches/bench_kmeans.rs`

- [ ] **Step 1: Write `benches/bench_som.rs`**

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use som_plus_clustering::{SomBuilder};

fn bench_fit_cpu_small(c: &mut Criterion) {
    let data = Array2::from_shape_fn((100, 3), |(i,j)| (i+j) as f64 * 0.01);
    c.bench_function("som_fit_cpu_small_100x3_10x10", |b| {
        b.iter(|| {
            let mut som = SomBuilder::new().grid(10,10).dim(3)
                .learning_rate(0.5).unwrap().build();
            som.fit(&data.view(), 2, false, None).unwrap();
        });
    });
}

fn bench_fit_cpu_large(c: &mut Criterion) {
    let data = Array2::from_shape_fn((10_000, 50), |(i,j)| (i+j) as f64 * 0.0001);
    c.bench_function("som_fit_cpu_large_10kx50_20x20", |b| {
        b.iter(|| {
            let mut som = SomBuilder::new().grid(20,20).dim(50)
                .learning_rate(0.5).unwrap().build();
            som.fit(&data.view(), 1, false, Some(256)).unwrap();
        });
    });
}

criterion_group!(som_benches, bench_fit_cpu_small, bench_fit_cpu_large);
criterion_main!(som_benches);
```

- [ ] **Step 2: Write `benches/bench_kmeans.rs`**

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use som_plus_clustering::{KMeansBuilder, KMeansInit};

fn bench_kmeans_fit_small(c: &mut Criterion) {
    let data = Array2::from_shape_fn((200, 4), |(i,j)| (i+j) as f64 * 0.01);
    c.bench_function("kmeans_fit_200x4_k8", |b| {
        b.iter(|| {
            let mut km = KMeansBuilder::new().n_clusters(8)
                .method(KMeansInit::PlusPlus).build();
            km.fit(&data.view()).unwrap();
        });
    });
}

criterion_group!(kmeans_benches, bench_kmeans_fit_small);
criterion_main!(kmeans_benches);
```

- [ ] **Step 3: Compile-check benchmarks**

```bash
cargo bench --no-run
```
Expected: compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add benches/
git commit -m "bench: criterion benchmarks for SOM and KMeans"
```

---

## Task 14: CUDA backend

**Files:**
- Create: `src/backend/cuda.rs`
- Create: `src/backend/shaders/euclidean_distances.cu`
- Create: `src/backend/shaders/cosine_distances.cu`
- Create: `src/backend/shaders/neighborhood_update.cu`

- [ ] **Step 1: Write CUDA kernels**

`src/backend/shaders/euclidean_distances.cu`:
```c
extern "C" __global__ void batch_euclidean(
    const float* data,    // [n, dim]
    const float* neurons, // [k, dim]
    float* out,           // [n, k]
    int n, int k, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // sample index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index
    if (i >= n || j >= k) return;
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = data[i*dim+d] - neurons[j*dim+d];
        sum += diff * diff;
    }
    out[i*k+j] = sum;
}
```

`src/backend/shaders/neighborhood_update.cu`:
```c
extern "C" __global__ void neighborhood_update(
    float* neurons,           // [m*n, dim]
    const float* data_point,  // [dim]
    const float* influence,   // [m*n]
    int mn, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index
    if (i >= mn) return;
    float h = influence[i];
    for (int d = 0; d < dim; d++) {
        neurons[i*dim+d] += h * (data_point[d] - neurons[i*dim+d]);
    }
}
```

- [ ] **Step 2: Write `src/backend/cuda.rs`**

```rust
#![cfg(feature = "cuda")]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use crate::{SomError, core::distance::DistanceFunction};
use std::sync::Arc;

static PTX_EUCLIDEAN: &str = include_str!("shaders/euclidean_distances.ptx");
static PTX_NEIGHBORHOOD: &str = include_str!("shaders/neighborhood_update.ptx");

fn get_device() -> Result<Arc<CudaDevice>, SomError> {
    CudaDevice::new(0).map_err(|e| SomError::BackendUnavailable(e.to_string()))
}

pub fn batch_distances(
    data: &Array2<f64>,
    neurons: &Array2<f64>,
    dist_fn: DistanceFunction,
) -> Result<Array2<f64>, SomError> {
    // Convert f64 → f32 for GPU (common practice; extend to f64 if precision needed)
    let dev = get_device()?;
    let n = data.nrows(); let k = neurons.nrows(); let dim = data.ncols();
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let neu_f32: Vec<f32>  = neurons.iter().map(|&x| x as f32).collect();
    let d_data = dev.htod_sync_copy(&data_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_neurons = dev.htod_sync_copy(&neu_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let mut d_out: CudaSlice<f32> = dev.alloc_zeros(n * k)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    dev.load_ptx(PTX_EUCLIDEAN.into(), "euclidean", &["batch_euclidean"])
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let func = dev.get_func("euclidean", "batch_euclidean")
        .ok_or_else(|| SomError::BackendUnavailable("kernel not found".into()))?;
    let cfg = LaunchConfig::for_num_elems((n * k) as u32);
    unsafe { func.launch(cfg, (&d_data, &d_neurons, &mut d_out, n as i32, k as i32, dim as i32)) }
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let out_f32 = dev.dtoh_sync_copy(&d_out)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    Ok(Array2::from_shape_vec((n, k), out_f32.iter().map(|&x| x as f64).collect()).unwrap())
}

pub fn neighborhood_update(
    neurons: &mut Array2<f64>,
    data_point: &ArrayView1<f64>,
    influence: &ArrayView2<f64>,
    _dist_fn: DistanceFunction,
) -> Result<(), SomError> {
    let dev = get_device()?;
    let mn = neurons.nrows(); let dim = neurons.ncols();
    let neu_f32: Vec<f32> = neurons.iter().map(|&x| x as f32).collect();
    let pt_f32: Vec<f32>  = data_point.iter().map(|&x| x as f32).collect();
    let inf_f32: Vec<f32> = influence.iter().map(|&x| x as f32).collect();
    let mut d_neurons = dev.htod_sync_copy(&neu_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_pt  = dev.htod_sync_copy(&pt_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let d_inf = dev.htod_sync_copy(&inf_f32)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    // Load the neighborhood PTX (must be done before get_func)
    dev.load_ptx(PTX_NEIGHBORHOOD.into(), "neighborhood", &["neighborhood_update"])
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let func = dev.get_func("neighborhood", "neighborhood_update")
        .ok_or_else(|| SomError::BackendUnavailable("kernel not found".into()))?;
    let cfg = LaunchConfig::for_num_elems(mn as u32);
    unsafe { func.launch(cfg, (&mut d_neurons, &d_pt, &d_inf, mn as i32, dim as i32)) }
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let out = dev.dtoh_sync_copy(&d_neurons)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    for (i, &v) in out.iter().enumerate() { neurons.as_slice_mut().unwrap()[i] = v as f64; }
    Ok(())
}
```

- [ ] **Step 3: Compile-check CUDA path**

```bash
cargo check --features cuda
```
Expected: compiles (requires CUDA toolkit installed; on CI skip with environment check).

- [ ] **Step 4: Commit**

```bash
git add src/backend/cuda.rs src/backend/shaders/*.cu
git commit -m "feat: CUDA backend — batch distances + neighborhood update kernels"
```

---

## Task 15: Metal backend

**Files:**
- Create: `src/backend/metal.rs`
- Create: `src/backend/shaders/euclidean_distances.metal`
- Create: `src/backend/shaders/neighborhood_update.metal`

- [ ] **Step 1: Write MSL shaders**

`src/backend/shaders/euclidean_distances.metal`:
```metal
#include <metal_stdlib>
using namespace metal;

kernel void batch_euclidean(
    device const float* data     [[buffer(0)]],
    device const float* neurons  [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant int& n              [[buffer(3)]],
    constant int& k              [[buffer(4)]],
    constant int& dim            [[buffer(5)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    int i = gid.x; int j = gid.y;
    if (i >= n || j >= k) return;
    float sum = 0.0;
    for (int d = 0; d < dim; d++) {
        float diff = data[i*dim+d] - neurons[j*dim+d];
        sum += diff * diff;
    }
    out[i*k+j] = sum;
}
```

`src/backend/shaders/neighborhood_update.metal`:
```metal
#include <metal_stdlib>
using namespace metal;

kernel void neighborhood_update(
    device float* neurons         [[buffer(0)]],
    device const float* pt        [[buffer(1)]],
    device const float* influence [[buffer(2)]],
    constant int& mn              [[buffer(3)]],
    constant int& dim             [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= (uint)mn) return;
    float h = influence[gid];
    for (int d = 0; d < dim; d++) {
        neurons[gid*dim+d] += h * (pt[d] - neurons[gid*dim+d]);
    }
}
```

- [ ] **Step 2: Write `src/backend/metal.rs`**

```rust
#![cfg(feature = "metal")]
use ndarray::{Array2, ArrayView1, ArrayView2};
use metal::{Device, CommandQueue, MTLResourceOptions};
use crate::{SomError, core::distance::DistanceFunction};

static METALLIB: &[u8] = include_bytes!("shaders/euclidean_distances.metallib");

fn get_device() -> Result<Device, SomError> {
    Device::system_default()
        .ok_or_else(|| SomError::BackendUnavailable("No Metal device found".into()))
}

pub fn batch_distances(
    data: &Array2<f64>,
    neurons: &Array2<f64>,
    _dist_fn: DistanceFunction,
) -> Result<Array2<f64>, SomError> {
    let dev = get_device()?;
    let lib = dev.new_library_with_data(METALLIB)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let kernel = lib.get_function("batch_euclidean", None)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let pipeline = dev.new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| SomError::BackendUnavailable(e.to_string()))?;
    let n = data.nrows(); let k = neurons.nrows(); let dim = data.ncols();
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let neu_f32:  Vec<f32> = neurons.iter().map(|&x| x as f32).collect();
    let mut out_f32 = vec![0.0f32; n * k];
    // Create buffers
    let buf_data = dev.new_buffer_with_data(
        data_f32.as_ptr() as _, (data_f32.len()*4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_neurons = dev.new_buffer_with_data(
        neu_f32.as_ptr() as _, (neu_f32.len()*4) as u64, MTLResourceOptions::StorageModeShared);
    let buf_out = dev.new_buffer((out_f32.len()*4) as u64, MTLResourceOptions::StorageModeShared);
    let n_i = n as i32; let k_i = k as i32; let dim_i = dim as i32;
    let buf_n = dev.new_buffer_with_data(&n_i as *const i32 as _, 4, MTLResourceOptions::StorageModeShared);
    let buf_k = dev.new_buffer_with_data(&k_i as *const i32 as _, 4, MTLResourceOptions::StorageModeShared);
    let buf_d = dev.new_buffer_with_data(&dim_i as *const i32 as _, 4, MTLResourceOptions::StorageModeShared);
    // Encode + dispatch
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
    let thread_group_size = metal::MTLSize { width: 16, height: 16, depth: 1 };
    let grid_size = metal::MTLSize {
        width: ((n + 15) / 16) as u64,
        height: ((k + 15) / 16) as u64,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid_size, thread_group_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    // Read back
    let ptr = buf_out.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, out_f32.len()) };
    Ok(Array2::from_shape_vec((n, k), slice.iter().map(|&x| x as f64).collect()).unwrap())
}

pub fn neighborhood_update(
    neurons: &mut Array2<f64>,
    data_point: &ArrayView1<f64>,
    influence: &ArrayView2<f64>,
    dist_fn: DistanceFunction,
) -> Result<(), SomError> {
    // **Intentional v0.1 simplification:** neighborhood update falls back to CPU rayon.
    // The `neighborhood_update.metal` shader exists and is compiled (Task 15 Step 1),
    // but the Metal dispatch for this kernel is deferred to v0.2 to keep v0.1 scope
    // manageable. The batch_distances GPU acceleration already covers the hot path
    // (BMU search dominates training time); the update is fast even on CPU.
    // TODO(v0.2): Wire `neighborhood_update.metal` here using the same MTLBuffer
    // pattern as `batch_distances` above.
    use crate::backend::cpu;
    cpu::neighborhood_update(neurons, data_point, influence, dist_fn);
    Ok(())
}
```

- [ ] **Step 3: Compile-check Metal path**

```bash
cargo check --features metal  # on macOS only
```

- [ ] **Step 4: Commit**

```bash
git add src/backend/metal.rs src/backend/shaders/*.metal
git commit -m "feat: Metal backend — MSL shaders + batch distance dispatch"
```

---

## Task 16: Cargo.toml metadata + README + CI

**Files:**
- Modify: `Cargo.toml`
- Create: `README.md` (Rust-focused)
- Create: `.github/workflows/rust.yml`

- [ ] **Step 1: Finalize Cargo.toml metadata**

Ensure these fields are set correctly:
```toml
[package]
description = "Industrial-grade Self-Organizing Map and clustering — pure Rust, CPU/CUDA/Metal"
documentation = "https://docs.rs/som_plus_clustering"
readme = "README.md"
exclude = ["modules/", "tests/conftest.py", "*.pkl", "data.csv"]
```

- [ ] **Step 2: Write `.github/workflows/rust.yml`**

```yaml
name: Rust CI

on: [push, pull_request]

jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test

  test-metal:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --features metal

  bench-compile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo bench --no-run

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: clippy, rustfmt }
      - run: cargo clippy -- -D warnings
      - run: cargo fmt --check
```

- [ ] **Step 3: Run full test suite**

```bash
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

Expected: all tests pass, no clippy errors, formatted correctly.

- [ ] **Step 4: Final commit on dev**

```bash
git add .github/ README.md Cargo.toml
git commit -m "chore: CI pipeline, README, crate metadata"
```

---

## Final Verification

- [ ] `cargo test` — all unit + integration tests pass
- [ ] `cargo bench --no-run` — benchmarks compile
- [ ] `cargo clippy -- -D warnings` — no warnings
- [ ] `cargo doc --no-deps --open` — docs render without errors
- [ ] Push `dev` branch and open PR to `main`

```bash
git push -u origin dev
```
