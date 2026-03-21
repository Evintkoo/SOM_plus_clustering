# Rust Migration Design — som_plus_clustering

**Date:** 2026-03-21
**Author:** Evint Leovonzko
**Status:** Approved (rev 2 — post spec-review fixes)

---

## Overview

Full migration of the `som_plus_clustering` Python library to an industrial-grade pure Rust crate published to `crates.io`. The Rust crate preserves full algorithmic parity with the Python version while delivering native performance, memory safety, and optional GPU acceleration via CUDA and Metal feature flags.

---

## Goals

- Publish `som_plus_clustering` as a pure Rust crate on `crates.io`
- Preserve all algorithms: SOM (unsupervised + classification), KMeans, KDE, all init methods, all eval metrics
- CPU parallelism via `rayon` (always on)
- Optional CUDA backend via `cudarc` (feature flag `cuda`)
- Optional Metal backend via `metal` crate (feature flag `metal`)
- Idiomatic Rust API: `SomBuilder` for construction, sklearn-style `fit`/`predict`/`evaluate` methods
- Model serialization: `bincode` v2 (default) + JSON (feature flag `serde-json`)
- Full test coverage: unit, integration, property-based (`proptest`), criterion benchmarks

---

## Non-Goals

- Python bindings (PyO3/maturin) — deferred to a future release
- WebAssembly target — deferred
- Streaming/online learning — not in scope for this migration
- `compare_distribution` (histogram comparison utility from Python `evals.py`) — intentionally dropped; not used by core SOM/KMeans APIs and can be added as a utility function in a later release

---

## Crate Structure

```
som_plus_clustering/
├── Cargo.toml
├── build.rs                         # compiles CUDA .cu and Metal .metal shaders
├── src/
│   ├── lib.rs                       # public re-exports, feature gates
│   ├── core/
│   │   ├── mod.rs
│   │   ├── som.rs                   # Som struct, SomBuilder, fit/predict/evaluate
│   │   ├── som_classification.rs    # supervised SOM variant
│   │   ├── kmeans.rs                # KMeans + KMeansBuilder
│   │   ├── kde.rs                   # KDE kernel, bandwidth estimator, initiate_kde
│   │   ├── init.rs                  # He, LeCun, LSUV, ZerO, naive_sharding, som++
│   │   ├── distance.rs              # euclidean, cosine (rayon-parallelized)
│   │   ├── neighborhood.rs          # Gaussian neighborhood function
│   │   ├── evals.rs                 # silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn, BCubed
│   │   └── model_picker.rs          # automated init-method selection
│   ├── backend/
│   │   ├── mod.rs                   # Backend enum, dispatch functions
│   │   ├── cpu.rs                   # rayon CPU backend (always compiled)
│   │   ├── cuda.rs                  # cudarc backend (feature = "cuda")
│   │   ├── metal.rs                 # metal backend (feature = "metal")
│   │   └── shaders/
│   │       ├── euclidean_distances.cu
│   │       ├── cosine_distances.cu
│   │       ├── neighborhood_update.cu
│   │       ├── euclidean_distances.metal
│   │       ├── cosine_distances.metal
│   │       └── neighborhood_update.metal
│   └── serialize.rs                 # save/load (bincode v2 + optional JSON)
├── tests/
│   ├── test_som.rs
│   ├── test_kmeans.rs
│   ├── test_evals.rs
│   └── test_serialization.rs
└── benches/
    ├── bench_som.rs
    └── bench_kmeans.rs
```

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ndarray` | 0.16 | N-dimensional arrays (replaces numpy) |
| `ndarray-rand` | 0.16 | Random array generation (must match ndarray minor) |
| `rayon` | 1.10 | CPU data parallelism |
| `rand` | 0.9 | RNG |
| `serde` | 1 | Serialization derive macros |
| `bincode` | 2 | Binary serialization — v2 encode/decode API (default) |
| `thiserror` | 1 | Ergonomic error types |
| `cudarc` | 0.12 | CUDA device management + kernel launch (optional) |
| `metal` | 0.33 | Metal device + compute pipeline (optional) |
| `serde_json` | 1 | JSON serialization (optional) |
| `criterion` | 0.5 | Benchmarks (dev) |
| `proptest` | 1 | Property-based tests (dev) |

**Notes:**
- `ndarray` and `ndarray-rand` must use the same minor version (both `0.16`).
- `bincode` v2 uses a different encode/decode API than v1: use `bincode::encode_to_vec` / `bincode::decode_from_slice` with a `bincode::config::standard()` config object — not the v1 `serialize`/`deserialize` functions.
- `rand` 0.9 has a revised trait API vs 0.8; use `rand::rng()` (not `rand::thread_rng()`).
- `cudarc` 0.12 targets CUDA 12.x. Pin this version; later minor releases have breaking API changes.
- `metal` 0.33 targets macOS 14+. MSL shader compilation via `xcrun -sdk macosx metal` produces a `.metallib` binary embedded with `include_bytes!`.

---

## Feature Flags

```toml
[features]
default    = []
cuda       = ["dep:cudarc"]
metal      = ["dep:metal"]
serde-json = ["dep:serde_json"]
```

Usage:
```toml
# CPU only
som_plus_clustering = "0.1"

# With CUDA
som_plus_clustering = { version = "0.1", features = ["cuda"] }

# With Metal + JSON serialization
som_plus_clustering = { version = "0.1", features = ["metal", "serde-json"] }
```

---

## Error Type

```rust
#[derive(thiserror::Error, Debug)]
pub enum SomError {
    #[error("learning rate must be <= 1.76, got {0}")]
    InvalidLearningRate(f64),
    #[error("invalid init method: {0}")]
    InvalidInitMethod(String),
    #[error("invalid distance function: {0}")]
    InvalidDistanceFunction(String),
    /// method name is always a `'static str` literal (e.g., "predict", "evaluate")
    #[error("SOM must be fitted before calling {0}")]
    NotFitted(&'static str),
    #[error("already fitted — create a new instance to retrain")]
    AlreadyFitted,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("input data contains NaN or infinite values")]
    InvalidInputData,
    #[error("degenerate cluster: within-cluster variance is zero (all points identical?)")]
    ZeroWithinClusterVariance,
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("KDE found {found} local maxima, need at least {needed}")]
    KdeInsufficientMaxima { found: usize, needed: usize },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

---

## Core Enums

```rust
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InitMethod {
    Random, Kde, KMeans, KdekMeans, KMeansPlusPlus,
    SomPlusPlus, Zero, He, NaiveSharding, LeCun, Lsuv,
}
// KdekMeans: runs KDE to find local density maxima, then feeds those
// maxima as the initial centroid candidates into standard KMeans (Lloyd's).
// Equivalent to the Python path: KMeans(method="kde_kmeans") which calls
// initiate_kde → local_maxima → KMeans.fit.

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceFunction { Euclidean, Cosine }

/// Backend is NOT serialized. Deserialized Som always defaults to Backend::Cpu.
/// Callers must re-specify GPU backend via `som.set_backend(Backend::Cuda)` before
/// further training if GPU is desired after a load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")] Cuda,
    #[cfg(feature = "metal")] Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvalMethod {
    Silhouette, DaviesBouldin, CalinskiHarabasz, Dunn, All,
}
// BCubed metrics require ground-truth labels and are NOT part of EvalMethod.
// They are exposed via a separate function: see bcubed_scores() below.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KMeansInit { Random, PlusPlus }
```

---

## Public API

### SOM (Unsupervised)

```rust
// Construction via builder
let som = SomBuilder::new()
    .grid(10, 10)                              // m x n grid
    .dim(3)                                    // input dimensionality
    .init_method(InitMethod::SomPlusPlus)
    .learning_rate(0.5)?                       // Err(SomError::InvalidLearningRate) if > 1.76
    .neighbor_radius(1.0)
    .distance(DistanceFunction::Euclidean)
    .max_iter(1000usize)
    .backend(Backend::Cpu)
    .build()?;                                 // Result<Som, SomError>

// Training — data: Array2<f64> shape [n_samples, n_features]
// batch_size: None = adaptive (max(32, n_samples/100)), Some(k) = fixed mini-batch
som.fit(&data, epoch: usize, shuffle: bool, batch_size: Option<usize>)?;

// Inference — returns Array1<usize> shape [n_samples], values in [0, m*n)
let labels: Array1<usize> = som.predict(&data)?;

// Combined
let labels = som.fit_predict(&data, epoch, shuffle, batch_size)?;

// Unsupervised evaluation — returns HashMap<EvalMethod, f64>
// Does NOT accept BcubedPrecision/BcubedRecall — those require ground truth labels.
// CalinskiHarabasz returns Err(SomError::ZeroWithinClusterVariance) if all points identical.
let scores = som.evaluate(&data, &[EvalMethod::Silhouette, EvalMethod::Dunn])?;
let all_scores = som.evaluate(&data, &[EvalMethod::All])?;

// BCubed scores (requires ground-truth labels)
let (bcubed_prec, bcubed_rec) = som.bcubed_scores(&data, &y_true)?;

// Cluster centers — shape [m*n, dim]
let centers: ArrayView2<f64> = som.cluster_centers();

// Switch backend after load (Backend is not serialized)
som.set_backend(Backend::Cpu);

// Serialization (bincode v2)
som.save("model.bin")?;
let som = Som::load("model.bin")?;             // backend resets to Cpu on load

// JSON (feature = "serde-json")
#[cfg(feature = "serde-json")]
som.save_json("model.json")?;
#[cfg(feature = "serde-json")]
let som = Som::load_json("model.json")?;
```

**Thread safety:** `Som` is `Send`. When the `metal` feature is active, `Som` is `Send` but NOT `Sync` (Metal's `Device` handle is not `Sync`). CPU and CUDA paths are both `Send + Sync`. Document this in rustdoc.

### SOM (Supervised Classification)

```rust
// fit takes features x: Array2<f64> and labels y: Array1<usize>
som_clf.fit(&x, &y, epoch: usize, shuffle: bool, batch_size: Option<usize>)?;

// predict returns predicted class labels Array1<usize>
let y_pred = som_clf.predict(&x)?;

// evaluate with classification metrics — returns HashMap<ClassEvalMethod, f64>
let scores = som_clf.evaluate(&x, &y, &[ClassEvalMethod::Accuracy, ClassEvalMethod::F1])?;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassEvalMethod { Accuracy, F1, Recall, All }
```

### KMeans

```rust
let km = KMeansBuilder::new()
    .n_clusters(10)
    .method(KMeansInit::PlusPlus)  // or KMeansInit::Random
    .max_iters(300)
    .tol(1e-6)
    .build();

km.fit(&data)?;                              // Err(SomError::AlreadyFitted) if called twice
let labels = km.predict(&data)?;
let inertia: f64 = km.inertia()?;           // Result<f64, SomError> — Err(NotFitted) if not trained
let n_iter: usize = km.n_iter();
```

### ModelPicker

```rust
// Trains a Som with every InitMethod variant, evaluates each by silhouette score,
// returns the best-performing Som. Evaluates all InitMethod variants automatically.
let picker = ModelPicker::new();
picker.evaluate_all_init_methods(
    &data,
    grid_m: usize, grid_n: usize,
    learning_rate: f64, neighbor_rad: f64,
    dist_fn: DistanceFunction,
    max_iter: Option<usize>,
    epoch: usize,
)?;
// Returns the best Som (highest silhouette score), owned
let best: Som = picker.best_model()?;  // Err(SomError::NotFitted) if not evaluated yet
```

---

## Data Representation

All public API uses `ndarray`:
- Input data: `Array2<f64>` — row-major, shape `[n_samples, n_features]`
- Labels (unsupervised): `Array1<usize>` — linear BMU index in `[0, m*n)`
- Labels (supervised): `Array1<usize>` — class indices
- Unsupervised scores: `HashMap<EvalMethod, f64>`
- Classification scores: `HashMap<ClassEvalMethod, f64>`

No `Vec<Vec<f64>>` at the public boundary. Internal GPU buffers use backend-specific types behind the dispatch layer.

---

## Backend Dispatch

Compile-time dispatch via `match backend { }` — no `dyn Trait`, zero-cost abstraction.

```rust
// backend/mod.rs
pub(crate) fn batch_distances(
    data: &Array2<f64>,
    neurons: &Array2<f64>,
    dist_fn: DistanceFunction,
    backend: Backend,
) -> Result<Array2<f64>, SomError> {
    match backend {
        Backend::Cpu => cpu::batch_distances(data, neurons, dist_fn),
        #[cfg(feature = "cuda")]
        Backend::Cuda => cuda::batch_distances(data, neurons, dist_fn),
        #[cfg(feature = "metal")]
        Backend::Metal => metal::batch_distances(data, neurons, dist_fn),
    }
}
```

**Serialization safety:** `Backend` is excluded from `Serialize`/`Deserialize` derives on `Som`. On `Som::load`, the backend field is always initialized to `Backend::Cpu`. This prevents a model saved with `Backend::Cuda` from being loaded in a build without the `cuda` feature (where `Backend::Cuda` does not exist as a type-level variant).

### CPU Backend
- `rayon::par_iter` for parallel BMU search and neuron updates
- Distance matrix via BLAS-accelerated `ndarray` matmul: `||a-b||² = ||a||² + ||b||² - 2·aᵀb`
- Neighborhood grid update parallelized with `rayon` over `m×n`

### CUDA Backend (feature = `"cuda"`)
- `cudarc` 0.12 for device memory and kernel launch
- Custom PTX kernels compiled via `nvcc` in `build.rs`, embedded with `include_str!`
- Kernels: `batch_euclidean_distances`, `batch_cosine_distances`, `neighborhood_update`
- Data transferred to device once per `fit()`, resident for full training loop
- Returns `Err(SomError::BackendUnavailable)` if no CUDA device found at runtime

### Metal Backend (feature = `"metal"`)
- `metal` 0.33 for device and command queue
- MSL compute shaders compiled via `xcrun -sdk macosx metal` in `build.rs` → `.metallib` binary, embedded with `include_bytes!`
- Shaders: `euclidean_distances.metal`, `cosine_distances.metal`, `neighborhood_update.metal`
- Same data residency strategy as CUDA backend
- `Som` is `Send` but NOT `Sync` when Metal backend is active

### build.rs
```
if feature "cuda"  → compile src/backend/shaders/*.cu  → *.ptx  → embed via include_str!
if feature "metal" → compile src/backend/shaders/*.metal → *.metallib → embed via include_bytes!
```

---

## Algorithms Ported

### Initialization Methods (init.rs)
| Python | Rust | Implementation notes |
|--------|------|----------------------|
| `random` | `InitMethod::Random` | Uniform in [min, max] of data |
| `kde` | `InitMethod::Kde` | KDE local maxima → farthest-first selection |
| `kmeans` | `InitMethod::KMeans` | Random centroid init → Lloyd's |
| `kde_kmeans` | `InitMethod::KdekMeans` | KDE local maxima as centroid seeds → Lloyd's |
| `kmeans++` | `InitMethod::KMeansPlusPlus` | D² probabilistic seeding → Lloyd's |
| `som++` | `InitMethod::SomPlusPlus` | Farthest-first from data mean |
| `zero` | `InitMethod::Zero` | ZerO init: identity / partial-identity / Hadamard |
| `he` | `InitMethod::He` | N(0, sqrt(2/fan_in)) |
| `naive_sharding` | `InitMethod::NaiveSharding` | Sort by feature sum, split into k shards, mean each |
| `lecun` | `InitMethod::LeCun` | N(0, sqrt(1/fan_in)) |
| `lsuv` | `InitMethod::Lsuv` | SVD orthonormal init + iterative variance scaling |

### Evaluation Metrics (evals.rs)
| Metric | Access | Notes |
|--------|--------|-------|
| Silhouette score | `EvalMethod::Silhouette` | Pairwise distances, vectorized; O(n²) |
| Davies-Bouldin index | `EvalMethod::DaviesBouldin` | Centroid pairwise distances |
| Calinski-Harabasz score | `EvalMethod::CalinskiHarabasz` | Returns `Err(ZeroWithinClusterVariance)` if denominator = 0 |
| Dunn index | `EvalMethod::Dunn` | Min inter-cluster / max intra-cluster |
| BCubed precision/recall | `som.bcubed_scores(&x, &y)` | Separate method — requires ground-truth labels |
| Accuracy / F1 / Recall | `ClassEvalMethod` on SomClassification | Classification SOM only |

### KDE Kernel (kde.rs)
- Multidimensional Gaussian kernel (manual, no scipy dependency)
- Bandwidth estimator: `h = (max - min) / (1 + log2(n))`
- Local maxima detection in 1D KDE value array
- Farthest-first neuron selection from detected maxima

---

## Serialization

```rust
// Som and KMeans derive Serialize/Deserialize — Backend field excluded
#[derive(Serialize, Deserialize)]
pub struct Som {
    m: usize, n: usize, dim: usize,
    neurons: Array3<f64>,
    initial_neurons: Array3<f64>,
    init_method: InitMethod,
    dist_func: DistanceFunction,
    initial_learning_rate: f64,
    initial_neighbor_rad: f64,
    cur_learning_rate: f64,
    cur_neighbor_rad: f64,
    max_iter: Option<usize>,
    trained: bool,
    // backend: Backend  ← NOT serialized; always restored as Backend::Cpu on load
}

// Bincode v2 (always available)
// Uses bincode::config::standard() configuration
pub fn save(&self, path: &str) -> Result<(), SomError>
pub fn load(path: &str) -> Result<Self, SomError>

// JSON (feature = "serde-json")
#[cfg(feature = "serde-json")]
pub fn save_json(&self, path: &str) -> Result<(), SomError>
#[cfg(feature = "serde-json")]
pub fn load_json(path: &str) -> Result<Self, SomError>
```

---

## Testing Strategy

### Unit Tests (inline `#[cfg(test)]`)
- `distance.rs` — euclidean/cosine against known values, symmetry, triangle inequality
- `init.rs` — output shapes correct, no NaN/inf, He/LeCun variance properties
- `kmeans.rs` — convergence, centroid shape `[k, dim]`, labels in `[0, k)`, `AlreadyFitted` error
- `evals.rs` — silhouette in `[-1,1]`, DB ≥ 0, Dunn > 0 for separable clusters, `ZeroWithinClusterVariance` on degenerate input
- `som.rs` — BMU within grid bounds, neurons change after fit, predict shape matches input

### Integration Tests (`tests/`)
- `test_som.rs` — full fit/predict/evaluate cycle on synthetic blobs
- `test_kmeans.rs` — fit + predict, inertia decreases each restart
- `test_evals.rs` — all metrics on perfectly-separated clusters (expected ranges)
- `test_serialization.rs` — save → load → predict produces byte-identical output; backend resets to Cpu on load

### Property Tests (`proptest`)
```rust
proptest! {
    fn bmu_always_in_grid_bounds(data, m, n, dim) { ... }
    fn silhouette_score_in_range(x, labels) { assert!(-1.0 <= s && s <= 1.0) }
    fn kmeans_labels_in_range(data, k) { assert!(all labels < k) }
    fn fit_predict_deterministic_with_seed(data, seed) { assert_eq!(labels1, labels2) }
    fn neuron_count_matches_grid(m, n, dim) { assert_eq!(som.cluster_centers().nrows(), m*n) }
}
```

### Criterion Benchmarks (`benches/`)
```rust
// CPU benches always compiled
criterion_group!(
    som_benches,
    bench_fit_cpu_small,     // 100 samples, 3 dim, 10x10
    bench_fit_cpu_large,     // 10_000 samples, 50 dim, 20x20
    bench_predict_cpu,
);
criterion_main!(som_benches);

// GPU benches in separate cfg-gated groups
#[cfg(feature = "cuda")]
criterion_group!(som_cuda_benches, bench_fit_cuda);
#[cfg(feature = "metal")]
criterion_group!(som_metal_benches, bench_fit_metal);

criterion_group!(
    kmeans_benches,
    bench_kmeans_fit_small,
    bench_kmeans_fit_large,
);
```
Note: `#[cfg(...)]` cannot appear inside `criterion_group!` macro arguments. GPU bench groups are defined in their own `#[cfg]`-gated blocks.

---

## CI Pipeline (`.github/workflows/rust.yml`)

```yaml
jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps: [cargo test]

  test-metal:
    runs-on: macos-latest
    steps: [cargo test --features metal]

  bench-compile:
    runs-on: ubuntu-latest
    steps: [cargo bench --no-run]

  lint:
    steps:
      - cargo clippy -- -D warnings
      - cargo fmt --check
```

---

## Git Branch

All implementation work happens on branch `dev`, merged to `main` via PR when complete.

---

## Implementation Order

1. Create `dev` branch, scaffold Cargo.toml with all dependencies and feature flags
2. `core/distance.rs` + `core/neighborhood.rs` + `core/init.rs`
3. `core/kmeans.rs`
4. `core/kde.rs`
5. `core/evals.rs`
6. `core/som.rs` (CPU path via `backend/cpu.rs`)
7. `core/som_classification.rs`
8. `core/model_picker.rs`
9. `serialize.rs` (bincode v2 + optional JSON)
10. `backend/cuda.rs` + CUDA shaders + `build.rs` CUDA path
11. `backend/metal.rs` + Metal shaders + `build.rs` Metal path
12. Integration tests + proptest + criterion benchmarks
13. `Cargo.toml` metadata, rustdoc, README
