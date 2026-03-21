# Rust Migration Design — som_plus_clustering

**Date:** 2026-03-21
**Author:** Evint Leovonzko
**Status:** Approved

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
- Model serialization: `bincode` (default) + JSON (feature flag `serde-json`)
- Full test coverage: unit, integration, property-based (`proptest`), criterion benchmarks

---

## Non-Goals

- Python bindings (PyO3/maturin) — deferred to a future release
- WebAssembly target — deferred
- Streaming/online learning — not in scope for this migration

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
│   └── serialize.rs                 # save/load (bincode + optional JSON)
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
| `ndarray-rand` | 0.15 | Random array generation |
| `rayon` | 1.10 | CPU data parallelism |
| `rand` | 0.8 | RNG |
| `serde` | 1 | Serialization derive macros |
| `bincode` | 1 | Binary serialization (default) |
| `thiserror` | 1 | Ergonomic error types |
| `cudarc` | 0.12 | CUDA device management + kernel launch (optional) |
| `metal` | 0.29 | Metal device + compute pipeline (optional) |
| `serde_json` | 1 | JSON serialization (optional) |
| `criterion` | 0.5 | Benchmarks (dev) |
| `proptest` | 1 | Property-based tests (dev) |

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
    #[error("SOM must be fitted before calling {0}")]
    NotFitted(&'static str),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("input data contains NaN or infinite values")]
    InvalidInputData,
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceFunction { Euclidean, Cosine }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")] Cuda,
    #[cfg(feature = "metal")] Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvalMethod {
    Silhouette, DaviesBouldin, CalinskiHarabasz, Dunn,
    BcubedPrecision, BcubedRecall, All,
}
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
    .learning_rate(0.5)?                       // Err if > 1.76
    .neighbor_radius(1.0)
    .distance(DistanceFunction::Euclidean)
    .max_iter(1000usize)
    .backend(Backend::Cpu)
    .build()?;                                 // Result<Som, SomError>

// Training — data: Array2<f64> shape [n_samples, n_features]
som.fit(&data, epoch, shuffle, batch_size)?;

// Inference — returns Array1<usize> shape [n_samples]
let labels = som.predict(&data)?;

// Combined
let labels = som.fit_predict(&data, epoch, shuffle, batch_size)?;

// Evaluation — returns HashMap<EvalMethod, f64>
let scores = som.evaluate(&data, &[EvalMethod::Silhouette, EvalMethod::Dunn])?;
let all_scores = som.evaluate(&data, &[EvalMethod::All])?;

// Cluster centers — ArrayView2<f64> shape [m*n, dim]
let centers = som.cluster_centers();

// Serialization
som.save("model.bin")?;                        // bincode
som.save_json("model.json")?;                  // feature = "serde-json"
let som = Som::load("model.bin")?;
let som = Som::load_json("model.json")?;       // feature = "serde-json"
```

### SOM (Supervised Classification)

```rust
// fit takes labels y: Array1<usize>
som_clf.fit(&x, &y, epoch, shuffle, batch_size)?;

// predict returns predicted class labels
let y_pred = som_clf.predict(&x)?;

// evaluate with classification metrics
let scores = som_clf.evaluate(&x, &y, &[ClassEvalMethod::Accuracy, ClassEvalMethod::F1])?;
```

### KMeans

```rust
let km = KMeansBuilder::new()
    .n_clusters(10)
    .method(KMeansInit::PlusPlus)
    .max_iters(300)
    .tol(1e-6)
    .build();

km.fit(&data)?;
let labels = km.predict(&data)?;
let inertia = km.inertia();
```

### ModelPicker

```rust
let picker = ModelPicker::new();
picker.evaluate_init_methods(&data, grid_m, grid_n, learning_rate, neighbor_rad, dist_fn, max_iter, epoch)?;
let best = picker.best_model();
```

---

## Data Representation

All public API uses `ndarray`:
- Input data: `Array2<f64>` — row-major, shape `[n_samples, n_features]`
- Labels (unsupervised): `Array1<usize>` — linear BMU index in `[0, m*n)`
- Labels (supervised): `Array1<usize>` — class indices
- Scores: `HashMap<EvalMethod, f64>`

No `Vec<Vec<f64>>` at the public boundary. Internal GPU buffers use backend-specific types (device pointers for CUDA, Metal buffers) behind the dispatch layer.

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

### CPU Backend
- `rayon::par_iter` for parallel BMU search and neuron updates
- Distance matrix via BLAS-accelerated `ndarray` matmul: `||a-b||² = ||a||² + ||b||² - 2·aᵀb`
- Neighborhood grid update parallelized with `rayon` over `m×n`

### CUDA Backend (feature = `"cuda"`)
- `cudarc` for device memory and kernel launch
- Custom PTX kernels compiled via `nvcc` in `build.rs`, embedded with `include_str!`
- Kernels: `batch_euclidean_distances`, `batch_cosine_distances`, `neighborhood_update`
- Data transferred to device once per `fit()`, resident for full training loop
- Returns `Err(SomError::BackendUnavailable)` if no CUDA device found at runtime

### Metal Backend (feature = `"metal"`)
- `metal` crate for device and command queue
- MSL compute shaders compiled via `xcrun -sdk macosx metal` in `build.rs`, embedded with `include_bytes!`
- Shaders: `euclidean_distances.metal`, `cosine_distances.metal`, `neighborhood_update.metal`
- Same data residency strategy as CUDA backend

### build.rs
```
if feature "cuda"  → compile src/backend/shaders/*.cu  → *.ptx  → embed
if feature "metal" → compile src/backend/shaders/*.metal → metallib → embed
```

---

## Algorithms Ported

### Initialization Methods (init.rs)
| Python | Rust |
|--------|------|
| `random` | `InitMethod::Random` |
| `kde` | `InitMethod::Kde` |
| `kmeans` | `InitMethod::KMeans` |
| `kde_kmeans` | `InitMethod::KdekMeans` |
| `kmeans++` | `InitMethod::KMeansPlusPlus` |
| `som++` | `InitMethod::SomPlusPlus` |
| `zero` | `InitMethod::Zero` (ZerO init with Hadamard) |
| `he` | `InitMethod::He` |
| `naive_sharding` | `InitMethod::NaiveSharding` |
| `lecun` | `InitMethod::LeCun` |
| `lsuv` | `InitMethod::Lsuv` (SVD orthonormal + variance scaling) |

### Evaluation Metrics (evals.rs)
| Metric | Notes |
|--------|-------|
| Silhouette score | Pairwise distances via `ndarray`, vectorized |
| Davies-Bouldin index | Centroid distances via `pdist` equivalent |
| Calinski-Harabasz score | Between/within dispersion ratio |
| Dunn index | Min inter-cluster / max intra-cluster |
| BCubed precision/recall | Per-sample correctness vs ground truth |
| Accuracy / F1 / Recall | Classification SOM only |

### KDE Kernel (kde.rs)
- Multidimensional Gaussian kernel
- Bandwidth estimator: `h = (max - min) / (1 + log2(n))`
- Local maxima detection in KDE values
- Farthest-first neuron selection from maxima

---

## Serialization

```rust
// Som and KMeans derive serde Serialize/Deserialize
#[derive(Serialize, Deserialize)]
pub struct Som { ... }

// bincode (always available)
pub fn save(&self, path: &str) -> Result<(), SomError>
pub fn load(path: &str) -> Result<Self, SomError>

// JSON (feature = "serde-json")
#[cfg(feature = "serde-json")]
pub fn save_json(&self, path: &str) -> Result<(), SomError>
#[cfg(feature = "serde-json")]
pub fn load_json(path: &str) -> Result<Self, SomError>
```

GPU backend state (device pointers, Metal buffers) is not serialized. On load, the backend defaults to `Backend::Cpu`; callers can re-specify GPU backend before further training.

---

## Testing Strategy

### Unit Tests (inline `#[cfg(test)]`)
- `distance.rs` — euclidean/cosine against known values, symmetry, triangle inequality
- `init.rs` — output shapes correct, no NaN/inf, He/LeCun variance properties
- `kmeans.rs` — convergence, centroid shape `[k, dim]`, labels in `[0, k)`
- `evals.rs` — silhouette in `[-1,1]`, DB ≥ 0, Dunn > 0 for separable clusters
- `som.rs` — BMU within grid bounds, neurons change after fit, predict shape matches input

### Integration Tests (`tests/`)
- `test_som.rs` — full fit/predict/evaluate cycle on synthetic blobs
- `test_kmeans.rs` — fit + predict, inertia decreases each restart
- `test_evals.rs` — all metrics on perfectly-separated clusters (expected ranges)
- `test_serialization.rs` — save → load → predict produces byte-identical output

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
criterion_group!(
    som_benches,
    bench_fit_cpu_small,     // 100 samples, 3 dim, 10x10
    bench_fit_cpu_large,     // 10_000 samples, 50 dim, 20x20
    bench_predict_cpu,
    #[cfg(feature="cuda")]  bench_fit_cuda,
    #[cfg(feature="metal")] bench_fit_metal,
);

criterion_group!(
    kmeans_benches,
    bench_kmeans_fit_small,
    bench_kmeans_fit_large,
);
```

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

1. Create `dev` branch, scaffold Cargo workspace
2. `core/distance.rs` + `core/neighborhood.rs` + `core/init.rs`
3. `core/kmeans.rs`
4. `core/kde.rs`
5. `core/evals.rs`
6. `core/som.rs` (CPU path via `backend/cpu.rs`)
7. `core/som_classification.rs`
8. `core/model_picker.rs`
9. `serialize.rs`
10. `backend/cuda.rs` + CUDA shaders + `build.rs` CUDA path
11. `backend/metal.rs` + Metal shaders + `build.rs` Metal path
12. Integration tests + proptest + criterion benchmarks
13. `Cargo.toml` metadata, docs, README
