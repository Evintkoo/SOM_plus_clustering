# DenSOM — Density-Aware Self-Organising Map

**Date:** 2026-03-22
**Status:** Approved for implementation

---

## 1. Problem Statement

The standard SOM assigns every data point to a neuron, even points that lie in
low-density void regions (outliers, noise, sensor artefacts). The benchmark
confirmed this: on real-world datasets with noisy samples, cluster quality is
degraded because noise points silently inflate or distort cluster assignments.

DenSOM extends the existing `Som` to:
1. Produce a **hard noise label** (`-1`) for points in low-density regions.
2. Produce a **soft per-point density score** in `[0, 1]` so callers can
   threshold however they need.
3. Auto-detect the noise threshold from the data — no manual parameter required.

---

## 2. Algorithm

DenSOM runs in five sequential stages after (or during) SOM training:

### Stage 1 — Train
Standard SOM weight updates via `Som::fit`. During training, an auxiliary
counter `bmu_hits: Array1<usize>` of length `m × n` is incremented each time
neuron `i` wins a BMU competition. No change to the weight-update rule.

### Stage 2 — Smooth
The `bmu_hits` map is convolved with a 2-D Gaussian on the neuron grid using
the existing `neighborhood::gaussian_grid` kernel at spatial sigma `σ`
(default `1.0` grid units, user-tunable via `smooth_sigma`). Result:
`smooth_density: Array1<f64>` of length `m × n`.

Mathematically:
```
smooth_density[i] = Σ_j  gaussian(grid_dist²(i,j), 1.0, σ) · bmu_hits[j]
```

### Stage 3 — Threshold (Otsu)
Compute a 256-bin histogram of `smooth_density` across all `m × n` neurons.
Apply Otsu's method to find threshold `τ` that minimises within-class variance.

- `smooth_density[i] >= τ` → **core neuron**
- `smooth_density[i] <  τ` → **noise neuron**

Edge case: if all neurons fall below or above `τ` (flat density), all neurons
are treated as core (no noise). This produces `n_clusters ≥ 1` and
`noise_ratio = 0.0`.

### Stage 4 — Cluster Extraction
BFS connected-components over the `m × n` grid, connecting core neurons to
their 4-neighbours (up/down/left/right). Each connected component receives an
integer cluster ID `0, 1, 2, …`. Noise neurons receive cluster ID `-1`.

### Stage 5 — Label & Score Assignment
For each data point `p`:
- Find BMU `b = argmin_i dist(p, w_i)`.
- `labels[p]  = cluster_map[b]`  (i32, `-1` for noise)
- `density[p] = smooth_density[b] / max(smooth_density)`  (f64, `[0, 1]`)

---

## 3. API

### 3.1 Standalone path

```rust
let mut densom = DenSomBuilder::new()
    .grid(10, 10)
    .dim(2)
    .learning_rate(0.5)?
    .neighbor_radius(3.0)
    .init_method(InitMethod::Random)
    .distance(DistanceFunction::Euclidean)
    .smooth_sigma(1.0)          // optional, default 1.0
    .build();

densom.fit(&data.view(), epochs, shuffle, None)?;

let result: DenSomResult = densom.predict(&data.view())?;
// or shorthand:
let result = densom.fit_predict(&data.view(), epochs, shuffle, None)?;
```

### 3.2 Wrapper path (pre-trained Som)

```rust
let mut densom = DenSom::from_som(trained_som);   // takes ownership
densom.refit_density(&data.view())?;              // replays BMU hits, runs finalize()
let result = densom.predict(&data.view())?;
```

`refit_density` does **not** modify the SOM weights — it only rebuilds
`bmu_hits`, `smooth_density`, and `cluster_map`.

### 3.3 DenSomResult

```rust
pub struct DenSomResult {
    pub labels:      Array1<i64>,  // cluster id per point; -1 = noise
    pub density:     Array1<f64>,  // normalised density score [0,1] per point
    pub n_clusters:  usize,        // number of clusters found (excluding noise)
    pub noise_ratio: f64,          // fraction of points labelled -1
}
```

---

## 4. Module Layout

```
src/core/
  densom.rs              ← new file
    pub struct DenSomBuilder
    pub struct DenSom
    pub struct DenSomResult
    fn smooth_hits(hits, m, n, sigma) -> Array1<f64>
    fn otsu(values: &[f64]) -> f64
    fn connected_components(core_mask, m, n) -> Array1<i32>
    fn finalize(&mut self)              // smooth → otsu → cc
  mod.rs                 ← add pub mod densom
src/lib.rs               ← re-export DenSom, DenSomBuilder, DenSomResult
```

### DenSom fields

```rust
pub struct DenSom {
    som:            Som,
    smooth_sigma:   f64,
    bmu_hits:       Array1<usize>,   // m*n
    smooth_density: Array1<f64>,     // m*n
    cluster_map:    Array1<i32>,     // m*n; -1=noise, >=0 cluster id
    n_clusters:     usize,
    fitted:         bool,
}
```

---

## 5. Error Handling

Reuses `SomError`. Two new variants:

| Variant | When |
|---|---|
| `SomError::NotFitted` | `predict` / `refit_density` called before `fit` |
| `SomError::InsufficientData { n }` | fewer data points than 2 (Otsu needs variance) |

- `smooth_sigma = 0.0` is silently clamped to `1e-6`.
- All neurons classified as noise → valid output: `n_clusters = 0`,
  all labels `-1`, `noise_ratio = 1.0`.

---

## 6. Tests

### Unit tests (in `src/core/densom.rs`)

| Test | What it checks |
|---|---|
| `otsu_two_class` | Synthetic bimodal slice → threshold falls between the two modes |
| `connected_components_two_islands` | 5×5 grid, two separated blobs → 2 clusters |
| `smooth_hits_peak_preserved` | Single hot neuron stays local max after smoothing |
| `all_noise_returns_zero_clusters` | Flat activation below threshold → all -1, n_clusters=0 |

### Integration tests (in `tests/`)

| Test | What it checks |
|---|---|
| `densom_circles` | Circles benchmark dataset: assert `n_clusters == 2`, `noise_ratio < 0.15` |
| `densom_from_som_matches_standalone` | Same data, both API paths → identical `labels` array |

---

## 7. Out of Scope (v0.1)

- Diagonal grid connectivity (8-neighbours) — 4-neighbours is sufficient and simpler.
- KDE-based density estimator (`DensityMode::Kde`) — deferred to v0.2.
- Density-modulated training loop (Approach C) — deferred to v0.2.
- `save` / `load` for `DenSom` — can be added alongside `Som`'s serialisation.
