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
Standard SOM weight updates via `Som::fit`. `bmu_hits: Array1<usize>` of
length `m × n` is **reset to zero at the start of every `fit` call**.

**Implementation amendment:** Rather than counting hits during training (which
would require modifying `Som::fit`'s internal loop), hits are computed via a
single `Som::predict(data)` call on the *final trained weights* after `fit`
completes. This is equivalent for density estimation — it reflects which
neurons "own" each point in the converged map — and produces a cleaner density
signal than mid-training counts biased toward initial weights. `DenSom::fit` is **re-entrant** (matching
`Som` behaviour): calling it a second time continues training on the existing
weights, but resets `bmu_hits` so the density map reflects only the most recent
training run. `AlreadyFitted` is never returned.

### Stage 2 — Smooth
The `bmu_hits` map is convolved with a 2-D Gaussian across the neuron grid
using the scalar `neighborhood::gaussian` function. `gaussian_grid` is **not**
used here (it is centred on a single BMU position; this stage needs a full
cross-neuron convolution). `smooth_hits` is implemented as:

```
for each neuron i at grid position (ri, ci):
    smooth_density[i] = Σ_j  gaussian(grid_dist²(i,j), lr=1.0, radius=sigma)
                             · bmu_hits[j]
```

where `grid_dist²(i,j) = (ri−rj)² + (ci−cj)²` and `sigma` is `smooth_sigma`
(default `1.0` grid units, user-tunable). Result: `smooth_density: Array1<f64>`
of length `m × n`.

If `max(smooth_density) == 0.0` (no training data reached any neuron), all
density scores are set to `0.0` and Stage 3 is skipped — all neurons are
treated as core (no noise until data is seen).

### Stage 3 — Threshold (Otsu)
**Pre-check:** if the standard deviation of `smooth_density` across all neurons
is below `1e-9` (effectively flat), skip Otsu and classify all neurons as core.
This gives `n_clusters ≥ 1` and `noise_ratio = 0.0`.

Otherwise, compute a 256-bin histogram of `smooth_density` across all `m × n`
neurons and apply Otsu's method to find threshold `τ` that minimises
within-class variance.

- `smooth_density[i] >= τ` → **core neuron**
- `smooth_density[i] <  τ` → **noise neuron**

### Stage 4 — Cluster Extraction
BFS connected-components over the `m × n` grid, connecting core neurons to
their 4-neighbours (up/down/left/right). Each connected component receives an
integer cluster ID `0, 1, 2, …`. Noise neurons receive cluster ID `-1`.

### Stage 5 — Label & Score Assignment
For each data point `p`:
- Find BMU `b = argmin_i dist(p, w_i)`.
- `labels[p]  = cluster_map[b]`  (i32, `-1` for noise)
- `density[p] = smooth_density[b] / max(smooth_density)`  (f64, `[0, 1]`)
  If `max(smooth_density) == 0.0`, all density scores are `0.0`.
- `noise_ratio = (count of labels == -1) / total_points`

---

## 3. API

### 3.1 Standalone path

`DenSomBuilder::build()` is **infallible** (returns `DenSom` directly, not
`Result`). `smooth_sigma` is silently clamped to `[1e-6, ∞)` inside the
builder — no builder method returns `Result` other than `learning_rate`, which
inherits that behaviour from `SomBuilder`. The default value of `smooth_sigma`
is `1.0`.

```rust
let mut densom = DenSomBuilder::new()
    .grid(10, 10)
    .dim(2)
    .learning_rate(0.5)?   // only fallible method; inherited from SomBuilder
    .neighbor_radius(3.0)
    .init_method(InitMethod::Random)
    .distance(DistanceFunction::Euclidean)
    .smooth_sigma(1.0)     // optional, default 1.0; clamped silently
    .build();              // infallible

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
`bmu_hits`, `smooth_density`, and `cluster_map`. It validates
`data.ncols() == som.dim` and returns `SomError::DimensionMismatch` on
mismatch, consistent with `Som::fit`. On success it sets `fitted = true`.

`DenSom::from_som` sets `fitted = true` immediately (the wrapped `Som` is
already trained), so `refit_density` and `predict` work without a prior
`DenSom::fit` call.

### 3.3 DenSomResult

```rust
pub struct DenSomResult {
    pub labels:      Array1<i32>,  // cluster id per point; -1 = noise
    pub density:     Array1<f64>,  // normalised density score [0,1] per point
    pub n_clusters:  usize,        // number of clusters found (excluding noise)
    pub noise_ratio: f64,          // (count labels==-1) / total_points
}
```

---

## 4. Module Layout

`src/core/mod.rs` exists and declares all core submodules; add `pub mod densom`
there.

```
src/core/
  densom.rs              ← new file
    pub struct DenSomBuilder
    pub struct DenSom
    pub struct DenSomResult
    fn smooth_hits(hits: &[usize], m: usize, n: usize, sigma: f64) -> Array1<f64>
    fn otsu(values: &[f64]) -> f64
    fn connected_components(core_mask: &[bool], m: usize, n: usize) -> Array1<i32>
    fn finalize(&mut self)              // smooth → otsu → cc
  mod.rs                 ← add `pub mod densom;`
src/lib.rs               ← re-export DenSom, DenSomBuilder, DenSomResult
```

### DenSom fields

```rust
pub struct DenSom {
    som:            Som,
    smooth_sigma:   f64,             // default 1.0, clamped >= 1e-6
    bmu_hits:       Array1<usize>,   // m*n; reset each fit()
    smooth_density: Array1<f64>,     // m*n; derived in finalize()
    cluster_map:    Array1<i32>,     // m*n; -1=noise, >=0 cluster id
    n_clusters:     usize,
    fitted:         bool,
}
```

---

## 5. Error Handling

Reuses `SomError`. Two new variants added to `SomError`:

| Variant | When |
|---|---|
| `SomError::NotFitted` | `predict` / `refit_density` called before `fit` (already exists in error.rs — verify before adding) |
| `SomError::InsufficientData { n: usize }` | fewer than 2 data points (Otsu needs variance) |

- `smooth_sigma = 0.0` is silently clamped to `1e-6` in the builder.
- `DenSom::fit` is re-entrant; `AlreadyFitted` is never returned.
- `refit_density` returns `SomError::DimensionMismatch` if
  `data.ncols() != som.dim`.
- `InsufficientData { n }` is checked at the start of `fit` and
  `refit_density` (before `finalize`) against `data.nrows()`.
- The flat-density pre-check (Stage 3, std < 1e-9) always produces all-core
  (`n_clusters ≥ 1`, `noise_ratio = 0.0`). Given the pre-check and Otsu's
  guaranteed two-class split, `n_clusters = 0` is not reachable in the v0.1
  algorithm — every run with ≥ 2 data points will yield at least one cluster.

---

## 6. Tests

### Unit tests (in `src/core/densom.rs`)

| Test | What it checks |
|---|---|
| `otsu_two_class` | Synthetic bimodal slice → threshold falls strictly between the two modes |
| `connected_components_two_islands` | 5×5 grid, two separated core blobs → 2 clusters |
| `smooth_hits_peak_preserved` | Single hot neuron remains local max after smoothing |
| `flat_activation_all_core` | Flat activation (std < 1e-9) → pre-check fires, all neurons core, n_clusters ≥ 1, noise_ratio = 0.0 |
| `zero_hits_all_density_zero` | `max(smooth_density) == 0.0` guard → all density scores 0.0, Stage 3 skipped, all neurons core |

### Integration tests (in `tests/`)

| Test | What it checks |
|---|---|
| `densom_circles` | Circles benchmark dataset: assert `n_clusters == 2`, `noise_ratio < 0.15` |
| `densom_from_som_matches_standalone` | Same data, `shuffle=false`, both API paths → identical `labels` array |

The `densom_from_som_matches_standalone` test must use `shuffle=false` on both
paths so BMU hit counts are deterministic.

---

## 7. Out of Scope (v0.1)

- Diagonal grid connectivity (8-neighbours) — 4-neighbours is sufficient and simpler.
- KDE-based density estimator (`DensityMode::Kde`) — deferred to v0.2.
- Density-modulated training loop (Approach C) — deferred to v0.2.
- `save` / `load` for `DenSom` — can be added alongside `Som`'s serialisation.
