# AutoSom Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve AutoSom k-detection quality (elbow + gap statistic) and speed (separable Gaussian, adaptive sigma, remove per-k silhouette).

**Architecture:** Stage 2 k-selection is replaced: inertia-only sweep → elbow detection → gap statistic on top-5 candidates → best_k. DenSOM `smooth_hits` is replaced with a separable+truncated O(mn·r) Gaussian. Adaptive sigma (Silverman/Scott) replaces the fixed default. All other stages are unchanged.

**Tech Stack:** Rust 2021, ndarray 0.16, rand 0.9 (StdRng + SeedableRng + SliceRandom)

**Spec:** `docs/superpowers/specs/2026-03-22-autosom-improvements-design.md`

---

## File Map

| File | What changes |
|------|-------------|
| `src/core/mod.rs` | Add `SigmaMethod` enum |
| `src/lib.rs` | Re-export `SigmaMethod` |
| `src/core/densom.rs` | Replace `smooth_hits` fn with separable+truncated version |
| `src/core/autosom.rs` | Add `sigma_method`/`gap_n_refs` to builder; add `compute_adaptive_sigma`, `elbow_candidates`, `gap_statistic` helpers; replace k-sweep body; replace `build_sample_idx` with random; remove unused `calinski_harabasz_score` import |

---

## Task 1: Add `SigmaMethod` enum

**Files:**
- Modify: `src/core/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add enum to `src/core/mod.rs`**

Append after the existing `pub mod densom;` line:

```rust
/// Bandwidth selection rule for DenSOM's adaptive Gaussian smoothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaMethod {
    /// Silverman's rule (robust to outliers via IQR). Default.
    Silverman,
    /// Scott's rule (`n^(-1/(d+4)) * σ_data`).
    Scott,
}
```

- [ ] **Step 2: Re-export from `src/lib.rs`**

Add to the `pub use` block, after the `AutoSomBuilder` line:

```rust
pub use core::SigmaMethod;
```

- [ ] **Step 3: Verify it compiles**

```bash
cd /Users/evintleovonzko/Documents/research/SOM_plus_clustering
cargo check 2>&1 | head -20
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/core/mod.rs src/lib.rs
git commit -m "feat: add SigmaMethod enum for adaptive DenSOM bandwidth selection"
```

---

## Task 2: Replace `smooth_hits` with separable+truncated Gaussian

**Files:**
- Modify: `src/core/densom.rs` (replace `smooth_hits` fn only — lines 38–57)

The current `smooth_hits` is O(m²n²). The replacement is two 1D passes, each truncated at radius `r = ceil(3σ)`, giving O(mn·(4r+2)).

- [ ] **Step 1: Write the two unit tests first**

Add inside the `#[cfg(test)] mod tests` block at the bottom of `src/core/densom.rs`:

```rust
#[test]
fn smooth_hits_separable_matches_full_square() {
    // 5×5 grid, mixed hits — compare new vs old output element-by-element.
    let mut hits = [0usize; 25];
    hits[0]  = 10;
    hits[6]  = 5;
    hits[12] = 20;
    hits[18] = 3;
    hits[24] = 8;
    let old = smooth_hits_reference(&hits, 5, 5, 1.0);
    let new = smooth_hits(&hits, 5, 5, 1.0);
    for (o, n) in old.iter().zip(new.iter()) {
        assert!((o - n).abs() < 1e-8, "square mismatch: old={o} new={n}");
    }
}

#[test]
fn smooth_hits_separable_matches_full_rect() {
    // 4×6 grid — catches m≠n off-by-one bugs in row/col indexing.
    let mut hits = [0usize; 24];
    hits[0]  = 7;
    hits[5]  = 3;
    hits[10] = 15;
    hits[17] = 9;
    hits[23] = 2;
    let old = smooth_hits_reference(&hits, 4, 6, 1.5);
    let new = smooth_hits(&hits, 4, 6, 1.5);
    for (o, n) in old.iter().zip(new.iter()) {
        assert!((o - n).abs() < 1e-8, "rect mismatch: old={o} new={n}");
    }
}
```

- [ ] **Step 2: Run tests — expect compile failure (reference fn doesn't exist yet)**

```bash
cargo test -p som_plus_clustering smooth_hits_separable 2>&1 | tail -10
```

Expected: compile error (missing `smooth_hits_reference`).

- [ ] **Step 3: Add `smooth_hits_reference` (the old O(m²n²) impl) inside the test module**

Add inside `#[cfg(test)] mod tests`:

```rust
/// Old O(m²n²) implementation kept as reference for testing only.
fn smooth_hits_reference(hits: &[usize], m: usize, n: usize, sigma: f64) -> ndarray::Array1<f64> {
    let sigma = sigma.max(1e-6);
    let mn = m * n;
    let mut out = ndarray::Array1::<f64>::zeros(mn);
    for i in 0..mn {
        let ri = (i / n) as f64;
        let ci = (i % n) as f64;
        let mut acc = 0.0f64;
        for (j, &hit) in hits.iter().enumerate().take(mn) {
            let rj = (j / n) as f64;
            let cj = (j % n) as f64;
            let dr = ri - rj;
            let dc = ci - cj;
            let dist_sq = dr * dr + dc * dc;
            acc += crate::core::neighborhood::gaussian(dist_sq, 1.0, sigma) * hit as f64;
        }
        out[i] = acc;
    }
    out
}
```

- [ ] **Step 4: Run tests — expect FAIL because `smooth_hits` still uses old implementation**

```bash
cargo test -p som_plus_clustering smooth_hits_separable 2>&1 | tail -10
```

Expected: tests pass (old == old), but this is now comparing old vs old. Actually — the tests compare `smooth_hits_reference` vs `smooth_hits`. They should pass right now (since smooth_hits IS the reference). Keep this in mind: after replacing smooth_hits in the next step, the tests will verify the new vs old.

- [ ] **Step 5: Replace the `smooth_hits` function body**

In `src/core/densom.rs`, replace the entire `smooth_hits` function (lines 38–57, from `fn smooth_hits` through the closing `}`):

```rust
/// Gaussian-smooth the flat BMU hit map over the m×n neuron grid.
///
/// Uses separable 1D convolution (row pass then column pass), truncated at
/// radius `r = ceil(3σ)`. Complexity: O(mn·(4r+2)) vs the naive O(m²n²).
/// Output is mathematically identical to a full 2D Gaussian (kernel separability).
fn smooth_hits(hits: &[usize], m: usize, n: usize, sigma: f64) -> Array1<f64> {
    let sigma = sigma.max(1e-6);
    let r = (3.0 * sigma).ceil() as usize;
    let mn = m * n;

    // Precompute 1D Gaussian kernel weights for offsets -r..=r (length = 2r+1).
    // weights[i] corresponds to offset (i as isize - r as isize).
    let weights: Vec<f64> = (0..=(2 * r))
        .map(|i| {
            let d = i as f64 - r as f64;
            (-d * d / (2.0 * sigma * sigma)).exp()
        })
        .collect();

    // Row pass: convolve each row horizontally.
    // out_h[i*n + j] = Σ_{|dj|≤r} hits[i*n+(j+dj)] · w[dj+r]
    let mut out_h = vec![0.0f64; mn];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for (wi, dj_signed) in (0..=(2 * r)).map(|x| (x, x as isize - r as isize)) {
                let jj = j as isize + dj_signed;
                if jj >= 0 && jj < n as isize {
                    acc += weights[wi] * hits[i * n + jj as usize] as f64;
                }
            }
            out_h[i * n + j] = acc;
        }
    }

    // Column pass: convolve each column vertically over out_h.
    // out[i*n + j] = Σ_{|di|≤r} out_h[(i+di)*n+j] · w[di+r]
    let mut out = Array1::<f64>::zeros(mn);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for (wi, di_signed) in (0..=(2 * r)).map(|x| (x, x as isize - r as isize)) {
                let ii = i as isize + di_signed;
                if ii >= 0 && ii < m as isize {
                    acc += weights[wi] * out_h[ii as usize * n + j];
                }
            }
            out[i * n + j] = acc;
        }
    }
    out
}
```

- [ ] **Step 6: Run the two new tests — expect PASS**

```bash
cargo test -p som_plus_clustering smooth_hits_separable 2>&1 | tail -10
```

Expected: both `smooth_hits_separable_matches_full_square` and `smooth_hits_separable_matches_full_rect` pass.

- [ ] **Step 7: Run the full existing DenSOM test suite**

```bash
cargo test -p som_plus_clustering densom 2>&1 | tail -20
```

Expected: all existing densom tests still pass.

- [ ] **Step 8: Commit**

```bash
git add src/core/densom.rs
git commit -m "perf: replace smooth_hits with separable+truncated O(mn*r) Gaussian"
```

---

## Task 3: Adaptive sigma + AutoSomBuilder fields

**Files:**
- Modify: `src/core/autosom.rs`

Adds `sigma_method` and `gap_n_refs` to the builder, adds the `compute_adaptive_sigma` helper, and wires it into `fit_predict` Stage 4. Does NOT yet change the k-sweep (Task 6 does that).

- [ ] **Step 1: Write the unit test first**

Add inside `#[cfg(test)] mod tests` in `src/core/autosom.rs`:

```rust
#[test]
fn silverman_sigma_formula() {
    // d=2, n=100, σ=1.0 per dim, IQR=1.349 per dim (≈ normal dist IQR)
    // Silverman: h = 0.9 * min(1.0, 1.349/1.34) * 100^(-0.2)
    //            = 0.9 * min(1.0, 1.0067) * 0.3981
    //            = 0.9 * 1.0 * 0.3981 ≈ 0.3583
    // Clamped to [0.5, grid_side/2]: clamp(0.3583, 0.5, 5.0) = 0.5
    use ndarray::Array2;
    use crate::core::SigmaMethod;
    // Build a 2D dataset with known std ≈ 1.0 and IQR ≈ 1.349 (standard normal)
    // Use evenly-spaced quantiles of N(0,1) for n=100.
    // σ_data ≈ 1.0, IQR ≈ 1.349 for a standard normal.
    // For simplicity, use manually constructed data.
    let mut v = Vec::with_capacity(200);
    for i in 0..100i64 {
        // uniform -2..2 gives std≈1.155, IQR≈2.0 — not ideal, but tests the formula
        let x = -2.0 + 4.0 * i as f64 / 99.0;
        v.push(x); v.push(x * 0.5);
    }
    let data = Array2::from_shape_vec((100, 2), v).unwrap();
    let h = compute_adaptive_sigma(&data.view(), 10, 10, SigmaMethod::Silverman);
    // Result must be in [0.5, 5.0] (clamped range for 10×10 grid)
    assert!(h >= 0.5 && h <= 5.0, "sigma {h} out of clamped range [0.5, 5.0]");
    // Silverman result is deterministic — must not change between runs
    let h2 = compute_adaptive_sigma(&data.view(), 10, 10, SigmaMethod::Silverman);
    assert_eq!(h, h2, "sigma computation must be deterministic");
}
```

Note: the test references `SigmaMethod` and `compute_adaptive_sigma`. These will be added in the next step.

- [ ] **Step 2: Add `sigma_method` and `gap_n_refs` fields to `AutoSomBuilder`**

In `src/core/autosom.rs`, update the imports at the top to include `SigmaMethod`:

```rust
use crate::core::SigmaMethod;
```

Then add two fields to `AutoSomBuilder`:

```rust
pub struct AutoSomBuilder {
    som_m:           usize,
    som_n:           usize,
    dim:             usize,
    learning_rate:   Option<f64>,
    neighbor_radius: f64,
    init_method:     InitMethod,
    distance:        DistanceFunction,
    smooth_sigma:    f64,
    sigma_method:    SigmaMethod,   // NEW
    gap_n_refs:      usize,         // NEW
}
```

Update `AutoSomBuilder::new()` defaults:

```rust
sigma_method:    SigmaMethod::Silverman,
gap_n_refs:      10,
```

Add builder methods after `smooth_sigma`:

```rust
pub fn sigma_method(mut self, m: SigmaMethod) -> Self {
    self.sigma_method = m;
    self
}

pub fn gap_n_refs(mut self, n: usize) -> Self {
    self.gap_n_refs = n.max(1);
    self
}
```

Update `AutoSom` struct to hold the new fields (they're already on `cfg: AutoSomBuilder`, so no separate struct field needed).

- [ ] **Step 3: Add `compute_adaptive_sigma` helper function**

Add as a module-level free function in `src/core/autosom.rs`, before the `#[cfg(test)]` block:

```rust
/// Compute bandwidth `h` for DenSOM Gaussian smoothing from data statistics.
///
/// Uses Silverman's rule (default) or Scott's rule across all d dimensions,
/// averaging per-dimension std and IQR. Clamps result to `[0.5, max(m,n)/2]`.
fn compute_adaptive_sigma(
    data:   &ArrayView2<f64>,
    m:      usize,
    n:      usize,
    method: SigmaMethod,
) -> f64 {
    let n_samples = data.nrows();
    let d = data.ncols();
    let grid_side = m.max(n) as f64;

    let mut std_sum = 0.0f64;
    let mut iqr_sum = 0.0f64;

    for j in 0..d {
        let mut col: Vec<f64> = data.column(j).to_vec();
        let mean = col.iter().sum::<f64>() / n_samples as f64;
        let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / n_samples as f64;
        std_sum += variance.sqrt();

        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q25_idx = ((n_samples as f64 * 0.25) as usize).min(n_samples - 1);
        let q75_idx = ((n_samples as f64 * 0.75) as usize).min(n_samples - 1);
        iqr_sum += col[q75_idx] - col[q25_idx];
    }

    let sigma_data = (std_sum / d as f64).max(1e-12);
    let iqr_mean   = (iqr_sum / d as f64).max(1e-12);

    let h = match method {
        SigmaMethod::Scott => {
            let exp = -1.0 / (d as f64 + 4.0);
            (n_samples as f64).powf(exp) * sigma_data
        }
        SigmaMethod::Silverman => {
            let iqr_adj = iqr_mean / 1.34;
            0.9 * sigma_data.min(iqr_adj) * (n_samples as f64).powf(-0.2)
        }
    };

    h.clamp(0.5, grid_side / 2.0)
}
```

- [ ] **Step 4: Run the silverman test — expect PASS**

```bash
cargo test -p som_plus_clustering silverman_sigma_formula 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Wire adaptive sigma into Stage 4 in `fit_predict`**

In `fit_predict`, the Stage 4 block currently reads:

```rust
let mut densom = DenSom::from_som(som.clone());
densom.set_smooth_sigma(c.smooth_sigma);
```

Replace that `set_smooth_sigma` line with:

```rust
let adaptive_sigma = compute_adaptive_sigma(data, c.som_m, c.som_n, c.sigma_method);
densom.set_smooth_sigma(adaptive_sigma);
```

- [ ] **Step 6: Run full AutoSom tests**

```bash
cargo test -p som_plus_clustering autosom 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/core/autosom.rs
git commit -m "feat: add sigma_method/gap_n_refs to AutoSomBuilder; add adaptive sigma computation"
```

---

## Task 4: Elbow detection helper

**Files:**
- Modify: `src/core/autosom.rs`

- [ ] **Step 1: Write the four unit tests first**

Add inside `#[cfg(test)] mod tests` in `src/core/autosom.rs`:

```rust
#[test]
fn elbow_detect_known_knee() {
    // Inertia drops steeply from k=2 to k=5, then flattens.
    // The knee should be detected at k=5.
    let inertias: Vec<(usize, f64)> = vec![
        (2, 1000.0), (3, 500.0), (4, 200.0), (5, 50.0),
        (6, 45.0), (7, 42.0), (8, 40.0),
    ];
    let (k_elbow, _) = elbow_candidates(&inertias, 8);
    assert_eq!(k_elbow, 5, "knee at k=5, got {k_elbow}");
}

#[test]
fn elbow_detect_linear_fallback() {
    // Perfectly linear inertia — no elbow — should fall back to k=2.
    let inertias: Vec<(usize, f64)> = (2..=8).map(|k| (k, 100.0 - k as f64 * 10.0)).collect();
    let (k_elbow, _) = elbow_candidates(&inertias, 8);
    assert_eq!(k_elbow, 2, "linear curve → fallback k=2, got {k_elbow}");
}

#[test]
fn elbow_candidates_clamped_dedup() {
    // k_elbow = max_k → candidates {max_k-2, max_k-1, max_k, max_k, max_k}
    // After clamp + dedup → {max_k-2, max_k-1, max_k} (length 3).
    let max_k = 6usize;
    let inertias: Vec<(usize, f64)> = vec![
        (2, 1000.0), (3, 500.0), (4, 200.0),
        (5, 100.0), (6, 10.0),
    ];
    let (_, cands) = elbow_candidates(&inertias, max_k);
    assert!(cands.len() < 5, "clamped dedup should give <5 candidates, got {:?}", cands);
    assert!(cands.iter().all(|&k| k >= 2 && k <= max_k),
        "all candidates must be in [2, max_k={max_k}], got {:?}", cands);
    // No duplicates
    let mut deduped = cands.clone();
    deduped.dedup();
    assert_eq!(deduped, cands, "candidates must have no duplicates");
}

#[test]
fn elbow_empty_sweep_fallback() {
    // max_k=2: only one inertia value, second-difference array is empty.
    // Must return k_elbow=2 without panicking.
    let inertias: Vec<(usize, f64)> = vec![(2, 100.0)];
    let (k_elbow, cands) = elbow_candidates(&inertias, 2);
    assert_eq!(k_elbow, 2);
    assert_eq!(cands, vec![2]);
}
```

- [ ] **Step 2: Run tests — expect compile failure**

```bash
cargo test -p som_plus_clustering elbow_ 2>&1 | tail -5
```

Expected: compile error (`elbow_candidates` not defined).

- [ ] **Step 3: Implement `elbow_candidates`**

Add as a module-level free function in `src/core/autosom.rs`:

```rust
/// Detects the elbow k from an inertia curve using the second-difference (Kneedle) method.
///
/// Returns `(k_elbow, candidates)` where candidates is a sorted, deduplicated Vec
/// of up to 5 k values centered on k_elbow, clamped to `[2, max_k]`.
fn elbow_candidates(
    inertias: &[(usize, f64)], // (k, inertia) pairs, must be sorted by k ascending
    max_k:    usize,
) -> (usize, Vec<usize>) {
    // Need ≥3 points to compute a second difference.
    if inertias.len() < 3 {
        return (2, vec![2]);
    }

    let w_min = inertias.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let w_max = inertias.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let w_range = (w_max - w_min).max(1e-12);

    let w_norm: Vec<f64> = inertias.iter()
        .map(|p| (p.1 - w_min) / w_range)
        .collect();

    // Second difference over interior indices only.
    let mut best_d2 = f64::NEG_INFINITY;
    let mut best_idx = 1usize;
    for i in 1..w_norm.len() - 1 {
        let d2 = w_norm[i - 1] - 2.0 * w_norm[i] + w_norm[i + 1];
        if d2 > best_d2 {
            best_d2 = d2;
            best_idx = i;
        }
    }

    let k_elbow = if best_d2 < 1e-9 {
        2 // linear/flat curve — no elbow
    } else {
        inertias[best_idx].0
    };

    // Build candidate window: {k_elbow-2 .. k_elbow+2}, clamped, deduplicated.
    let mut cands: Vec<usize> = [
        k_elbow.saturating_sub(2).max(2),
        k_elbow.saturating_sub(1).max(2),
        k_elbow,
        (k_elbow + 1).min(max_k),
        (k_elbow + 2).min(max_k),
    ]
    .into_iter()
    .collect();
    cands.sort();
    cands.dedup();

    (k_elbow, cands)
}
```

- [ ] **Step 4: Run elbow tests — expect PASS**

```bash
cargo test -p som_plus_clustering elbow_ 2>&1 | tail -10
```

Expected: all four elbow tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/autosom.rs
git commit -m "feat: add elbow_candidates (Kneedle) for fast k pre-selection"
```

---

## Task 5: Gap statistic helper

**Files:**
- Modify: `src/core/autosom.rs`

- [ ] **Step 1: Write the unit test first**

Add inside `#[cfg(test)] mod tests` in `src/core/autosom.rs`:

```rust
#[test]
fn gap_statistic_two_blobs() {
    // Two tight, well-separated blobs at (0,0) and (10,10).
    // Gap statistic with a fixed seed must select k=2.
    let mut v = Vec::with_capacity(200);
    for i in 0..50i64 { v.push(i as f64 * 0.01); v.push(i as f64 * 0.01); }
    for i in 0..50i64 { v.push(10.0 + i as f64 * 0.01); v.push(10.0 + i as f64 * 0.01); }
    let data = Array2::from_shape_vec((100, 2), v).unwrap();
    let candidates = vec![2, 3, 4];
    let best = gap_statistic(&data.view(), &candidates, 10, 2);
    assert_eq!(best, 2, "two blobs → gap must pick k=2, got k={best}");
}
```

- [ ] **Step 2: Run test — expect compile failure**

```bash
cargo test -p som_plus_clustering gap_statistic_two_blobs 2>&1 | tail -5
```

Expected: compile error (`gap_statistic` not defined).

- [ ] **Step 3: Add import for `rand` seeded RNG at the top of `autosom.rs`**

Add to the existing `use` block at the top:

```rust
use rand::{SeedableRng, rngs::StdRng, Rng};
```

- [ ] **Step 4: Implement `gap_statistic`**

Add as a module-level free function in `src/core/autosom.rs`:

```rust
/// Gap statistic (Tibshirani et al. 2001) for k selection.
///
/// For each candidate k, computes Gap(k) = E[log W_k^ref] - log W_k, where
/// W_k^ref is the within-cluster dispersion on `n_refs` uniformly random
/// reference datasets in the data bounding box. Uses Tibshirani's stopping
/// criterion over the sorted candidate list.
///
/// Falls back to `k_elbow` if all KMeans fits fail.
fn gap_statistic(
    sample_data: &ArrayView2<f64>,
    candidates:  &[usize],
    n_refs:      usize,
    k_elbow:     usize,
) -> usize {
    let n = sample_data.nrows();
    let d = sample_data.ncols();

    // Per-dimension bounding box for reference data generation.
    let mut col_min = vec![f64::INFINITY;     d];
    let mut col_max = vec![f64::NEG_INFINITY; d];
    for j in 0..d {
        for i in 0..n {
            let v = sample_data[[i, j]];
            if v < col_min[j] { col_min[j] = v; }
            if v > col_max[j] { col_max[j] = v; }
        }
    }

    // Per-call seed derived from sample size for reproducibility.
    let base_seed = n as u64;
    let mut gap_vals: Vec<(usize, f64, f64)> = Vec::new(); // (k, gap, se)

    for &k in candidates {
        if n < 2 * k { continue; } // too few points for this k

        // Inertia on actual data.
        let mut km = KMeansBuilder::new()
            .n_clusters(k)
            .method(KMeansInit::PlusPlus)
            .max_iters(100)
            .build();
        if km.fit(sample_data).is_err() { continue; }
        let wk = km.inertia().unwrap_or(0.0);
        let log_wk = if wk > 0.0 { wk.ln() } else { 0.0 };

        // Inertia on n_refs uniform reference datasets.
        let mut ref_log_wks: Vec<f64> = Vec::with_capacity(n_refs);
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(k as u64));

        for _ in 0..n_refs {
            let ref_data = Array2::from_shape_fn((n, d), |(_i, j)| {
                let lo = col_min[j];
                let hi = col_max[j];
                if (hi - lo).abs() < 1e-12 { lo }
                else { lo + rng.random::<f64>() * (hi - lo) }
            });
            let mut km_ref = KMeansBuilder::new()
                .n_clusters(k)
                .method(KMeansInit::PlusPlus)
                .max_iters(100)
                .build();
            if km_ref.fit(&ref_data.view()).is_err() { continue; }
            let w_ref = km_ref.inertia().unwrap_or(0.0);
            if w_ref > 0.0 { ref_log_wks.push(w_ref.ln()); }
        }

        if ref_log_wks.is_empty() { continue; }

        let b        = ref_log_wks.len() as f64;
        let mean_ref = ref_log_wks.iter().sum::<f64>() / b;
        let gap      = mean_ref - log_wk;
        let std_ref  = (ref_log_wks.iter()
            .map(|&x| (x - mean_ref).powi(2))
            .sum::<f64>()
            / b).sqrt();
        let se = std_ref * (1.0 + 1.0 / b).sqrt();

        gap_vals.push((k, gap, se));
    }

    if gap_vals.is_empty() {
        return k_elbow;
    }

    // Tibshirani criterion: smallest k where Gap(k) ≥ Gap(k_next) - SE(k_next).
    // "k_next" is the next element in gap_vals (not necessarily k+1 in integers).
    for i in 0..gap_vals.len().saturating_sub(1) {
        let (k, gap_k, _)      = gap_vals[i];
        let (_, gap_next, se_next) = gap_vals[i + 1];
        if gap_k >= gap_next - se_next {
            return k;
        }
    }

    // Fallback: k with maximum Gap(k).
    gap_vals.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(k, _, _)| k)
        .unwrap_or(k_elbow)
}
```

- [ ] **Step 5: Run the gap test — expect PASS**

```bash
cargo test -p som_plus_clustering gap_statistic_two_blobs 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/autosom.rs
git commit -m "feat: add gap_statistic (Tibshirani 2001) for principled k selection"
```

---

## Task 6: Wire Stage 2 + stratified sampling

**Files:**
- Modify: `src/core/autosom.rs`

This is the final wiring task. It replaces the k-sweep body, updates `build_sample_idx`, and removes the now-unused `calinski_harabasz_score` import.

- [ ] **Step 1: Replace `build_sample_idx` with random sampling**

Replace the entire `build_sample_idx` function:

```rust
fn build_sample_idx(n: usize, max_n: usize) -> Vec<usize> {
    if n <= max_n {
        return (0..n).collect();
    }
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(n as u64);
    indices.partial_shuffle(&mut rng, max_n);
    let mut sample = indices[..max_n].to_vec();
    sample.sort_unstable();
    sample
}
```

- [ ] **Step 2: Run existing autosom tests to make sure nothing breaks**

```bash
cargo test -p som_plus_clustering autosom 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 3: Replace Stage 2 in `fit_predict`**

In `fit_predict`, locate the Stage 2 block. The current block (roughly lines 160–241 in `autosom.rs`) runs a `for k in 2..=max_k` loop computing silhouette + CH per k.

Replace the **entire Stage 2 block** (from the `// Step 2` comment through the `};` that closes `best_k`) with:

```rust
        // ------------------------------------------------------------------ //
        // Step 2: k-Selection via elbow detection + gap statistic             //
        //                                                                     //
        // 2a. KMeans on neurons for k=2..max_k, record inertia only.         //
        // 2b. Elbow detection (Kneedle) → top-5 k candidates.                //
        // 2c. Gap statistic on raw data sample for top-5 → best_k.           //
        // ------------------------------------------------------------------ //
        let mn = c.som_m * c.som_n;
        let max_k = (mn / 2).min(50).max(2);

        // 2a: inertia sweep on the neuron lattice (cheap: neurons << data).
        let mut inertia_curve: Vec<(usize, f64)> = Vec::new();
        for k in 2..=max_k {
            let mut best_inertia = f64::INFINITY;
            for _ in 0..N_RESTARTS {
                let mut km = KMeansBuilder::new()
                    .n_clusters(k)
                    .method(KMeansInit::PlusPlus)
                    .max_iters(100)
                    .build();
                if km.fit(&neurons.view()).is_err() { continue; }
                let inertia = km.inertia().unwrap_or(f64::INFINITY);
                if inertia < best_inertia { best_inertia = inertia; }
            }
            if best_inertia.is_finite() {
                inertia_curve.push((k, best_inertia));
            }
        }

        // 2b: elbow → top-5 candidates.
        let (k_elbow, candidates) = elbow_candidates(&inertia_curve, max_k);

        // 2c: gap statistic on the subsample → best_k.
        let best_k = gap_statistic(&sample_data.view(), &candidates, c.gap_n_refs, k_elbow);
```

- [ ] **Step 4: Remove `calinski_harabasz_score` from imports**

In the `use` block at the top of `autosom.rs`, change:

```rust
        evals::{calinski_harabasz_score, silhouette_score},
```

to:

```rust
        evals::silhouette_score,
```

Also remove the now-unused `bmu_labels` variable. In the old code it was only used inside the k-sweep loop (to map neuron clusters back to data labels for silhouette scoring). After replacing Stage 2, it is unused. Delete the line `let bmu_labels = som.predict(data)?;` entirely.

- [ ] **Step 5: Verify `c.gap_n_refs` is accessible**

The `fit_predict` method accesses `c` as `let c = &self.cfg;`. Since `gap_n_refs` is now on `AutoSomBuilder`, `c.gap_n_refs` will work directly.

- [ ] **Step 6: Run the full test suite**

```bash
cargo test -p som_plus_clustering 2>&1 | tail -20
```

Expected: all tests pass, including `autosom_blobs_finds_two_clusters` and `autosom_returns_valid_labels`.

- [ ] **Step 7: Run clippy**

```bash
cargo clippy 2>&1 | grep "^error" | head -10
```

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/core/autosom.rs
git commit -m "feat: replace AutoSom k-sweep with elbow+gap statistic two-stage selection"
```

---

## Task 7: Verify benchmark regression

Run the full example benchmark to check ARI on s1 and a1.

- [ ] **Step 1: Build in release mode**

```bash
cd /Users/evintleovonzko/Documents/research/SOM_plus_clustering
cargo build --release --example benchmark_quality 2>&1 | tail -5
```

Expected: builds cleanly.

- [ ] **Step 2: Run the benchmark example (will take a few minutes)**

```bash
cargo run --release --example benchmark_quality 2>&1 | tail -30
```

Expected: completes, writes updated CSVs and `rust_metrics.json`.

- [ ] **Step 3: Inspect s1 and a1 results**

```bash
python3 -c "
import json
with open('experiments/benchmark/results/full_results.json') as f:
    r = json.load(f)
for name in ['s1', 'a1']:
    auto = r[name]['autosom']
    print(f'{name}: k_detected={auto[\"k_detected\"]}, ARI={auto.get(\"ari\", \"N/A\"):.4f}')
"
```

Expected: s1 ARI ≥ 0.90, a1 ARI ≥ 0.90. If regression detected, investigate the gap statistic candidate window — the elbow may be underestimating k for a particular dataset.

- [ ] **Step 4: Final commit with results note**

```bash
git add experiments/benchmark/results/
git commit -m "chore: update benchmark results after AutoSom elbow+gap improvements"
```

---

## Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| k-sweep scoring | sil + 0.5*CH per k, O(n²) silhouette × 50 | inertia only, then gap stat on 5 candidates |
| smooth_hits | O(m²n²) full 2D Gaussian | O(mn·(4r+2)) separable+truncated |
| DenSOM sigma | fixed `1.0` (scaled internally) | Silverman/Scott from data statistics |
| Sample indexing | evenly-spaced stride | random partial shuffle (fixed seed) |
| Public API additions | — | `SigmaMethod`, `sigma_method()`, `gap_n_refs()` |
