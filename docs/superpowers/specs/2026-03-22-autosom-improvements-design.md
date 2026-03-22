# AutoSom Improvements: Two-Stage k-Selection + Speed Fixes

**Date:** 2026-03-22
**Status:** Approved (v3 — spec-review fixes applied)

## Problem Statement

AutoSom has two interrelated weaknesses:

1. **k-detection quality** — on the S-set benchmarks (true k=15), AutoSom detects k=4, 8, or 20 due to a scoring function (`norm_sil + 0.5 * norm_ch`) that is noisy over the full k=2..50 sweep.
2. **Speed** — AutoSom runs ~1.4s vs KMeans at 8–50ms. The main bottlenecks are `smooth_hits` at O(m²n²) and silhouette computed O(n²) for each k in the sweep.

## Architecture

The overall pipeline shape is preserved. Stage 2 (k-selection) and Stage 4 (DenSOM smoothing) change internally. Stage 5 is unchanged and continues to use silhouette for the final algorithm comparison.

```
fit_predict(data)
│
├── [Stage 1] Train SOM                          (unchanged)
│
├── [Stage 2] k-Selection                        ← redesigned
│   ├── 2a. KMeans on neurons k=2..max_k, record inertia only (no silhouette/CH)
│   ├── 2b. Elbow detection (Kneedle) on inertia curve → top-5 candidates (deduplicated)
│   └── 2c. Gap statistic on subsampled raw data for each top-5 → best_k
│
├── [Stage 3] KMeans on raw data with best_k     (unchanged, 5 restarts)
│
├── [Stage 4] DenSOM comparison                  ← speed fixes
│   ├── smooth_hits: separable 1D Gaussian + truncated kernel (radius = ceil(3σ))
│   └── sigma: Silverman's rule (default) / Scott's rule (opt-in), from data stats
│
└── [Stage 5] Pick winner                        (unchanged — silhouette used here only)
```

Note: The old per-k sil+CH combined score is removed entirely from Stage 2. Silhouette remains in Stage 5 (one evaluation per algorithm, not per k).

## New Public API (backward-compatible)

```rust
pub enum SigmaMethod { Silverman, Scott }   // defined in src/core/mod.rs

AutoSomBuilder::sigma_method(SigmaMethod)   // default: Silverman
AutoSomBuilder::gap_n_refs(usize)           // default: 10
```

`SigmaMethod` is defined in `src/core/mod.rs` (re-exported from the crate root) to avoid a circular dependency between `autosom.rs` (which imports `densom`) and `densom.rs`.

## Mathematical Formulas

### 2b. Elbow Detection (Kneedle)

Normalize the inertia curve to unit range, then find the k with maximum second difference (interior points only — requires k-1 and k+1 to both exist in the sweep):

```
W_norm(k) = (W(k) - W_min) / (W_max - W_min)
Δ²W(k)   = W_norm(k-1) - 2·W_norm(k) + W_norm(k+1)   for k in [3, max_k-1]
k_elbow  = argmax Δ²W(k)
```

**Fallbacks** (applied in order):
1. If the second-difference array is empty (sweep has fewer than 3 points, i.e., `max_k < 3`), set `k_elbow = 2` directly.
2. If `max(Δ²W) < 1e-9` (inertia curve is effectively linear — no elbow), set `k_elbow = 2`.

**Top-5 candidates**: `{ k_elbow-2, k_elbow-1, k_elbow, k_elbow+1, k_elbow+2 }`, each clamped to `[2, max_k]`, then **deduplicated** (remove repeated values). Result is a sorted Vec of unique k values, length 1–5.

### 2c. Gap Statistic (Tibshirani et al. 2001)

Applied to the **subsample** used for evaluation (max 2000 points, same as Stage 5), not the full dataset, to bound cost for large datasets like `scale_50k`.

For each candidate k (sorted ascending), run KMeans on the subsample, compute log within-cluster dispersion W_k. Compare against B=10 reference datasets drawn uniform in the per-dimension bounding box of the subsample:

```
Gap(k)  = (1/B) Σ_b log W_k^(ref_b)  −  log W_k
SE(k)   = sqrt((1/B) Σ_b (log W_k^(ref_b) − E*[log W_k^ref])²) · sqrt(1 + 1/B)
```

**Criterion** (applied to the sorted candidate list, stopping at the second-to-last element since k+1 must be available):

```
best_k = smallest k in candidates[0..len-1] where Gap(k) ≥ Gap(k_next) − SE(k_next)
```

where `k_next` is the next element in the sorted candidate list (not necessarily k+1 in the integers — the candidate list may have gaps, e.g., `{3, 5, 7}` after clamping and deduplication).

**Fallbacks** (applied in order):
1. If no candidate satisfies the criterion → use candidate with maximum `Gap(k)`.
2. If all KMeans fits fail for all candidates → use `k_elbow`.

### 4a. Separable + Truncated smooth_hits

The current implementation computes a full 2D Gaussian in one pass: O(m²n²).

The new implementation decomposes into two 1D passes, truncated at radius `r = ceil(3σ)`:

```
Row pass:    out_h[i, j] = Σ_{|dj|≤r}  hits[i, j+dj] · exp(−dj²/(2σ²))
Column pass: out[i, j]   = Σ_{|di|≤r}  out_h[i+di, j] · exp(−di²/(2σ²))
```

Boundary: values outside the grid are treated as 0 (zero-padding).
Complexity: O(m·n·(2r+1)) per pass, two passes = O(m·n·(4r+2)).

Speedup examples vs the O(m²n²) baseline:
- σ=1.6, 10×10 grid: r=5, new=2×100×11=2200 ops, old=10000 ops → ~4.5× faster
- σ=3, 20×20 grid: r=9, new=2×400×19=15200 ops, old=160000 ops → ~10.5× faster

Output type and shape: identical to current (`Array1<f64>` of length `m*n`, row-major). The mathematical result is identical to the full 2D Gaussian (Gaussian kernel is separable).

### 4b. Adaptive Sigma (Silverman / Scott)

For multivariate data (d dimensions), compute per-dimension statistics, then take the mean:

```
σ_data = mean over d of std_dev(dimension j)
IQR    = mean over d of (Q75(dimension j) − Q25(dimension j))
```

Then apply the bandwidth rule:

```
Scott:     h = n^(−1/(d+4)) · σ_data
Silverman: h = 0.9 · min(σ_data, IQR/1.34) · n^(−0.2)
```

Note: for high-dimensional data (d >> 4), the Scott exponent `−1/(d+4) → 0`, making `n^(−1/(d+4)) ≈ 1`. In this case `h ≈ σ_data`, which can be large. The clamp below handles this.

The result `h` is clamped to `[0.5, grid_side/2]` where `grid_side = max(m, n)`.

This `h` is passed as `smooth_sigma` to `DenSom::set_smooth_sigma` before `refit_density`.

## Data Flow Details

- **Sampling**: `build_sample_idx` remains at 2000 points but switches to random stratified sampling (one draw per KMeans label) to eliminate the evenly-spaced bias. The same sample is reused for gap statistic evaluation and Stage 5 silhouette.
- **Reference data for gap statistic**: uniform random points within `[data_min_d, data_max_d]` per dimension, generated with a fixed per-call seed derived from the sample size for reproducibility.
- **smooth_hits output**: unchanged shape (`Array1<f64>` of length `m*n`) — DenSOM internals after this function are untouched.
- **Silhouette in Stage 5**: still computed once per algorithm (KMeans result, DenSOM result) as in the current implementation. Not computed during Stage 2.

## Error Handling

| Condition | Behaviour |
|-----------|-----------|
| Gap criterion not met by any candidate | Use candidate with maximum `Gap(k)` |
| All gap statistic KMeans fits fail | Fall back to `k_elbow` |
| `max(Δ²W) < 1e-9` (linear inertia curve) | Set `k_elbow = 2` |
| Candidate clamping produces duplicates | Deduplicate; result may have fewer than 5 candidates |
| Silverman/Scott sigma outside `[0.5, grid_side/2]` | Clamp to that range |
| Gap statistic subsample has fewer than 2·k points | Skip that candidate k |
| `max_k < 3` (sweep has fewer than 3 points) | Set `k_elbow = 2` directly (empty second-difference array) |

## Testing Strategy

| Test | Assertion |
|------|-----------|
| `elbow_detect_known_knee` | Synthetic inertia with knee at k=5 → returns 5 |
| `elbow_detect_linear_fallback` | Perfectly linear inertia curve → returns 2 |
| `elbow_candidates_clamped_dedup` | k_elbow=max_k → candidates are deduplicated, length < 5 |
| `elbow_empty_sweep_fallback` | max_k=2 (2×2 grid) → k_elbow=2 without panicking on empty array |
| `gap_statistic_two_blobs` | Two well-separated blobs, fixed RNG seed → gap selects k=2 |
| `smooth_hits_separable_matches_full_square` | 5×5 grid, max diff < 1e-10 vs O(m²n²) |
| `smooth_hits_separable_matches_full_rect` | 4×6 grid, max diff < 1e-10 (catches m≠n off-by-one) |
| `silverman_sigma_formula` | d=2, n=100, known σ/IQR → assert output matches hand-computed value |
| `autosom_blobs_finds_two_clusters` | Existing test must still pass |
| Benchmark regression (s1) | AutoSom ARI on `s1` (true k=15) must be ≥ 0.90 |
| Benchmark regression (a1) | AutoSom ARI on `a1` (true k=20) must stay ≥ 0.90 |

Note: `gap_statistic_two_blobs` must use a fixed RNG seed (e.g., `42`) to be deterministic in CI.

## Files to Change

| File | Change |
|------|--------|
| `src/core/mod.rs` | Add `SigmaMethod` enum (Silverman / Scott) |
| `src/lib.rs` | Add `pub use core::SigmaMethod;` re-export |
| `src/core/autosom.rs` | Replace k-sweep body with elbow detection + gap statistic; add `sigma_method` and `gap_n_refs` fields to builder; compute adaptive sigma and pass to DenSOM; replace `build_sample_idx` evenly-spaced logic with stratified random sampling |
| `src/core/densom.rs` | Replace `smooth_hits` with separable+truncated implementation; `set_smooth_sigma` already exists — no interface change needed |
