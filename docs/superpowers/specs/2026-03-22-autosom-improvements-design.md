# AutoSom Improvements: Two-Stage k-Selection + Speed Fixes

**Date:** 2026-03-22
**Status:** Approved

## Problem Statement

AutoSom has two interrelated weaknesses:

1. **k-detection quality** — on the S-set benchmarks (true k=15), AutoSom detects k=4, 8, or 20 due to a scoring function (`norm_sil + 0.5 * norm_ch`) that is noisy over the full k=2..50 sweep.
2. **Speed** — AutoSom runs ~1.4s vs KMeans at 8–50ms. The main bottlenecks are `smooth_hits` at O(mn²) and silhouette computed O(n²) for each k in the sweep.

## Architecture

The overall pipeline shape is preserved. Only Stage 2 (k-selection) and Stage 4 (DenSOM smoothing) change.

```
fit_predict(data)
│
├── [Stage 1] Train SOM                        (unchanged)
│
├── [Stage 2] k-Selection                      ← redesigned
│   ├── 2a. KMeans on neurons k=2..max_k, record inertia only (no silhouette)
│   ├── 2b. Elbow detection (Kneedle) on inertia curve → top-5 candidates
│   ├── 2c. Gap statistic on raw data sample for each top-5 → best_k
│   └── 2d. Davies-Bouldin verification on best_k (replaces old sil+CH combined score)
│
├── [Stage 3] KMeans on raw data with best_k   (unchanged, 5 restarts)
│
├── [Stage 4] DenSOM comparison                ← speed fixes
│   ├── smooth_hits: separable 1D Gaussian + truncated kernel (radius = ceil(3σ))
│   └── sigma: Silverman's rule (default) / Scott's rule (opt-in)
│
└── [Stage 5] Pick winner                      (unchanged logic)
```

## New Public API (backward-compatible)

```rust
pub enum SigmaMethod { Silverman, Scott }

AutoSomBuilder::sigma_method(SigmaMethod)    // default: Silverman
AutoSomBuilder::gap_n_refs(usize)            // default: 10
```

## Mathematical Formulas

### 2b. Elbow Detection (Kneedle)

Normalize the inertia curve to unit range, then find the k with maximum second difference:

```
W_norm(k) = (W(k) - W_min) / (W_max - W_min)
Δ²W(k)   = W_norm(k-1) - 2·W_norm(k) + W_norm(k+1)
k_elbow  = argmax Δ²W(k)
```

Top-5 candidates = { k_elbow-2, k_elbow-1, k_elbow, k_elbow+1, k_elbow+2 }, clamped to [2, max_k].

### 2c. Gap Statistic (Tibshirani et al. 2001)

For each candidate k, compute log within-cluster dispersion and compare to B=10 reference datasets drawn uniform in the data bounding box:

```
Gap(k)  = E*[log W_k^ref] - log W_k
SE(k)   = std(log W_k^ref) · sqrt(1 + 1/B)
best_k  = smallest k where Gap(k) ≥ Gap(k+1) - SE(k+1)
```

Fallback: if no k satisfies the criterion (flat data), use the k with maximum Gap(k). If gap statistic yields no valid candidate, fall back to k_elbow.

### 4a. Separable + Truncated smooth_hits

Decompose the 2D Gaussian into two 1D passes, truncated at radius `r = ceil(3σ)`:

```
Row pass:    out_h[i,j] = Σ_{|dj|≤r}  hits[i, j+dj] · exp(-dj²/(2σ²))
Column pass: out[i,j]   = Σ_{|di|≤r}  out_h[i+di, j] · exp(-di²/(2σ²))
```

Complexity: O(mn · 2r) vs O(mn²). For σ=1.6 on a 10×10 grid: 10× faster. For σ=3 on a 20×20 grid: ~20× faster. Output is mathematically identical to the full 2D Gaussian (separability of Gaussian kernel).

### 4b. Adaptive Sigma (Silverman / Scott)

Computed from the training data before DenSOM finalization:

```
Scott:     h = n^(-1/(d+4)) · σ_data
Silverman: h = 0.9 · min(σ_data, IQR/1.34) · n^(-0.2)
```

Where `σ_data` is the mean per-dimension standard deviation and `IQR` is the interquartile range. The result `h` is clamped to `[0.5, grid_side/2]` to prevent degenerate smoothing.

## Data Flow Details

- **Sampling**: `build_sample_idx` remains at 2000 points but switches to random stratified sampling (one draw per KMeans label) to eliminate the evenly-spaced bias.
- **Reference data for gap statistic**: uniform random points within `[data_min_d, data_max_d]` per dimension — no new dependencies required.
- **smooth_hits output shape**: unchanged (`Array1<f64>` of length `m*n`) — DenSOM internals after this function are untouched.

## Error Handling

| Condition | Behaviour |
|-----------|-----------|
| Gap statistic finds no k meeting criterion | Fall back to k_elbow |
| Elbow Δ²W all-equal (convex inertia curve) | Fall back to k=2 |
| Silverman/Scott sigma out of bounds | Clamp to `[0.5, grid_side/2]` |
| Gap statistic KMeans fit fails for a candidate k | Skip that candidate, continue |

## Testing Strategy

| Test | Assertion |
|------|-----------|
| `elbow_detect_known_knee` | Synthetic inertia with knee at k=5 → function returns 5 |
| `gap_statistic_two_blobs` | Two well-separated blobs → gap selects k=2 |
| `smooth_hits_separable_matches_full` | 5×5 grid, compare separable vs O(mn²) output, max diff < 1e-10 |
| `silverman_sigma_formula` | Known n/d/σ dataset, assert output matches hand-computed value |
| `autosom_blobs_finds_two_clusters` | Existing test must still pass |
| Benchmark regression | AutoSom ARI on `a1` (true k=20) must stay ≥ 0.90 |

## Files to Change

| File | Change |
|------|--------|
| `src/core/autosom.rs` | Replace k-sweep body with elbow+gap; add `SigmaMethod` enum; add `gap_n_refs` field; compute adaptive sigma; pass sigma to DenSOM |
| `src/core/densom.rs` | Replace `smooth_hits` with separable+truncated implementation; add `SigmaMethod`-aware sigma computation in `finalize` |
