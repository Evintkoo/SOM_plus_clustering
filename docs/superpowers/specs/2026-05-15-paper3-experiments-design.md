# Paper 3 Experiment Rerun — Design Spec

**Date:** 2026-05-15  
**Status:** Approved  
**Goal:** Re-run all 24-dataset benchmarks with the fully Rayon-parallelized codebase, add a three-way CPU-serial / CPU-Rayon / Metal-GPU timing comparison, add GPU quality runs, regenerate all statistics and figures, and update the LaTeX paper.

---

## 1. Architecture

Four sequential stages. Each stage produces artifacts consumed by the next.

```
Stage 1: benchmark_quality.rs (CPU + GPU)
    → rust_metrics.json
    → <name>_{som,km,densom,auto}_labels.csv
    → <name>_som_gpu_labels.csv

Stage 2: benchmark_timing.rs (serial / Rayon / Metal)
    → docs/paper3/timing_comparison.json

Stage 3: Python pipeline
    evaluate.py       → experiments/benchmark/results/full_results.json
    run_experiments.py → docs/paper3/figures/, stats tables

Stage 4: LaTeX update
    som_tsk_paper_v3.tex — timing table, ARI table, Friedman numbers, Rayon paragraph
```

---

## 2. Stage 1 — Quality Benchmark (`examples/benchmark_quality.rs`)

**Changes required:**

- Add a second SOM-TSK run per dataset using `Backend::Metal`:
  ```rust
  som.set_backend(Backend::Metal);
  let gpu_result = run_som(&som, &data, k);
  ```
- Write GPU cluster labels to `<name>_som_gpu_labels.csv` alongside the existing CPU label files.
- Record GPU wall-clock time and ARI proxy in `rust_metrics.json` under a `"som_gpu"` key.
- The CPU SOM run uses the default backend (Rayon-parallelized CPU).
- All other algorithms (KMeans, DenSOM, AutoSOM) remain CPU-only — adding GPU variants for them is out of scope.

**Determinism:** Metal GPU results may differ from CPU due to floating-point ordering. This is acceptable; both runs are reported.

**Output schema addition to `rust_metrics.json`:**
```json
{
  "dataset_name": {
    "som":     { "time_ms": ..., "inertia": ... },
    "som_gpu": { "time_ms": ..., "inertia": ... },
    "km":      { ... },
    "densom":  { ... },
    "auto":    { ... }
  }
}
```

---

## 3. Stage 2 — Timing Benchmark (`examples/benchmark_timing.rs`)

New example binary; does not run quality metrics.

**Three configurations measured per dataset:**

| Config | Backend | Rayon threads | Description |
|--------|---------|---------------|-------------|
| Serial | CPU | 1 (via `RAYON_NUM_THREADS=1`) | Baseline single-threaded |
| Rayon  | CPU | All cores (default) | Fully parallelized CPU |
| Metal  | GPU | n/a | Apple Metal compute shaders |

**Measurement:** Wall-clock time per `predict_clustered_refined()` call; median of 5 runs; warmup run discarded.

**Output:** `docs/paper3/timing_comparison.json`
```json
{
  "dataset_name": {
    "n_samples": ...,
    "n_features": ...,
    "k": ...,
    "serial_ms":  ...,
    "rayon_ms":   ...,
    "metal_ms":   ...
  }
}
```

**Serial mode implementation:** Set `rayon::ThreadPoolBuilder::new().num_threads(1).build_global()` in a sub-process or use a local pool via `ThreadPool::install`.

---

## 4. Stage 3 — Python Pipeline

### evaluate.py
- Reads `<name>_som_gpu_labels.csv` in addition to existing label files.
- Computes ARI/NMI/FMI for `som_gpu` against ground truth.
- Writes updated `full_results.json` with `"som_gpu"` column.

### run_experiments.py
- Adds `som_gpu` to the per-dataset comparison table.
- Generates:
  - `fig_ari_comparison.pdf` — bar chart: SOM-TSK CPU vs GPU vs KMeans++ vs DenSOM vs AutoSOM
  - `fig_timing_speedup.pdf` — speedup chart from `timing_comparison.json` (Rayon/serial and Metal/serial ratios vs dataset size)
  - `fig_cd_diagram.pdf` — Nemenyi critical difference diagram (existing; updated with new numbers)
  - Updated Friedman test p-value and Wilcoxon signed-rank p-values

---

## 5. Stage 4 — LaTeX Paper Update (`docs/paper3/som_tsk_paper_v3.tex`)

**Changes:**

1. **Table 5 (per-phase profiling):** Replace with new three-way timing table (serial / Rayon / Metal) showing median ms and speedup ratios for a representative 5-dataset subset.

2. **Table 3 (ARI comparison):** Verify all 24-dataset ARI values match `full_results.json`; add `SOM-TSK (GPU)` column.

3. **Table 8 (all-algorithm summary):** Add GPU row; update win/tie/loss counts.

4. **Section on Rayon parallelism:** ~150-word paragraph describing the parallel migration — `assign()`, `update()`, `compute_inertia()`, E-step, silhouette, SOM-PlusPlus init — and measured speedup on Apple M2 Pro (12 performance + 4 efficiency cores).

5. **Update abstract and conclusion** to reflect the validated results.

---

## 6. Error Handling

- If Metal backend fails on a dataset (unsupported shape, etc.), log the error and skip that dataset's GPU row — do not abort the benchmark.
- Python pipeline: missing label CSVs cause a warning, not a crash; that dataset's GPU column shows `NaN`.

---

## 7. Success Criteria

- `cargo run --example benchmark_quality --release` completes without panics for all 24 datasets.
- `cargo run --example benchmark_timing --release` completes; `timing_comparison.json` contains all 24 entries.
- `full_results.json` contains `som_gpu` ARI for ≥ 20 datasets.
- `run_experiments.py` produces all figures without error.
- Paper compiles with `pdflatex` without errors.
- SOM-TSK CPU ARI results are consistent with prior run (mean ARI ~0.715, 6+ wins vs KMeans++).
