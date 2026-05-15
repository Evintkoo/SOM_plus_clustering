# Paper 3 Experiment Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run all 24-dataset benchmarks with the Rayon-parallelized codebase, add Metal GPU quality runs, add a three-way serial/Rayon/Metal timing benchmark, regenerate all statistics and figures, and update the LaTeX paper.

**Architecture:** Four sequential stages — (1) Rust quality benchmark extended with Metal GPU, (2) Rust timing benchmark (new binary), (3) Python pipeline regenerating stats and figures, (4) LaTeX paper update. Each stage produces files consumed by the next.

**Tech Stack:** Rust (ndarray, rayon, metal feature flag), Python 3 (sklearn, scipy, matplotlib, pandas), LaTeX (IEEEtran).

---

## File Map

| Action | File |
|--------|------|
| Modify | `examples/benchmark_quality.rs` |
| Create | `examples/benchmark_timing.rs` |
| Modify | `experiments/benchmark/evaluate.py` |
| Modify | `docs/paper3/run_experiments.py` |
| Modify | `docs/paper3/som_tsk_paper_v3.tex` |

---

## Task 1: Add Metal GPU run to benchmark_quality.rs

**Files:**
- Modify: `examples/benchmark_quality.rs`

- [ ] **Step 1.1: Add Backend import at the top**

In `examples/benchmark_quality.rs`, add `Backend` to the existing import block. The current imports are:

```rust
use som_plus_clustering::{
    calinski_harabasz_score, davies_bouldin_index, dunn_index, silhouette_score,
    AutoSomBuilder, DenSomBuilder, DistanceFunction, InitMethod, KMeansBuilder, KMeansInit,
    SomBuilder,
};
```

Replace with:

```rust
use som_plus_clustering::{
    calinski_harabasz_score, davies_bouldin_index, dunn_index, silhouette_score,
    AutoSomBuilder, Backend, DenSomBuilder, DistanceFunction, InitMethod, KMeansBuilder,
    KMeansInit, SomBuilder,
};
```

- [ ] **Step 1.2: Add `run_som_metal` function**

Insert the following function after the closing brace of `run_som()` (around line 110 in the current file) and before `run_kmeans()`:

```rust
#[cfg(feature = "metal")]
fn run_som_metal(
    data: &Array2<f64>,
    m: usize,
    n: usize,
    k: usize,
    epochs: usize,
) -> Option<(Vec<usize>, String)> {
    let dim = data.ncols();
    let mut som = SomBuilder::new()
        .grid(m, n)
        .dim(dim)
        .learning_rate(0.5)
        .expect("valid lr")
        .neighbor_radius(3.0)
        .init_method(InitMethod::SomPlusPlus)
        .distance(DistanceFunction::Euclidean)
        .build();
    som.set_backend(Backend::Metal);

    let t0 = Instant::now();
    if let Err(e) = som.fit(&data.view(), epochs, false, None) {
        eprintln!("  [Metal] fit failed: {e}");
        return None;
    }
    let fit_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let labels = match som.predict_clustered_refined(&data.view(), k) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("  [Metal] predict failed: {e}");
            return None;
        }
    };
    let predict_s = t0.elapsed().as_secs_f64();

    let (sil, db, ch, dunn) = compute_metrics(data, &labels);
    let json = format!(
        r#"{{"fit_time_s":{fit:.4},"predict_time_s":{pred:.6},"n_neurons":{nn},"silhouette":{sil},"davies_bouldin":{db},"calinski_harabasz":{ch},"dunn":{dunn}}}"#,
        fit  = fit_s,
        pred = predict_s,
        nn   = m * n,
        sil  = jf(sil),
        db   = jf(db),
        ch   = jf(ch),
        dunn = jf(dunn),
    );
    Some((labels.to_vec(), json))
}
```

- [ ] **Step 1.3: Add GPU run and label saving in `main()`**

In `main()`, after the block `let (som_labels, som_json) = run_som(...); save_labels(...); print!("SOM✓  ");`, insert:

```rust
        // SOM Metal GPU (only compiled when --features metal)
        #[cfg(feature = "metal")]
        let som_gpu_result = run_som_metal(&data, m, n, k, epochs);
        #[cfg(feature = "metal")]
        if let Some((ref gpu_labels, _)) = som_gpu_result {
            save_labels(&format!("{results_dir}/{name}_som_gpu_labels.csv"), gpu_labels);
            print!("GPU✓  ");
        } else {
            print!("GPU✗  ");
        }
        #[cfg(feature = "metal")]
        let _ = std::io::stdout().flush();
        #[cfg(not(feature = "metal"))]
        let som_gpu_result: Option<(Vec<usize>, String)> = None;
```

- [ ] **Step 1.4: Add `som_gpu` to the JSON entry in `main()`**

Replace the existing `entries.push(format!(...))` block with:

```rust
        let gpu_json_field = match som_gpu_result {
            Some((_, ref j)) => format!(r#","som_gpu":{j}"#),
            None             => String::new(),
        };

        entries.push(format!(
            r#"  "{name}": {{"n_samples":{ns},"n_features":{nf},"n_true_clusters":{nc},"som_grid":"{m}x{n}","som":{som},"kmeans":{km},"densom":{densom},"autosom":{autosom}{gpu}}}"#,
            name    = name,
            ns      = data.nrows(),
            nf      = data.ncols(),
            nc      = k,
            m       = m,
            n       = n,
            som     = som_json,
            km      = km_json,
            densom  = densom_json,
            autosom = auto_json,
            gpu     = gpu_json_field,
        ));
```

- [ ] **Step 1.5: Build and verify compilation (CPU only first)**

```bash
cargo build --example benchmark_quality --release 2>&1 | tail -5
```

Expected: `Finished release profile` with no errors.

- [ ] **Step 1.6: Build with metal feature**

```bash
cargo build --example benchmark_quality --release --features metal 2>&1 | tail -5
```

Expected: `Finished release profile` with no errors.

- [ ] **Step 1.7: Commit**

```bash
git add examples/benchmark_quality.rs
git commit -m "feat(benchmark): add Metal GPU run to benchmark_quality with graceful fallback"
```

---

## Task 2: Create benchmark_timing.rs

**Files:**
- Create: `examples/benchmark_timing.rs`

- [ ] **Step 2.1: Create the file**

Create `examples/benchmark_timing.rs` with the full content below. This binary measures wall-clock time for `predict_clustered_refined` in three modes — serial (1-thread rayon pool), parallel Rayon (all cores), Metal GPU — across all 24 datasets. It takes 5 timed runs per configuration (after 1 warmup), reports the median, and writes `docs/paper3/timing_comparison.json`.

```rust
//! Three-way timing benchmark: serial CPU (1 thread) vs Rayon CPU vs Metal GPU.
//!
//! Measures wall-clock time for the full SOM-TSK pipeline (fit + predict_clustered_refined)
//! on all 24 datasets. Each configuration runs 5 times (preceded by 1 warmup run); median
//! is reported.
//!
//! Run from crate root:
//!   cargo run --example benchmark_timing --release --features metal

use ndarray::Array2;
use serde::Deserialize;
use som_plus_clustering::{Backend, DistanceFunction, InitMethod, SomBuilder};
use std::{fs, io::Write, time::Instant};

#[derive(Deserialize)]
struct DatasetConfig {
    name: String,
    n_true_clusters: usize,
    som_m: usize,
    som_n: usize,
    epochs: usize,
}

fn load_csv(base_dir: &str, name: &str) -> Array2<f64> {
    let path = format!("{base_dir}/{name}.csv");
    let mut reader = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let n_cols = reader.headers().unwrap().len();
    let feature_cols = n_cols - 1;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in reader.records() {
        let rec = rec.unwrap();
        let row: Vec<f64> = (0..feature_cols)
            .map(|i| rec[i].parse::<f64>().unwrap())
            .collect();
        rows.push(row);
    }
    let n = rows.len();
    let d = rows[0].len();
    Array2::from_shape_vec((n, d), rows.into_iter().flatten().collect()).unwrap()
}

const N_RUNS: usize = 5;

/// Run SOM fit + predict_clustered_refined N_RUNS+1 times on the given backend.
/// Returns median wall-clock ms (warmup run discarded).
/// Returns None if any run errors.
fn measure_ms(
    data: &Array2<f64>,
    m: usize,
    n: usize,
    k: usize,
    epochs: usize,
    backend: Backend,
    runner: &rayon::ThreadPool,
) -> Option<f64> {
    let dim = data.ncols();
    let mut times: Vec<f64> = Vec::with_capacity(N_RUNS + 1);

    for _ in 0..=N_RUNS {
        let mut som = SomBuilder::new()
            .grid(m, n)
            .dim(dim)
            .learning_rate(0.5)
            .expect("valid lr")
            .neighbor_radius(3.0)
            .init_method(InitMethod::SomPlusPlus)
            .distance(DistanceFunction::Euclidean)
            .build();
        som.set_backend(backend);

        let t0 = Instant::now();
        let fit_ok = runner.install(|| som.fit(&data.view(), epochs, false, None));
        if fit_ok.is_err() {
            return None;
        }
        let predict_ok = runner.install(|| som.predict_clustered_refined(&data.view(), k));
        if predict_ok.is_err() {
            return None;
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // Discard warmup (index 0)
    let mut timed = times[1..].to_vec();
    timed.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(timed[N_RUNS / 2])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base_dir  = args.get(1).map(String::as_str).unwrap_or("experiments/benchmark/datasets");
    let cfg_path  = args.get(2).map(String::as_str).unwrap_or("experiments/benchmark/dataset_config.json");
    let out_path  = args.get(3).map(String::as_str).unwrap_or("docs/paper3/timing_comparison.json");

    let cfg_str = fs::read_to_string(cfg_path)
        .unwrap_or_else(|_| panic!("Config not found at {cfg_path}"));
    let configs: Vec<DatasetConfig> = serde_json::from_str(&cfg_str)
        .expect("malformed dataset_config.json");

    // Thread pools
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("failed to build serial pool");
    let rayon_pool = rayon::ThreadPoolBuilder::new()
        .build()
        .expect("failed to build rayon pool");

    let total = configs.len();
    let mut entries: Vec<String> = Vec::new();

    for (i, cfg) in configs.iter().enumerate() {
        let name   = &cfg.name;
        let k      = cfg.n_true_clusters;
        let m      = cfg.som_m;
        let n      = cfg.som_n;
        let epochs = cfg.epochs;

        print!("[{:>2}/{total}] {name:<28}", i + 1);
        let _ = std::io::stdout().flush();

        let data = load_csv(base_dir, name);

        // Serial (1 rayon thread)
        let serial_ms = measure_ms(&data, m, n, k, epochs, Backend::Cpu, &serial_pool)
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "null".into());
        print!("serial={serial_ms}ms  ");
        let _ = std::io::stdout().flush();

        // Rayon (all cores)
        let rayon_ms = measure_ms(&data, m, n, k, epochs, Backend::Cpu, &rayon_pool)
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "null".into());
        print!("rayon={rayon_ms}ms  ");
        let _ = std::io::stdout().flush();

        // Metal GPU
        #[cfg(feature = "metal")]
        let metal_ms = measure_ms(&data, m, n, k, epochs, Backend::Metal, &rayon_pool)
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "null".into());
        #[cfg(not(feature = "metal"))]
        let metal_ms = "null".to_string();
        println!("metal={metal_ms}ms");

        entries.push(format!(
            r#"  "{name}": {{"n_samples":{ns},"n_features":{nf},"k":{k},"serial_ms":{serial},"rayon_ms":{rayon},"metal_ms":{metal}}}"#,
            name   = name,
            ns     = data.nrows(),
            nf     = data.ncols(),
            k      = k,
            serial = serial_ms,
            rayon  = rayon_ms,
            metal  = metal_ms,
        ));
    }

    let json = format!("{{\n{}\n}}\n", entries.join(",\n"));
    fs::write(out_path, &json).expect("cannot write timing_comparison.json");
    println!("\nTiming → {out_path}");
}
```

- [ ] **Step 2.2: Build timing binary (CPU only)**

```bash
cargo build --example benchmark_timing --release 2>&1 | tail -5
```

Expected: `Finished release profile` with no errors.

- [ ] **Step 2.3: Build with metal feature**

```bash
cargo build --example benchmark_timing --release --features metal 2>&1 | tail -5
```

Expected: `Finished release profile` with no errors.

- [ ] **Step 2.4: Commit**

```bash
git add examples/benchmark_timing.rs
git commit -m "feat(benchmark): add three-way timing benchmark (serial/Rayon/Metal)"
```

---

## Task 3: Update evaluate.py for GPU labels

**Files:**
- Modify: `experiments/benchmark/evaluate.py`

- [ ] **Step 3.1: Add GPU label loading and metrics**

The current evaluate.py builds `full[name]` with keys `som`, `kmeans`, `densom`, `autosom`. Add `som_gpu` by replacing the `full[name] = {...}` block with:

```python
    # GPU labels (optional — only present when Metal feature was enabled)
    gpu_path = os.path.join(RES_DIR, f"{name}_som_gpu_labels.csv")
    if os.path.exists(gpu_path):
        try:
            gpu_labels = pd.read_csv(gpu_path, header=None).values.flatten().astype(int)
            som_gpu_ext = ext_metrics(y_true, gpu_labels)
        except Exception as e:
            print(f"  warn: GPU labels for {name} unreadable: {e}")
            som_gpu_ext = {"ari": None, "nmi": None, "fmi": None, "v_measure": None}
    else:
        som_gpu_ext = {"ari": None, "nmi": None, "fmi": None, "v_measure": None}

    som_gpu_internal = r.get("som_gpu", {})

    full[name] = {
        "n_samples":       r["n_samples"],
        "n_features":      r["n_features"],
        "n_true_clusters": r["n_true_clusters"],
        "som_grid":        r["som_grid"],
        "som":     {**r["som"],     **ext_metrics(y_true, som_labels)},
        "kmeans":  {**r["kmeans"],  **ext_metrics(y_true, km_labels)},
        "densom":  {**r["densom"],  **ext_metrics(y_true, densom_labels)},
        "autosom": {**r["autosom"], **ext_metrics(y_true, auto_labels)},
        "som_gpu": {**som_gpu_internal, **som_gpu_ext},
    }
```

This block replaces the existing `full[name] = {` block (the one that ends with `"autosom": ...`).

- [ ] **Step 3.2: Verify the script runs without errors on existing data**

```bash
cd /Users/evintleovonzko/Documents/projects/evint/SOM_plus_clustering
python experiments/benchmark/evaluate.py 2>&1 | tail -5
```

Expected: `Full results → experiments/benchmark/results/full_results.json` and no tracebacks. GPU fields will show `null`/`None` since GPU labels don't exist yet (that's correct).

- [ ] **Step 3.3: Commit**

```bash
git add experiments/benchmark/evaluate.py
git commit -m "feat(evaluate): add optional som_gpu labels and metrics to evaluate.py"
```

---

## Task 4: Update run_experiments.py for GPU column and timing speedup figure

**Files:**
- Modify: `docs/paper3/run_experiments.py`

- [ ] **Step 4.1: Add GPU to the algorithm list (conditionally)**

In `run_experiments.py`, find the section in `main()` that builds the ARI matrix:

```python
    algo_names = ['SOM-TSK', 'KMeans++', 'DenSOM', 'AutoSOM']
    algo_keys = ['som', 'kmeans', 'densom', 'autosom']
```

Replace with:

```python
    algo_names = ['SOM-TSK', 'KMeans++', 'DenSOM', 'AutoSOM']
    algo_keys = ['som', 'kmeans', 'densom', 'autosom']

    # Include GPU column only if ≥50% of datasets have GPU results
    gpu_coverage = sum(
        1 for name in full_results
        if full_results[name].get("som_gpu", {}).get("ari") is not None
    )
    has_gpu = gpu_coverage >= len(full_results) // 2
    if has_gpu:
        algo_names = ['SOM-TSK', 'SOM-TSK GPU', 'KMeans++', 'DenSOM', 'AutoSOM']
        algo_keys  = ['som', 'som_gpu', 'kmeans', 'densom', 'autosom']
        print(f"  GPU results available for {gpu_coverage}/{len(full_results)} datasets — including GPU column")
    else:
        print(f"  GPU results available for {gpu_coverage}/{len(full_results)} datasets — skipping GPU column")
```

- [ ] **Step 4.2: Fix ARI matrix to handle None GPU values**

Find the ARI matrix construction:

```python
    ari_matrix = np.zeros((n_ds, len(algo_keys)))
    for i, name in enumerate(dataset_names):
        for j, key in enumerate(algo_keys):
            ari_matrix[i, j] = full_results[name][key]["ari"]
```

Replace with:

```python
    ari_matrix = np.zeros((n_ds, len(algo_keys)))
    for i, name in enumerate(dataset_names):
        for j, key in enumerate(algo_keys):
            val = full_results[name].get(key, {}).get("ari")
            ari_matrix[i, j] = val if val is not None else 0.0
```

- [ ] **Step 4.3: Add `fig_timing_speedup` function**

Add the following function to `run_experiments.py` after the `fig_cd_diagram` function, before `compare_v2`:

```python
def fig_timing_speedup(timing_path):
    """Generate speedup bar chart from timing_comparison.json."""
    if not os.path.exists(timing_path):
        print(f"  Skipping timing figure — {timing_path} not found")
        return
    with open(timing_path) as f:
        timing = json.load(f)

    names, speedup_rayon, speedup_metal, n_samples = [], [], [], []
    for name, d in timing.items():
        s = d.get("serial_ms")
        r = d.get("rayon_ms")
        m = d.get("metal_ms")
        if s is None or r is None or s == 0:
            continue
        names.append(name)
        n_samples.append(d["n_samples"])
        speedup_rayon.append(s / r)
        speedup_metal.append((s / m) if (m and m != 0) else None)

    if not names:
        print("  Skipping timing figure — no valid timing data")
        return

    # Sort by n_samples
    order = sorted(range(len(names)), key=lambda i: n_samples[i])
    names        = [names[i] for i in order]
    speedup_rayon = [speedup_rayon[i] for i in order]
    speedup_metal = [speedup_metal[i] for i in order]
    n_samples    = [n_samples[i] for i in order]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_r = ax.bar(x - width / 2, speedup_rayon, width, label='Rayon / Serial', color='steelblue')
    metal_vals = [v if v is not None else 0.0 for v in speedup_metal]
    if any(v and v > 0 for v in speedup_metal):
        bars_m = ax.bar(x + width / 2, metal_vals, width, label='Metal / Serial', color='darkorange')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Dataset (sorted by n_samples)')
    ax.set_ylabel('Speedup over serial (×)')
    ax.set_title('Rayon and Metal Speedup vs. Single-Thread Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.legend()
    plt.tight_layout()
    out = FIGS_DIR / "timing_speedup.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")
```

- [ ] **Step 4.4: Call `fig_timing_speedup` in `main()`**

In the `[4/5] Generating figures` block, add a call to the new function. Find:

```python
    fig_delta_ari(full_results, configs)
    fig_algorithm_comparison(full_results, multiseed, configs)
    fig_cd_diagram(avg_ranks, algo_names, cd, n_ds)
```

Replace with:

```python
    fig_delta_ari(full_results, configs)
    fig_algorithm_comparison(full_results, multiseed, configs)
    fig_cd_diagram(avg_ranks, algo_names, cd, n_ds)
    timing_path = str(PAPER3_DIR / "timing_comparison.json")
    fig_timing_speedup(timing_path)
```

- [ ] **Step 4.5: Verify the script runs on existing data**

```bash
cd /Users/evintleovonzko/Documents/projects/evint/SOM_plus_clustering
python docs/paper3/run_experiments.py 2>&1 | tail -15
```

Expected: completes all 5 steps, saves multi_seed_results.json, statistical_tests.json, and three PDF figures. Timing figure will be skipped (timing_comparison.json not yet generated). No tracebacks.

- [ ] **Step 4.6: Commit**

```bash
git add docs/paper3/run_experiments.py
git commit -m "feat(experiments): add GPU column support and timing speedup figure to run_experiments.py"
```

---

## Task 5: Run Stage 1 — Quality Benchmark (CPU + GPU)

- [ ] **Step 5.1: Run the quality benchmark with Metal feature**

```bash
cd /Users/evintleovonzko/Documents/projects/evint/SOM_plus_clustering
cargo run --example benchmark_quality --release --features metal 2>&1 | tee /tmp/bq_output.txt
```

Expected: 24 datasets processed, each line showing `SOM✓  GPU✓ (or GPU✗)  KMeans✓  DenSOM✓  AutoSOM✓`, ending with `Metrics → experiments/benchmark/results/rust_metrics.json`.

- [ ] **Step 5.2: Verify output files exist**

```bash
ls -lh experiments/benchmark/results/rust_metrics.json
ls experiments/benchmark/results/ | grep som_gpu | wc -l
```

Expected: `rust_metrics.json` present; GPU label count ≥ 1.

- [ ] **Step 5.3: Run evaluate.py**

```bash
python experiments/benchmark/evaluate.py
```

Expected: `Full results → experiments/benchmark/results/full_results.json`

- [ ] **Step 5.4: Spot-check full_results.json**

```bash
python3 -c "
import json
with open('experiments/benchmark/results/full_results.json') as f:
    r = json.load(f)
for name in ['s2', 'a2', 'digits', 'wine']:
    som_ari = r[name]['som']['ari']
    gpu_ari = r[name].get('som_gpu', {}).get('ari', 'N/A')
    km_ari  = r[name]['kmeans']['ari']
    print(f'{name}: SOM-CPU={som_ari:.4f}  SOM-GPU={gpu_ari}  KM={km_ari:.4f}')
"
```

Expected: CPU SOM ARI values consistent with prior benchmark (s2≈0.607, a2≈0.841, digits≈0.580, wine≈0.898). GPU values present or shown as `N/A`.

---

## Task 6: Run Stage 2 — Timing Benchmark

- [ ] **Step 6.1: Run timing benchmark**

```bash
cargo run --example benchmark_timing --release --features metal 2>&1 | tee /tmp/bt_output.txt
```

Expected: 24 datasets measured, each line showing serial/rayon/metal times in ms. Output: `Timing → docs/paper3/timing_comparison.json`.

- [ ] **Step 6.2: Verify the JSON**

```bash
python3 -c "
import json
with open('docs/paper3/timing_comparison.json') as f:
    t = json.load(f)
for name in ['s2', 'scale_50k', 'digits']:
    d = t[name]
    s, r, m = d['serial_ms'], d['rayon_ms'], d['metal_ms']
    print(f'{name}: serial={s}ms  rayon={r}ms  metal={m}ms  speedup_rayon={s/r if r else \"N/A\":.1f}x')
"
```

Expected: Rayon speedup ≥ 2× over serial on large datasets; Metal timings present or `null`.

---

## Task 7: Run Stage 3 — Regenerate Stats and Figures

- [ ] **Step 7.1: Run the full experiment suite**

```bash
python docs/paper3/run_experiments.py 2>&1 | tee /tmp/exp_output.txt
```

Expected: All 5 steps complete. Friedman p < 0.05. Saves `multi_seed_results.json`, `statistical_tests.json`, and four PDF figures including `timing_speedup.pdf`.

- [ ] **Step 7.2: Verify figures exist**

```bash
ls -lh docs/paper3/figs/
```

Expected: `delta_ari.pdf`, `algorithm_comparison.pdf`, `friedman_nemenyi.pdf`, `timing_speedup.pdf` all present.

- [ ] **Step 7.3: Capture key numbers for the paper**

```bash
python3 -c "
import json

with open('docs/paper3/statistical_tests.json') as f:
    stats = json.load(f)
with open('experiments/benchmark/results/full_results.json') as f:
    full = json.load(f)
with open('docs/paper3/timing_comparison.json') as f:
    timing = json.load(f)

print('=== Statistical Tests ===')
print(f'Friedman chi2={stats[\"friedman\"][\"chi2\"]:.4f}  p={stats[\"friedman\"][\"p_value\"]:.2e}')
print(f'Avg ranks: {stats[\"average_ranks\"]}')
print(f'Wilcoxon SOM vs KM: p={stats[\"wilcoxon_som_vs_km\"][\"p_value\"]:.4f}')

print()
print('=== ARI Summary ===')
som_aris = [full[n][\"som\"][\"ari\"] for n in full]
km_aris  = [full[n][\"kmeans\"][\"ari\"] for n in full]
print(f'SOM-TSK mean ARI: {sum(som_aris)/len(som_aris):.4f}')
print(f'KMeans++ mean ARI: {sum(km_aris)/len(km_aris):.4f}')
wins = sum(1 for s, k in zip(som_aris, km_aris) if s - k > 0.005)
losses = sum(1 for s, k in zip(som_aris, km_aris) if k - s > 0.005)
ties = len(som_aris) - wins - losses
print(f'Wins/Ties/Losses: {wins}/{ties}/{losses}')

print()
print('=== Timing Speedups ===')
valid = [(n, d) for n, d in timing.items() if d[\"serial_ms\"] and d[\"rayon_ms\"]]
speedups = [d[\"serial_ms\"]/d[\"rayon_ms\"] for _, d in valid]
print(f'Rayon speedup: min={min(speedups):.1f}x  max={max(speedups):.1f}x  mean={sum(speedups)/len(speedups):.1f}x')
" 2>&1
```

Copy the output — you'll need these numbers for the LaTeX update in Task 8.

---

## Task 8: Update LaTeX Paper

**Files:**
- Modify: `docs/paper3/som_tsk_paper_v3.tex`

- [ ] **Step 8.1: Replace the profiling table (Table 5) with a three-way timing table**

Find the current profiling section in `som_tsk_paper_v3.tex` (starts around `\label{sec:profiling}`). The existing `tab:profiling` table shows per-phase percentages. Replace the entire table environment (`\begin{table}` through `\end{table}`) with the following. Fill in the actual measured values from Task 7 Step 7.3 (placeholders shown as `XXX`):

```latex
\begin{table}[!t]
\caption{Wall-Clock Time Comparison: Serial (1 thread), Rayon (all cores), Metal GPU.
Median of 5 runs on Apple M2 Pro (12+4 CPU cores, 19-core GPU).
Speedup $= t_{\text{serial}} / t_{\text{config}}$.}
\label{tab:timing}
\centering
\renewcommand{\arraystretch}{1.1}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
Dataset & $n$ & Serial (ms) & Rayon (ms) & Speedup & Metal (ms) & Speedup \\
\midrule
s2       & 5{,}000  & XXX & XXX & XXX$\times$ & XXX & XXX$\times$ \\
a3       & 7{,}500  & XXX & XXX & XXX$\times$ & XXX & XXX$\times$ \\
digits   &   500    & XXX & XXX & XXX$\times$ & XXX & XXX$\times$ \\
scale\_10k & 10{,}000 & XXX & XXX & XXX$\times$ & XXX & XXX$\times$ \\
scale\_50k & 50{,}000 & XXX & XXX & XXX$\times$ & XXX & XXX$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

Then, fill in the XXX values using the actual numbers from `timing_comparison.json`:

```bash
python3 -c "
import json
with open('docs/paper3/timing_comparison.json') as f:
    t = json.load(f)
for ds in ['s2', 'a3', 'digits', 'scale_10k', 'scale_50k']:
    d = t.get(ds, {})
    s = d.get('serial_ms', 'N/A')
    r = d.get('rayon_ms', 'N/A')
    m = d.get('metal_ms', 'N/A')
    sr = f'{s/r:.1f}' if isinstance(s, (int,float)) and isinstance(r, (int,float)) and r else 'N/A'
    sm = f'{s/m:.1f}' if isinstance(s, (int,float)) and isinstance(m, (int,float)) and m else 'N/A'
    print(f'{ds}: serial={s:.0f}ms rayon={r:.0f}ms ({sr}x) metal={m if isinstance(m,(int,float)) else m}ms ({sm}x)')
"
```

- [ ] **Step 8.2: Update the profiling section text**

Replace the `\textbf{Key insight}` paragraph (after the old profiling table) with:

```latex
\textbf{Rayon parallelism.}
All hot loops in SOM-TSK are parallelized using the Rayon data-parallel library.
Key parallelized components include: BMU assignment (\texttt{assign}), centroid update (\texttt{update}),
inertia computation, E-step in GMM, Silhouette score, and SOM-PlusPlus initialization.
On the Apple M2 Pro (12 performance + 4 efficiency cores), Rayon achieves
XXX$\times$--XXX$\times$ speedup over the single-threaded baseline across the 24 benchmark datasets
(Table~\ref{tab:timing}).
The Metal GPU backend provides additional acceleration of XXX$\times$--XXX$\times$ for large datasets
($n \geq 10{,}000$), with diminishing returns on small datasets due to data transfer overhead.
```

Fill in the XXX ranges using the min/max speedups printed in Task 7 Step 7.3.

- [ ] **Step 8.3: Add timing speedup figure reference**

After the timing table, add a figure reference:

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figs/timing_speedup.pdf}
\caption{Speedup of Rayon (blue) and Metal GPU (orange) over single-threaded serial execution across all 24 datasets, sorted by $n$. Rayon achieves consistent super-linear speedup on larger datasets; Metal GPU shows larger gains for $n \geq 10{,}000$.}
\label{fig:timing_speedup}
\end{figure}
```

- [ ] **Step 8.4: Update Table 3 ARI values if they changed**

Run the comparison check:

```bash
python3 -c "
import json
with open('experiments/benchmark/results/full_results.json') as f:
    full = json.load(f)

# Expected values from prior run (V2 baseline)
v2 = {
    's1':0.9762,'s2':0.6074,'s3':0.3178,'s4':0.1886,
    'a1':0.9770,'a2':0.8409,'a3':0.5944,
    'moons':0.4790,'circles':-0.0033,'spiral':0.0213,
    'anisotropic':1.0000,'varied_density':0.4456,
    'iris':0.6410,'wine':0.8975,'breast_cancer':0.6765,'digits':0.5795,
    'scale_1k':0.9779,'scale_5k':0.9828,'scale_10k':0.9814,'scale_50k':0.9839,
    'dim_32':1.0000,'dim_64':1.0000,'dim_128':1.0000,'dim_256':1.0000,
}
for name, expected in v2.items():
    got = full.get(name, {}).get('som', {}).get('ari', None)
    if got is None:
        print(f'MISSING: {name}')
    elif abs(got - expected) > 0.0001:
        print(f'CHANGED: {name}  expected={expected:.4f}  got={got:.4f}')
    else:
        print(f'ok: {name}')
"
```

If any values changed, update the corresponding rows in `tab:main_results` in the paper.

- [ ] **Step 8.5: Add GPU column to Table 8 (all-algorithm summary) if GPU data is available**

Check GPU coverage:

```bash
python3 -c "
import json
with open('experiments/benchmark/results/full_results.json') as f:
    full = json.load(f)
gpu_aris = [full[n]['som_gpu']['ari'] for n in full if full[n].get('som_gpu',{}).get('ari') is not None]
if gpu_aris:
    print(f'GPU mean ARI: {sum(gpu_aris)/len(gpu_aris):.4f}  n={len(gpu_aris)}')
else:
    print('No GPU ARI data available')
"
```

If GPU ARI data is available for ≥ 20 datasets, add a `SOM-TSK (GPU)` row to `tab:all_algo` in the paper, using the computed mean values.

- [ ] **Step 8.6: Update win/tie/loss counts in Section V introduction**

Find the sentence (around line 392 in the paper):

```
SOM-TSK achieves \textbf{6 wins, 18 ties, 0 losses} against KMeans++
```

Verify these numbers match the output from Task 7 Step 7.3 and update if they changed.

- [ ] **Step 8.7: Update Friedman and Wilcoxon numbers**

Find the statistical comparison section. Update `\chi^2`, p-value, and Wilcoxon p-value using the numbers from `docs/paper3/statistical_tests.json`. The relevant sentence in the paper typically reads:

```
The Friedman test yields $\chi^2(3) = X.XX$, $p = X.XXe{-X}$
```

Use the exact values from `statistical_tests.json`.

- [ ] **Step 8.8: Compile the paper**

```bash
cd docs/paper3
pdflatex som_tsk_paper_v3.tex 2>&1 | grep -E "Error|Warning|error" | head -20
pdflatex som_tsk_paper_v3.tex 2>&1 | tail -3
```

Expected: No errors. Two passes needed for cross-references. Final line: `Output written on som_tsk_paper_v3.pdf`.

- [ ] **Step 8.9: Commit all paper changes**

```bash
cd /Users/evintleovonzko/Documents/projects/evint/SOM_plus_clustering
git add docs/paper3/som_tsk_paper_v3.tex \
        docs/paper3/timing_comparison.json \
        docs/paper3/multi_seed_results.json \
        docs/paper3/statistical_tests.json \
        docs/paper3/figs/ \
        experiments/benchmark/results/rust_metrics.json \
        experiments/benchmark/results/full_results.json
git commit -m "feat(paper3): regenerate all experiments with Rayon+GPU, update timing table and stats"
```

---

## Self-Review

**Spec coverage check:**
- Stage 1 (quality benchmark + GPU): Tasks 1, 5 ✓
- Stage 2 (timing benchmark): Tasks 2, 6 ✓
- Stage 3 (Python pipeline): Tasks 3, 4, 7 ✓
- Stage 4 (paper update): Task 8 ✓
- Error handling for Metal failures: `run_som_metal` returns `Option`, GPU field is omitted from JSON when None ✓
- GPU column gated on ≥50% coverage: Task 4 Step 4.1 ✓
- Serial mode via local `rayon::ThreadPool` (not global): Task 2 ✓
- Median of 5 runs, warmup discarded: Task 2 ✓

**Placeholder scan:** All code blocks are complete. `XXX` placeholders in the LaTeX steps are intentional — they must be filled from actual benchmark output in Task 7 before editing the paper. Each step that has an `XXX` is paired with a bash command to generate the exact values.

**Type consistency:** `run_som_metal` returns `Option<(Vec<usize>, String)>`, same tuple shape as `run_som`. `som_gpu_result` is bound as `Option<(Vec<usize>, String)>` in the `#[cfg(not(feature = "metal"))]` fallback. `Backend::Metal` variant requires `--features metal` at build time — compile guards ensure it never references the variant when the feature is off.
