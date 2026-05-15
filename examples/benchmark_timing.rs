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
        let result = runner.install(|| -> Result<ndarray::Array1<usize>, _> {
            som.fit(&data.view(), epochs, false, None)?;
            som.predict_clustered_refined(&data.view(), k)
        });
        match result {
            Ok(_) => times.push(t0.elapsed().as_secs_f64() * 1000.0),
            Err(e) => {
                eprintln!("  [timing] run failed: {e}");
                return None;
            }
        }
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
