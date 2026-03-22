//! Rust quality + speed benchmark across the full SIPU + shape + real-world + scalability suite.
//!
//! Reads dataset_config.json, runs SOM and KMeans on every dataset, outputs:
//!   experiments/benchmark/results/rust_metrics.json    (timing + internal metrics)
//!   experiments/benchmark/results/<name>_som_labels.csv
//!   experiments/benchmark/results/<name>_km_labels.csv
//!
//! Run from crate root:
//!   cargo run --example benchmark_quality --release

use ndarray::Array2;
use serde::Deserialize;
use som_plus_clustering::{
    calinski_harabasz_score, davies_bouldin_index, dunn_index, silhouette_score,
    DistanceFunction, InitMethod, KMeansBuilder, KMeansInit, SomBuilder,
};
use std::{fs, io::Write, time::Instant};

// ---------------------------------------------------------------------------
// Config (mirrors dataset_config.json)
// ---------------------------------------------------------------------------
#[derive(Deserialize)]
struct DatasetConfig {
    name: String,
    n_true_clusters: usize,
    som_m: usize,
    som_n: usize,
    epochs: usize,
}

// ---------------------------------------------------------------------------
// CSV loader: last column is label (ignored here, ground truth handled by Python)
// ---------------------------------------------------------------------------
fn load_csv(base_dir: &str, name: &str) -> Array2<f64> {
    let path = format!("{base_dir}/{name}.csv");
    let mut reader = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let n_cols = reader.headers().unwrap().len();
    let feature_cols = n_cols - 1; // last col = label

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

// ---------------------------------------------------------------------------
// Save predicted labels (one per line)
// ---------------------------------------------------------------------------
fn save_labels(path: &str, labels: &[usize]) {
    let content: String = labels.iter().map(|l| l.to_string() + "\n").collect();
    fs::write(path, content).unwrap_or_else(|e| eprintln!("warn: cannot write {path}: {e}"));
}

// ---------------------------------------------------------------------------
// JSON helpers (hand-rolled to avoid pulling in a full serde_json dep in example)
// ---------------------------------------------------------------------------
fn jf(v: f64) -> String {
    if v.is_finite() { format!("{v:.6}") } else { "null".into() }
}

// Silhouette and Dunn are O(n²) — skip them when n exceeds this threshold.
const QUADRATIC_LIMIT: usize = 8_000;

fn compute_metrics(data: &Array2<f64>, labels: &ndarray::Array1<usize>) -> (f64, f64, f64, f64) {
    let n = data.nrows();
    let lv = labels.view();
    let sil  = if n <= QUADRATIC_LIMIT { silhouette_score(&data.view(), &lv).unwrap_or(f64::NAN) } else { f64::NAN };
    let db   = davies_bouldin_index(&data.view(), &lv).unwrap_or(f64::NAN);
    let ch   = calinski_harabasz_score(&data.view(), &lv).unwrap_or(f64::NAN);
    let dunn = if n <= QUADRATIC_LIMIT { dunn_index(&data.view(), &lv).unwrap_or(f64::NAN) } else { f64::NAN };
    (sil, db, ch, dunn)
}

fn run_som(data: &Array2<f64>, m: usize, n: usize, epochs: usize) -> (Vec<usize>, String) {
    let dim = data.ncols();
    let mut som = SomBuilder::new()
        .grid(m, n)
        .dim(dim)
        .learning_rate(0.5)
        .expect("valid lr")
        .neighbor_radius(3.0)
        .init_method(InitMethod::Random)
        .distance(DistanceFunction::Euclidean)
        .build();

    let t0 = Instant::now();
    som.fit(&data.view(), epochs, true, None)
        .expect("SOM fit failed");
    let fit_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let labels = som.predict(&data.view()).expect("SOM predict failed");
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
    (labels.to_vec(), json)
}

fn run_kmeans(data: &Array2<f64>, k: usize) -> (Vec<usize>, String) {
    let mut km = KMeansBuilder::new()
        .n_clusters(k)
        .method(KMeansInit::PlusPlus)
        .max_iters(300)
        .build();

    let t0 = Instant::now();
    km.fit(&data.view()).expect("KMeans fit failed");
    let fit_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let labels = km.predict(&data.view()).expect("KMeans predict failed");
    let predict_s = t0.elapsed().as_secs_f64();

    let (sil, db, ch, dunn) = compute_metrics(data, &labels);
    let inertia = km.inertia().unwrap_or(f64::NAN);

    let json = format!(
        r#"{{"fit_time_s":{fit:.4},"predict_time_s":{pred:.6},"inertia":{inertia:.4},"silhouette":{sil},"davies_bouldin":{db},"calinski_harabasz":{ch},"dunn":{dunn}}}"#,
        fit     = fit_s,
        pred    = predict_s,
        inertia = inertia,
        sil     = jf(sil),
        db      = jf(db),
        ch      = jf(ch),
        dunn    = jf(dunn),
    );
    (labels.to_vec(), json)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base_dir    = args.get(1).map(String::as_str).unwrap_or("experiments/benchmark/datasets");
    let cfg_path    = args.get(2).map(String::as_str).unwrap_or("experiments/benchmark/dataset_config.json");
    let results_dir = args.get(3).map(String::as_str).unwrap_or("experiments/benchmark/results");

    fs::create_dir_all(results_dir).unwrap();

    // Load config
    let cfg_str = fs::read_to_string(cfg_path)
        .unwrap_or_else(|_| panic!("Run generate.py first — config not found at {cfg_path}"));
    let configs: Vec<DatasetConfig> = serde_json::from_str(&cfg_str)
        .expect("malformed dataset_config.json");

    let total = configs.len();
    let mut entries: Vec<String> = Vec::new();

    for (i, cfg) in configs.iter().enumerate() {
        let name    = &cfg.name;
        let k       = cfg.n_true_clusters;
        let m       = cfg.som_m;
        let n       = cfg.som_n;
        let epochs  = cfg.epochs;

        print!("[{:>2}/{total}] {name:<28}", i + 1);
        let _ = std::io::stdout().flush();

        let data = load_csv(base_dir, name);
        print!("  {}x{}  ", data.nrows(), data.ncols());
        let _ = std::io::stdout().flush();

        // SOM
        let (som_labels, som_json) = run_som(&data, m, n, epochs);
        save_labels(&format!("{results_dir}/{name}_som_labels.csv"), &som_labels);
        print!("SOM✓  ");
        let _ = std::io::stdout().flush();

        // KMeans
        let (km_labels, km_json) = run_kmeans(&data, k);
        save_labels(&format!("{results_dir}/{name}_km_labels.csv"), &km_labels);
        println!("KMeans✓");

        entries.push(format!(
            r#"  "{name}": {{"n_samples":{ns},"n_features":{nf},"n_true_clusters":{nc},"som_grid":"{m}x{n}","som":{som},"kmeans":{km}}}"#,
            name = name,
            ns   = data.nrows(),
            nf   = data.ncols(),
            nc   = k,
            m    = m,
            n    = n,
            som  = som_json,
            km   = km_json,
        ));
    }

    let json = format!("{{\n{}\n}}\n", entries.join(",\n"));
    let out_path = format!("{results_dir}/rust_metrics.json");
    fs::write(&out_path, &json).expect("cannot write metrics");
    println!("\nMetrics → {out_path}");
    println!("Labels  → {results_dir}/<name>_{{som,km}}_labels.csv");
}
