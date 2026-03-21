//! Rust benchmark: SOM + KMeans on standard benchmark datasets.
//! Reads normalized CSVs from experiments/datasets/, outputs JSON to experiments/results/rust_results.json.
//!
//! Run from crate root:
//!   cargo run --example benchmark --release

use ndarray::Array2;
use som_plus_clustering::{
    calinski_harabasz_score, davies_bouldin_index, dunn_index, silhouette_score,
    DistanceFunction, InitMethod, KMeansBuilder, KMeansInit, SomBuilder,
};
use std::{fs, io::Write, time::Instant};

// ---------------------------------------------------------------------------
// Dataset config: (csv_name, n_classes, som_grid_m, som_grid_n)
// ---------------------------------------------------------------------------
const DATASETS: &[(&str, usize, usize, usize)] = &[
    ("iris",          3,  5, 5),
    ("wine",          3,  5, 5),
    ("breast_cancer", 2,  6, 6),
    ("digits",       10,  8, 8),
    ("blobs",         5,  7, 7),
];

const SOM_EPOCHS:  usize = 10;
const SOM_LR:      f64   = 0.5;
const SOM_RADIUS:  f64   = 3.0;

// ---------------------------------------------------------------------------

fn load_csv(name: &str) -> (Array2<f64>, Vec<usize>) {
    let path = format!("experiments/datasets/{name}.csv");
    let mut reader = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));

    let headers = reader.headers().unwrap().clone();
    let n_cols = headers.len();
    let label_col = n_cols - 1;

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    for record in reader.records() {
        let rec = record.unwrap();
        let row: Vec<f64> = (0..label_col)
            .map(|i| rec[i].parse::<f64>().unwrap())
            .collect();
        let lbl: usize = rec[label_col].parse::<f64>().unwrap() as usize;
        rows.push(row);
        labels.push(lbl);
    }

    let n = rows.len();
    let d = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    (Array2::from_shape_vec((n, d), flat).unwrap(), labels)
}

// Simple JSON serializer (avoids feature-flag complexity)
fn json_f(v: f64) -> String {
    if v.is_finite() { format!("{v:.6}") } else { "null".into() }
}

fn metrics_json(sil: f64, db: f64, ch: f64, dunn: f64) -> String {
    format!(
        r#"{{"silhouette":{sil},"davies_bouldin":{db},"calinski_harabasz":{ch},"dunn":{dunn}}}"#,
        sil  = json_f(sil),
        db   = json_f(db),
        ch   = json_f(ch),
        dunn = json_f(dunn),
    )
}

fn run_som(data: &Array2<f64>, m: usize, n: usize) -> String {
    let dim = data.ncols();
    let mut som = SomBuilder::new()
        .grid(m, n)
        .dim(dim)
        .learning_rate(SOM_LR)
        .expect("valid lr")
        .neighbor_radius(SOM_RADIUS)
        .init_method(InitMethod::Random)
        .distance(DistanceFunction::Euclidean)
        .build();

    let t0 = Instant::now();
    som.fit(&data.view(), SOM_EPOCHS, true, None)
        .expect("SOM fit failed");
    let fit_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let labels = som.predict(&data.view()).expect("SOM predict failed");
    let predict_s = t0.elapsed().as_secs_f64();

    let lv = labels.view();
    let sil = silhouette_score(&data.view(), &lv).unwrap_or(f64::NAN);
    let db  = davies_bouldin_index(&data.view(), &lv).unwrap_or(f64::NAN);
    let ch  = calinski_harabasz_score(&data.view(), &lv).unwrap_or(f64::NAN);
    let dunn = dunn_index(&data.view(), &lv).unwrap_or(f64::NAN);

    let metrics = metrics_json(sil, db, ch, dunn);
    format!(
        r#"{{"fit_time_s":{fit:.4},"predict_time_s":{pred:.4},"n_neurons":{nn},{metrics}}}"#,
        fit     = fit_s,
        pred    = predict_s,
        nn      = m * n,
        metrics = &metrics[1..metrics.len()-1],  // strip outer braces
    )
}

fn run_kmeans(data: &Array2<f64>, k: usize) -> String {
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

    let lv = labels.view();
    let sil  = silhouette_score(&data.view(), &lv).unwrap_or(f64::NAN);
    let db   = davies_bouldin_index(&data.view(), &lv).unwrap_or(f64::NAN);
    let ch   = calinski_harabasz_score(&data.view(), &lv).unwrap_or(f64::NAN);
    let dunn = dunn_index(&data.view(), &lv).unwrap_or(f64::NAN);
    let inertia = km.inertia().unwrap_or(f64::NAN);

    let metrics = metrics_json(sil, db, ch, dunn);
    format!(
        r#"{{"fit_time_s":{fit:.4},"predict_time_s":{pred:.4},"inertia":{inertia:.4},{metrics}}}"#,
        fit     = fit_s,
        pred    = predict_s,
        inertia = inertia,
        metrics = &metrics[1..metrics.len()-1],
    )
}

fn main() {
    fs::create_dir_all("experiments/results").unwrap();

    let mut entries: Vec<String> = Vec::new();

    for &(name, n_classes, m, n) in DATASETS {
        println!("\n{}", "=".repeat(60));
        println!("Dataset: {name}  (k={n_classes}, grid={m}x{n})");

        let (data, _labels) = load_csv(name);
        println!("  Shape: {} x {}", data.nrows(), data.ncols());

        print!("  Running SOM ...  ");
        let _ = std::io::stdout().flush();
        let som_json = run_som(&data, m, n);
        println!("done");

        print!("  Running KMeans ...");
        let _ = std::io::stdout().flush();
        let km_json = run_kmeans(&data, n_classes);
        println!("  done");

        entries.push(format!(
            r#"  "{name}": {{"n_samples":{ns},"n_features":{nf},"n_classes":{nc},"som":{som},"kmeans":{km}}}"#,
            name = name,
            ns   = data.nrows(),
            nf   = data.ncols(),
            nc   = n_classes,
            som  = som_json,
            km   = km_json,
        ));
    }

    let json = format!("{{\n{}\n}}\n", entries.join(",\n"));
    let out_path = "experiments/results/rust_results.json";
    fs::write(out_path, &json).expect("cannot write results");
    println!("\nResults saved to {out_path}");
}
