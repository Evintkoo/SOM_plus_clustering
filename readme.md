# som_plus_clustering

Industrial-grade Self-Organizing Map (SOM) and clustering library for Rust.

[![Crates.io](https://img.shields.io/crates/v/som_plus_clustering.svg)](https://crates.io/crates/som_plus_clustering)
[![docs.rs](https://img.shields.io/docsrs/som_plus_clustering)](https://docs.rs/som_plus_clustering)
[![CI](https://github.com/Evintkoo/SOM_plus_clustering/actions/workflows/rust.yml/badge.svg)](https://github.com/Evintkoo/SOM_plus_clustering/actions/workflows/rust.yml)

## Features

- **Self-Organizing Maps** — unsupervised topology-preserving dimensionality reduction
- **SOM Classification** — supervised label propagation via SOM
- **KMeans** — multiple init strategies: Random, KMeans++, KDE-KMeans, SOM++, and more
- **Model selection** — automatic best-model picking via Silhouette score
- **CPU backend** — parallel training via [rayon](https://github.com/rayon-rs/rayon)
- **CUDA backend** — optional GPU acceleration (`--features cuda`, requires CUDA Toolkit)
- **Metal backend** — optional GPU acceleration on macOS (`--features metal`)
- **Serialization** — bincode v2 (default) + JSON (`--features serde-json`)

## Installation

```toml
[dependencies]
som_plus_clustering = "0.1"
```

For GPU acceleration on macOS:
```toml
[dependencies]
som_plus_clustering = { version = "0.1", features = ["metal"] }
```

## Quick Start

```rust
use som_plus_clustering::{SomBuilder, DistanceFunction, InitMethod, EvalMethod};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 100 samples, 4 features
    let data = Array2::<f64>::zeros((100, 4));

    // Build and train a 10x10 SOM
    let mut som = SomBuilder::new(10, 10, 4)
        .learning_rate(0.5)?
        .radius(3.0)
        .max_iter(200)
        .init_method(InitMethod::KMeansPlusPlus)
        .distance_function(DistanceFunction::Euclidean)
        .build();

    som.fit(&data)?;

    // Predict cluster index for each sample
    let labels = som.predict(&data)?;

    // Evaluate
    let score = som.evaluate(&data, EvalMethod::Silhouette)?;
    println!("Silhouette score: {score:.4}");

    Ok(())
}
```

## Initialization Methods

| Method | Description |
|--------|-------------|
| `Random` | Uniform random in data range |
| `KMeans` | K-Means centroids |
| `KMeansPlusPlus` | K-Means++ seeding |
| `KdekMeans` | KDE-guided K-Means |
| `SomPlusPlus` | SOM-aware diversity seeding |
| `Zero` | Identity-based initialization |
| `He` | He weight initialization |
| `LeCun` | LeCun weight initialization |
| `NaiveSharding` | Column-based sharding |
| `Lsuv` | Layer-sequential unit variance |
| `Kde` | Kernel Density Estimation peaks |

## Evaluation Metrics

`EvalMethod::Silhouette`, `DaviesBouldin`, `CalinskiHarabasz`, `Dunn`, or `All`.

## Serialization

```rust
// Save
som.save("model.bin")?;

// Load
let som = som_plus_clustering::SomBuilder::load("model.bin")?;
```

JSON support:
```toml
som_plus_clustering = { version = "0.1", features = ["serde-json"] }
```

## License

MIT