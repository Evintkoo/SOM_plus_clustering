use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use som_plus_clustering::{DistanceFunction, InitMethod, SomBuilder, EvalMethod};
use std::sync::{Arc, Mutex};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn, Level};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    models: Arc<Mutex<std::collections::HashMap<String, TrainedModel>>>,
}

#[derive(Clone)]
struct TrainedModel {
    som: som_plus_clustering::Som,
    #[allow(dead_code)]
    data: Array2<f64>,
}

#[derive(Serialize, Deserialize)]
struct TrainRequest {
    data: Vec<Vec<f64>>,
    grid_size: usize,
    learning_rate: f64,
    neighbor_radius: f64,
    epochs: usize,
    init_method: String,
    distance: String,
    #[serde(default)]
    use_gpu: bool,
}

#[derive(Serialize, Deserialize)]
struct TrainResponse {
    model_id: String,
    status: String,
    grid_size: usize,
    dimensions: usize,
    samples: usize,
    elapsed_ms: f64,
    silhouette: Option<f64>,
    davies_bouldin: Option<f64>,
    calinski_harabasz: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct PredictRequest {
    model_id: String,
    data: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
struct PredictResponse {
    predictions: Vec<usize>,
    model_info: String,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkResponse {
    total_time_ms: f64,
    samples_per_sec: f64,
    optimization_notes: String,
}

#[tokio::main]
async fn main() {
    // Industrial-grade structured logging — pretty for terminal, JSON via RUST_LOG_FORMAT=json
    let use_json = std::env::var("RUST_LOG_FORMAT").map(|v| v == "json").unwrap_or(false);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "som_api=info,tower_http=info".into());

    if use_json {
        tracing_subscriber::fmt()
            .with_target(true)
            .with_level(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_env_filter(env_filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_target(true)
            .with_level(true)
            .with_thread_ids(true)
            .with_env_filter(env_filter)
            .pretty()
            .init();
    }

    let state = AppState {
        models: Arc::new(Mutex::new(std::collections::HashMap::new())),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/train", post(train_model))
        .route("/predict", post(predict))
        .route("/benchmark", post(run_benchmark))
        .with_state(state)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &axum::http::Request<_>| {
                    let request_id = Uuid::new_v4().to_string();
                    tracing::info_span!(
                        "http_request",
                        method = %request.method(),
                        uri = %request.uri(),
                        request_id = %request_id,
                    )
                })
                .on_response(
                    tower_http::trace::DefaultOnResponse::new().level(Level::INFO),
                )
                .on_failure(
                    tower_http::trace::DefaultOnFailure::new().level(Level::ERROR),
                ),
        )
        .layer(tower_http::cors::CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    info!(
        addr = "127.0.0.1:3000",
        version = "0.1.0",
        "SOM API server started"
    );

    axum::serve(listener, app).await.unwrap();
}

async fn health() -> Json<serde_json::Value> {
    info!("Health check requested");
    Json(serde_json::json!({
        "status": "healthy",
        "version": "0.1.0",
        "optimizations": [
            "fast inverse sqrt",
            "Taylor exp() approximation",
            "gaussian_grid caching",
            "norm caching",
            "squared distance in KMeans",
            "Manhattan distance support"
        ]
    }))
}

async fn train_model(
    State(state): State<AppState>,
    Json(req): Json<TrainRequest>,
) -> (StatusCode, Json<TrainResponse>) {
    let start = std::time::Instant::now();
    let n_samples = req.data.len();
    let n_features = if n_samples > 0 { req.data[0].len() } else { 0 };

    info!(
        samples = n_samples,
        features = n_features,
        grid_size = req.grid_size,
        epochs = req.epochs,
        init_method = %req.init_method,
        distance = %req.distance,
        learning_rate = req.learning_rate,
        use_gpu = req.use_gpu,
        "Training request received"
    );

    // Validate grid size vs sample count
    let total_neurons = req.grid_size * req.grid_size;
    let grid_size = if total_neurons > n_samples {
        let clamped = (n_samples as f64).sqrt().floor() as usize;
        warn!(requested = req.grid_size, clamped = clamped, samples = n_samples, "Grid too large for dataset, clamping");
        clamped.max(2)
    } else {
        req.grid_size
    };

    let flat_data: Vec<f64> = req.data.into_iter().flatten().collect();
    let data = match Array2::from_shape_vec((n_samples, n_features), flat_data) {
        Ok(d) => d,
        Err(e) => {
            error!(error = %e, "Failed to reshape input data");
            return (
                StatusCode::BAD_REQUEST,
                Json(TrainResponse {
                    model_id: String::new(),
                    status: "invalid data shape".to_string(),
                    grid_size: 0,
                    dimensions: 0,
                    samples: 0,
                    elapsed_ms: 0.0,
                    silhouette: None,
                    davies_bouldin: None,
                    calinski_harabasz: None,
                }),
            );
        }
    };

    let dist_fn = match req.distance.as_str() {
        "cosine" => DistanceFunction::Cosine,
        "manhattan" => DistanceFunction::Manhattan,
        _ => DistanceFunction::Euclidean,
    };

    let init_method = match req.init_method.as_str() {
        "kmeans" => InitMethod::KMeans,
        "kmeans_plus_plus" => InitMethod::KMeansPlusPlus,
        "kde" => InitMethod::Kde,
        "som_plus_plus" => InitMethod::SomPlusPlus,
        _ => InitMethod::Random,
    };

    let mut som = SomBuilder::new()
        .grid(grid_size, grid_size)
        .dim(n_features)
        .learning_rate(req.learning_rate)
        .unwrap_or_else(|_| {
            warn!(learning_rate = req.learning_rate, "Invalid learning rate, falling back to 0.5");
            SomBuilder::new()
                .grid(grid_size, grid_size)
                .dim(n_features)
                .learning_rate(0.5)
                .unwrap()
        })
        .neighbor_radius(req.neighbor_radius)
        .init_method(init_method)
        .distance(dist_fn)
        .max_iter(500_000) // Hard cap to prevent runaway training
        .build();

    if req.use_gpu {
        som.set_backend(som_plus_clustering::Backend::Metal);
    }

    let result = som.fit(&data.view(), req.epochs, true, None);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    if result.is_ok() {
        // Compute evaluation metrics
        let eval_results = som.evaluate(&data.view(), &[
            EvalMethod::Silhouette,
            EvalMethod::DaviesBouldin,
            EvalMethod::CalinskiHarabasz,
        ]);

        let (silhouette, davies_bouldin, calinski_harabasz) = match eval_results {
            Ok(scores) => (
                scores.get(&EvalMethod::Silhouette).copied(),
                scores.get(&EvalMethod::DaviesBouldin).copied(),
                scores.get(&EvalMethod::CalinskiHarabasz).copied(),
            ),
            Err(e) => {
                warn!(error = %e, "Evaluation metrics failed, returning None");
                (None, None, None)
            }
        };

        let model_id = Uuid::new_v4().to_string();
        let model = TrainedModel {
            som: som.clone(),
            data: data.clone(),
        };
        state.models.lock().unwrap().insert(model_id.clone(), model);

        info!(
            model_id = %model_id,
            elapsed_ms = elapsed_ms,
            samples = n_samples,
            features = n_features,
            grid = format!("{}x{}", grid_size, grid_size),
            silhouette = ?silhouette,
            davies_bouldin = ?davies_bouldin,
            calinski_harabasz = ?calinski_harabasz,
            "Model trained successfully"
        );

        (
            StatusCode::OK,
            Json(TrainResponse {
                model_id,
                status: "success".to_string(),
                grid_size,
                dimensions: n_features,
                samples: n_samples,
                elapsed_ms,
                silhouette,
                davies_bouldin,
                calinski_harabasz,
            }),
        )
    } else {
        error!(
            elapsed_ms = elapsed_ms,
            samples = n_samples,
            "Training failed"
        );
        (
            StatusCode::BAD_REQUEST,
            Json(TrainResponse {
                model_id: String::new(),
                status: "training failed".to_string(),
                grid_size: 0,
                dimensions: 0,
                samples: 0,
                elapsed_ms,
                silhouette: None,
                davies_bouldin: None,
                calinski_harabasz: None,
            }),
        )
    }
}

async fn predict(
    State(state): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> (StatusCode, Json<PredictResponse>) {
    let start = std::time::Instant::now();
    let models = state.models.lock().unwrap();

    match models.get(&req.model_id) {
        Some(model) => {
            let n_samples = req.data.len();
            let n_features = if n_samples > 0 { req.data[0].len() } else { 0 };
            let flat_data: Vec<f64> = req.data.into_iter().flatten().collect();
            let data = match Array2::from_shape_vec((n_samples, n_features), flat_data) {
                Ok(d) => d,
                Err(e) => {
                    error!(error = %e, model_id = %req.model_id, "Failed to reshape prediction data");
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(PredictResponse {
                            predictions: vec![],
                            model_info: "Invalid data shape".to_string(),
                        }),
                    );
                }
            };

            match model.som.predict(&data.view()) {
                Ok(labels) => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    info!(
                        model_id = %req.model_id,
                        samples = n_samples,
                        elapsed_ms = elapsed_ms,
                        "Prediction completed"
                    );
                    (
                        StatusCode::OK,
                        Json(PredictResponse {
                            predictions: labels.to_vec(),
                            model_info: format!("Predicted {} samples", n_samples),
                        }),
                    )
                }
                Err(e) => {
                    error!(model_id = %req.model_id, error = %e, "Prediction failed");
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(PredictResponse {
                            predictions: vec![],
                            model_info: "Prediction failed".to_string(),
                        }),
                    )
                }
            }
        }
        None => {
            warn!(model_id = %req.model_id, "Model not found");
            (
                StatusCode::NOT_FOUND,
                Json(PredictResponse {
                    predictions: vec![],
                    model_info: "Model not found".to_string(),
                }),
            )
        }
    }
}

#[derive(Serialize, Deserialize)]
struct BenchmarkRequest {
    n_samples: usize,
    n_features: usize,
    grid_size: usize,
    learning_rate: f64,
    neighbor_radius: f64,
    epochs: usize,
    init_method: String,
    distance: String,
    #[serde(default)]
    use_gpu: bool,
}

async fn run_benchmark(
    State(_state): State<AppState>,
    Json(req): Json<BenchmarkRequest>,
) -> Json<BenchmarkResponse> {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    let start = std::time::Instant::now();

    info!(
        samples = req.n_samples,
        features = req.n_features,
        epochs = req.epochs,
        "Benchmark started"
    );

    // Generate data server-side to avoid massive JSON payloads
    let data = Array2::random((req.n_samples, req.n_features), Uniform::new(0.0, 10.0));

    let dist_fn = match req.distance.as_str() {
        "cosine" => DistanceFunction::Cosine,
        "manhattan" => DistanceFunction::Manhattan,
        _ => DistanceFunction::Euclidean,
    };

    let mut som = SomBuilder::new()
        .grid(req.grid_size, req.grid_size)
        .dim(req.n_features)
        .learning_rate(req.learning_rate)
        .unwrap_or_else(|_| {
            SomBuilder::new()
                .grid(req.grid_size, req.grid_size)
                .dim(req.n_features)
                .learning_rate(0.5)
                .unwrap()
        })
        .neighbor_radius(req.neighbor_radius)
        .distance(dist_fn)
        .max_iter((req.epochs * req.n_samples).min(500_000))
        .build();

    if req.use_gpu {
        som.set_backend(som_plus_clustering::Backend::Metal);
    }

    let _ = som.fit(&data.view(), req.epochs, true, None);

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let samples_per_sec = (req.n_samples as f64) / (elapsed_ms / 1000.0);

    info!(
        samples = req.n_samples,
        elapsed_ms = elapsed_ms,
        samples_per_sec = samples_per_sec,
        "Benchmark completed"
    );

    Json(BenchmarkResponse {
        total_time_ms: elapsed_ms,
        samples_per_sec,
        optimization_notes: "Data generated server-side. Using fast inverse sqrt, Taylor exp(), squared distance".to_string(),
    })
}
