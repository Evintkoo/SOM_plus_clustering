#[derive(thiserror::Error, Debug)]
pub enum SomError {
    #[error("learning rate must be <= 1.76, got {0}")]
    InvalidLearningRate(f64),
    #[error("invalid init method: {0}")]
    InvalidInitMethod(String),
    #[error("SOM must be fitted before calling {0}")]
    NotFitted(&'static str),
    #[error("already fitted — create a new instance to retrain")]
    AlreadyFitted,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("input data contains NaN or infinite values")]
    InvalidInputData,
    #[error("degenerate cluster: within-cluster variance is zero")]
    ZeroWithinClusterVariance,
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("KDE found {found} local maxima, need at least {needed}")]
    KdeInsufficientMaxima { found: usize, needed: usize },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
