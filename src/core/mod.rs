pub mod distance;
pub mod evals;
pub mod gmm;
pub mod init;
pub mod kde;
pub mod kmeans;
pub mod model_picker;
pub mod neighborhood;
pub mod som;
pub mod som_classification;
pub mod autosom;
pub mod densom;

/// Bandwidth selection rule for DenSOM's adaptive Gaussian smoothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaMethod {
    /// Silverman's rule (robust to outliers via IQR). Default.
    Silverman,
    /// Scott's rule (`n^(-1/(d+4)) * σ_data`).
    Scott,
}
