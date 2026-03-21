pub mod core;
pub mod backend;
pub mod serialize;
mod error;

pub use error::SomError;

pub use core::som::{Som, SomBuilder, InitMethod};
pub use core::som_classification::{SomClassification, SomClassificationBuilder};
pub use core::kmeans::{KMeans, KMeansBuilder, KMeansInit};
pub use core::evals::EvalMethod;
pub use core::evals::bcubed_scores;
pub use backend::Backend;
pub use core::distance::DistanceFunction;
pub use serialize::SomState;

// Re-export ndarray types users need at the API boundary
pub use ndarray::{Array1, Array2, ArrayView2};
