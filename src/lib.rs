pub mod backend;
pub mod core;
mod error;
pub mod serialize;

pub use error::SomError;

pub use backend::Backend;
pub use core::distance::DistanceFunction;
pub use core::evals::bcubed_scores;
pub use core::evals::{
    calinski_harabasz_score, davies_bouldin_index, dunn_index, silhouette_score, ClassEvalMethod,
    EvalMethod,
};
pub use core::kmeans::{KMeans, KMeansBuilder, KMeansInit};
pub use core::model_picker::{ModelPicker, PickerConfig};
pub use core::som::{InitMethod, Som, SomBuilder};
pub use core::som_classification::{SomClassification, SomClassificationBuilder};
pub use serialize::SomState;

// Re-export ndarray types users need at the API boundary
pub use ndarray::{Array1, Array2, ArrayView2};
