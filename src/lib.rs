pub mod core;
pub mod backend;
pub mod serialize;
mod error;

pub use error::SomError;

// The following re-exports reference types not yet implemented.
// They will be restored as modules are filled in (Tasks 2–16).
// pub use core::som::{Som, SomBuilder};
// pub use core::som_classification::{SomClassification, SomClassificationBuilder};
// pub use core::kmeans::{KMeans, KMeansBuilder, KMeansInit};
// pub use core::evals::{EvalMethod, ClassEvalMethod};
// pub use core::model_picker::ModelPicker;
// pub use backend::Backend;

// Re-export ndarray types users need at the API boundary
pub use ndarray::{Array1, Array2, ArrayView2};
