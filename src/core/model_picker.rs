use crate::{SomError, core::som::{Som, SomBuilder, InitMethod}, core::evals::EvalMethod};
use ndarray::ArrayView2;
use crate::core::distance::DistanceFunction;

pub struct ModelPicker {
    models: Vec<Som>,
    scores: Vec<f64>,
}

impl ModelPicker {
    pub fn new() -> Self {
        Self { models: vec![], scores: vec![] }
    }

    pub fn evaluate_all_init_methods(
        &mut self,
        data: &ArrayView2<f64>,
        grid_m: usize,
        grid_n: usize,
        learning_rate: f64,
        neighbor_rad: f64,
        dist_fn: DistanceFunction,
        max_iter: Option<usize>,
        epoch: usize,
    ) -> Result<(), SomError> {
        let all_methods = [
            InitMethod::Random, InitMethod::KMeans, InitMethod::KMeansPlusPlus,
            InitMethod::KdekMeans, InitMethod::SomPlusPlus, InitMethod::Zero,
            InitMethod::He, InitMethod::NaiveSharding, InitMethod::LeCun,
            InitMethod::Lsuv, InitMethod::Kde,
        ];
        for method in &all_methods {
            let mut b = SomBuilder::new()
                .grid(grid_m, grid_n)
                .dim(data.ncols())
                .learning_rate(learning_rate)?
                .neighbor_radius(neighbor_rad)
                .distance(dist_fn)
                .init_method(*method);
            if let Some(mi) = max_iter {
                b = b.max_iter(mi);
            }
            let mut som = b.build();
            if som.fit(data, epoch, true, None).is_err() {
                continue;
            }
            let score = som
                .evaluate(data, &[EvalMethod::Silhouette])
                .ok()
                .and_then(|s| s.get(&EvalMethod::Silhouette).copied())
                .unwrap_or(f64::NEG_INFINITY);
            self.models.push(som);
            self.scores.push(score);
        }
        Ok(())
    }

    pub fn best_model(self) -> Result<Som, SomError> {
        if self.models.is_empty() {
            return Err(SomError::NotFitted("best_model"));
        }
        let best_idx = self
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;
        Ok(self.models.into_iter().nth(best_idx).unwrap())
    }
}
