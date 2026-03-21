use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::{
    SomError,
    core::evals::{accuracy, ClassEvalMethod},
    core::som::{Som, SomBuilder, InitMethod},
};
use crate::backend::Backend;
use std::collections::HashMap;

/// Builder for SomClassification — wraps SomBuilder.
pub struct SomClassificationBuilder(SomBuilder);

impl SomClassificationBuilder {
    pub fn new() -> Self {
        Self(SomBuilder::new())
    }

    pub fn grid(mut self, m: usize, n: usize) -> Self {
        self.0 = self.0.grid(m, n);
        self
    }

    pub fn dim(mut self, d: usize) -> Self {
        self.0 = self.0.dim(d);
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Result<Self, SomError> {
        self.0 = self.0.learning_rate(lr)?;
        Ok(self)
    }

    pub fn init_method(mut self, m: InitMethod) -> Self {
        self.0 = self.0.init_method(m);
        self
    }

    pub fn build(self) -> SomClassification {
        SomClassification {
            som: self.0.build(),
            neuron_label: None,
        }
    }
}

/// Supervised SOM classifier.
/// After fit, each neuron is assigned the label of its nearest training point.
pub struct SomClassification {
    som: Som,
    neuron_label: Option<Array2<usize>>,  // shape [m, n]
}

impl SomClassification {
    pub fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<usize>,
        epoch: usize,
        shuffle: bool,
        batch_size: Option<usize>,
    ) -> Result<(), SomError> {
        self.som.fit(x, epoch, shuffle, batch_size)?;
        // Assign each neuron the label of its closest training point
        let centers = self.som.cluster_centers(); // [m*n, dim]
        let m = self.som.m;
        let n = self.som.n;
        let mut labels = Array2::<usize>::zeros((m, n));
        for ci in 0..m * n {
            let best = (0..x.nrows())
                .min_by(|&a, &b| {
                    let da = (&x.row(a) - &centers.row(ci)).mapv(|v| v * v).sum();
                    let db = (&x.row(b) - &centers.row(ci)).mapv(|v| v * v).sum();
                    da.partial_cmp(&db)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            labels[[ci / n, ci % n]] = y[best];
        }
        self.neuron_label = Some(labels);
        Ok(())
    }

    pub fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<usize>, SomError> {
        let bmu_ids = self.som.predict(x)?;
        let labels = self
            .neuron_label
            .as_ref()
            .ok_or(SomError::NotFitted("predict"))?;
        let n = self.som.n;
        Ok(Array1::from_iter(
            bmu_ids.iter().map(|&idx| labels[[idx / n, idx % n]]),
        ))
    }

    pub fn evaluate(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<usize>,
        methods: &[ClassEvalMethod],
    ) -> Result<HashMap<ClassEvalMethod, f64>, SomError> {
        let pred = self.predict(x)?;
        let mut out = HashMap::new();
        let run_all = methods.contains(&ClassEvalMethod::All);
        if run_all || methods.contains(&ClassEvalMethod::Accuracy) {
            out.insert(ClassEvalMethod::Accuracy, accuracy(y, &pred.view()));
        }
        Ok(out)
    }

    pub fn set_backend(&mut self, b: Backend) {
        self.som.set_backend(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn labeled_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_fn((30, 2), |(i, j)| i as f64 * 0.1 + j as f64 * 0.01);
        let y = Array1::from_iter((0..30).map(|i| i % 3));
        (x, y)
    }

    #[test]
    fn fit_predict_shape() {
        let (x, y) = labeled_data();
        let mut clf = SomClassificationBuilder::new()
            .grid(3, 3)
            .dim(2)
            .learning_rate(0.5)
            .unwrap()
            .build();
        clf.fit(&x.view(), &y.view(), 2, false, None).unwrap();
        let pred = clf.predict(&x.view()).unwrap();
        assert_eq!(pred.len(), 30);
    }
}
