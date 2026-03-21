use ndarray::{Array1, Array2, ArrayView2, Axis};
use crate::SomError;
use crate::core::init::{init_random, init_som_plus_plus};
use std::cell::Cell;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KMeansInit {
    Random,
    PlusPlus,
}

pub struct KMeansBuilder {
    n_clusters: usize,
    method: KMeansInit,
    max_iters: usize,
    tol: f64,
}

impl KMeansBuilder {
    pub fn new() -> Self {
        Self { n_clusters: 8, method: KMeansInit::PlusPlus, max_iters: 300, tol: 1e-6 }
    }
    pub fn n_clusters(mut self, k: usize) -> Self { self.n_clusters = k; self }
    pub fn method(mut self, m: KMeansInit) -> Self { self.method = m; self }
    pub fn max_iters(mut self, n: usize) -> Self { self.max_iters = n; self }
    pub fn tol(mut self, t: f64) -> Self { self.tol = t; self }
    pub fn build(self) -> KMeans {
        KMeans {
            n_clusters: self.n_clusters,
            method: self.method,
            max_iters: self.max_iters,
            tol: self.tol,
            centroids: None,
            inertia: Cell::new(f64::INFINITY),
            n_iter: Cell::new(0),
        }
    }
}

pub struct KMeans {
    n_clusters: usize,
    method: KMeansInit,
    max_iters: usize,
    tol: f64,
    centroids: Option<Array2<f64>>,
    inertia: Cell<f64>,
    n_iter: Cell<usize>,
}

impl KMeans {
    pub fn fit(&mut self, data: &ArrayView2<f64>) -> Result<(), SomError> {
        if self.centroids.is_some() {
            return Err(SomError::AlreadyFitted);
        }
        let dim = data.ncols();
        let min = data.fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut cents = match self.method {
            KMeansInit::Random => init_random(self.n_clusters, dim, min, max),
            KMeansInit::PlusPlus => init_som_plus_plus(data, self.n_clusters),
        };

        let mut prev_inertia = f64::INFINITY;
        let mut iter = 0;
        for _ in 0..self.max_iters {
            iter += 1;
            let labels = self.assign(data, &cents.view());
            let new_cents = self.update(data, &labels, dim);
            let shift = (&new_cents - &cents).mapv(|x| x * x).sum().sqrt();
            cents = new_cents;
            let inert = self.compute_inertia(data, &labels, &cents.view());
            if shift < self.tol || (prev_inertia - inert).abs() < self.tol {
                break;
            }
            prev_inertia = inert;
        }
        let final_labels = self.assign(data, &cents.view());
        self.inertia.set(self.compute_inertia(data, &final_labels, &cents.view()));
        self.n_iter.set(iter);
        self.centroids = Some(cents);
        Ok(())
    }

    fn assign(&self, data: &ArrayView2<f64>, cents: &ArrayView2<f64>) -> Array1<usize> {
        use crate::core::distance::batch_euclidean;
        let d = batch_euclidean(data, cents);
        d.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
    }

    fn update(&self, data: &ArrayView2<f64>, labels: &Array1<usize>, dim: usize) -> Array2<f64> {
        let mut new = Array2::<f64>::zeros((self.n_clusters, dim));
        let mut counts = vec![0usize; self.n_clusters];
        for (i, &c) in labels.iter().enumerate() {
            new.row_mut(c).scaled_add(1.0, &data.row(i));
            counts[c] += 1;
        }
        for (ci, &cnt) in counts.iter().enumerate() {
            if cnt > 0 {
                new.row_mut(ci).mapv_inplace(|x| x / cnt as f64);
            }
        }
        new
    }

    fn compute_inertia(
        &self,
        data: &ArrayView2<f64>,
        labels: &Array1<usize>,
        cents: &ArrayView2<f64>,
    ) -> f64 {
        labels
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let d = &data.row(i) - &cents.row(c);
                d.dot(&d)
            })
            .sum()
    }

    pub fn predict(&self, data: &ArrayView2<f64>) -> Result<Array1<usize>, SomError> {
        let cents = self.centroids.as_ref().ok_or(SomError::NotFitted("predict"))?;
        Ok(self.assign(data, &cents.view()))
    }

    pub fn centroids(&self) -> &Array2<f64> {
        self.centroids.as_ref().unwrap()
    }

    pub fn inertia(&self) -> Result<f64, SomError> {
        self.centroids.as_ref().ok_or(SomError::NotFitted("inertia"))?;
        Ok(self.inertia.get())
    }

    pub fn n_iter(&self) -> usize {
        self.n_iter.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn blobs() -> Array2<f64> {
        let mut d = Array2::<f64>::zeros((20, 2));
        for i in 0..10 { d[[i,0]] = i as f64 * 0.01; d[[i,1]] = i as f64 * 0.01; }
        for i in 10..20 { d[[i,0]] = 10.0 + i as f64 * 0.01; d[[i,1]] = 10.0; }
        d
    }

    #[test]
    fn centroids_shape() {
        let mut km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::Random).build();
        km.fit(&blobs().view()).unwrap();
        assert_eq!(km.centroids().shape(), &[2, 2]);
    }

    #[test]
    fn labels_in_range() {
        let mut km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::PlusPlus).build();
        km.fit(&blobs().view()).unwrap();
        let labels = km.predict(&blobs().view()).unwrap();
        assert!(labels.iter().all(|&l| l < 2));
    }

    #[test]
    fn already_fitted_error() {
        let mut km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::Random).build();
        km.fit(&blobs().view()).unwrap();
        assert!(matches!(km.fit(&blobs().view()), Err(crate::SomError::AlreadyFitted)));
    }

    #[test]
    fn inertia_not_fitted_error() {
        let km = KMeansBuilder::new().n_clusters(2).method(KMeansInit::Random).build();
        assert!(matches!(km.inertia(), Err(crate::SomError::NotFitted("inertia"))));
    }
}
