use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand_distr::{Normal, Distribution};
use rand::Rng;

pub fn init_random(k: usize, dim: usize, min: f64, max: f64) -> Array2<f64> {
    let mut rng = rand::rng();
    Array2::from_shape_fn((k, dim), |_| {
        rng.random_range(min..max)
    })
}

pub fn init_he(dim: usize, k: usize) -> Array2<f64> {
    let std = (2.0_f64 / dim as f64).sqrt();
    let dist = Normal::new(0.0, std).unwrap();
    let mut rng = rand::rng();
    Array2::from_shape_fn((k, dim), |_| {
        dist.sample(&mut rng)
    })
}

pub fn init_lecun(dim: usize, k: usize) -> Array2<f64> {
    let std = (1.0_f64 / dim as f64).sqrt();
    let dist = Normal::new(0.0, std).unwrap();
    let mut rng = rand::rng();
    Array2::from_shape_fn((dim, k), |_| {
        dist.sample(&mut rng)
    })
}

pub fn init_zero(p: usize, q: usize) -> Array2<f64> {
    if p == q {
        return ndarray::Array2::eye(p);
    }
    let mut w = Array2::<f64>::zeros((p, q));
    let diag_len = p.min(q);
    for i in 0..diag_len {
        w[[i, i]] = 1.0;
    }
    w
}

pub fn init_naive_sharding(data: &ArrayView2<f64>, k: usize) -> Array2<f64> {
    let n = data.nrows();
    let dim = data.ncols();
    let mut sums: Vec<(f64, usize)> = (0..n)
        .map(|i| (data.row(i).sum(), i))
        .collect();
    sums.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let chunk = (n + k - 1) / k;
    let mut centroids = Array2::<f64>::zeros((k, dim));
    for ci in 0..k {
        let start = ci * chunk;
        let end = ((ci + 1) * chunk).min(n);
        if start >= end {
            continue;
        }
        let mut mean = ndarray::Array1::<f64>::zeros(dim);
        for &(_, idx) in &sums[start..end] {
            mean += &data.row(idx);
        }
        mean /= (end - start) as f64;
        centroids.row_mut(ci).assign(&mean);
    }
    centroids
}

pub fn init_som_plus_plus(data: &ArrayView2<f64>, k: usize) -> Array2<f64> {
    let n = data.nrows();
    let dim = data.ncols();
    let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
    let first = (0..n)
        .max_by(|&a, &b| {
            let da: f64 = (&data.row(a) - &mean).mapv(|x| x * x).sum();
            let db: f64 = (&data.row(b) - &mean).mapv(|x| x * x).sum();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();

    let mut selected = vec![first];
    let mut min_dist: Vec<f64> = (0..n)
        .map(|i| (&data.row(i) - &data.row(first)).mapv(|x| x * x).sum())
        .collect();

    for _ in 1..k {
        let next = (0..n)
            .filter(|i| !selected.contains(i))
            .max_by(|&a, &b| min_dist[a].partial_cmp(&min_dist[b]).unwrap())
            .unwrap();
        selected.push(next);
        for i in 0..n {
            let d: f64 = (&data.row(i) - &data.row(next)).mapv(|x| x * x).sum();
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }

    let mut out = Array2::<f64>::zeros((k, dim));
    for (ci, &si) in selected.iter().enumerate() {
        out.row_mut(ci).assign(&data.row(si));
    }
    out
}

/// LSUV initialization (SVD-free variant).
/// Seeds with He initialization and applies iterative variance scaling.
pub fn init_lsuv(input_dim: usize, output_dim: usize, x_batch: &ArrayView2<f64>) -> Array2<f64> {
    let std = (2.0_f64 / input_dim as f64).sqrt();
    let dist = Normal::new(0.0, std).unwrap();
    let mut rng = rand::rng();
    let mut weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
        dist.sample(&mut rng)
    });
    for _ in 0..10 {
        let acts = x_batch.dot(&weights);
        let var = acts.var(0.0);
        if var < 1e-8 {
            break;
        }
        if (var - 1.0).abs() < 0.1 {
            break;
        }
        weights.mapv_inplace(|x| x / var.sqrt());
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn dummy_data(n: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |(i, j)| (i * d + j) as f64 / (n * d) as f64)
    }

    #[test]
    fn random_shape() {
        let c = init_random(6, 3, 0.0, 1.0);
        assert_eq!(c.shape(), &[6, 3]);
    }

    #[test]
    fn he_shape_no_nan() {
        let c = init_he(3, 10);
        assert_eq!(c.shape(), &[10, 3]);
        assert!(c.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn lecun_shape_no_nan() {
        let c = init_lecun(3, 10);
        assert_eq!(c.shape(), &[3, 10]);
        assert!(c.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_init_identity() {
        let w = init_zero(3, 3);
        assert_eq!(w.shape(), &[3, 3]);
        assert!(w.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn naive_sharding_shape() {
        let data = dummy_data(20, 4);
        let c = init_naive_sharding(&data.view(), 5);
        assert_eq!(c.shape(), &[5, 4]);
        assert!(c.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn som_plus_plus_shape() {
        let data = dummy_data(30, 3);
        let c = init_som_plus_plus(&data.view(), 9);
        assert_eq!(c.shape(), &[9, 3]);
    }

    #[test]
    fn lsuv_shape_no_nan() {
        let data = dummy_data(20, 4);
        let c = init_lsuv(4, 8, &data.view());
        assert_eq!(c.shape(), &[4, 8]);
        assert!(c.iter().all(|x| x.is_finite()));
    }
}
