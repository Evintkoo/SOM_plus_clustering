use ndarray::Array2;

/// Gaussian neighborhood influence for a single neuron.
/// dist_sq: squared grid distance from BMU to this neuron.
pub fn gaussian(dist_sq: f64, learning_rate: f64, radius: f64) -> f64 {
    let r2 = (radius * radius).max(1e-18);
    learning_rate * (-dist_sq / (2.0 * r2)).exp()
}

/// Compute full m×n influence grid for BMU at (bmu_row, bmu_col).
pub fn gaussian_grid(
    m: usize, n: usize,
    bmu_row: usize, bmu_col: usize,
    learning_rate: f64, radius: f64,
) -> Array2<f64> {
    debug_assert!(bmu_row < m, "bmu_row {} out of bounds for m={}", bmu_row, m);
    debug_assert!(bmu_col < n, "bmu_col {} out of bounds for n={}", bmu_col, n);

    let mut grid = Array2::<f64>::zeros((m, n));
    let r2 = (radius * radius).max(1e-18);
    for i in 0..m {
        for j in 0..n {
            let dr = i as f64 - bmu_row as f64;
            let dc = j as f64 - bmu_col as f64;
            grid[[i, j]] = learning_rate * (-(dr * dr + dc * dc) / (2.0 * r2)).exp();
        }
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmu_gets_max_influence() {
        // At BMU itself (dist_sq=0), influence equals learning_rate
        let lr = 0.5_f64;
        let radius = 1.0_f64;
        let h = gaussian(0.0, lr, radius);
        assert!((h - lr).abs() < 1e-10);
    }

    #[test]
    fn far_neuron_gets_near_zero() {
        let h = gaussian(1000.0, 0.5, 0.1);
        assert!(h < 1e-10);
    }

    #[test]
    fn grid_shape_correct() {
        let grid = gaussian_grid(3, 4, 1, 2, 0.5, 1.0);
        assert_eq!(grid.shape(), &[3, 4]);
    }

    #[test]
    fn grid_bmu_is_max() {
        let m = 5; let n = 5;
        let bmu_r = 2; let bmu_c = 2;
        let grid = gaussian_grid(m, n, bmu_r, bmu_c, 0.5, 1.0);
        let max_val = grid[[bmu_r, bmu_c]];
        assert!(grid.iter().all(|&v| v <= max_val + 1e-12));
    }
}
