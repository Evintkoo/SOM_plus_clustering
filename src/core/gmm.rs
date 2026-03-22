//! Diagonal-covariance Gaussian Mixture Model with EM.
//!
//! Used by AutoSom for BIC-based k-selection and as a third clustering
//! algorithm (alongside KMeans and DenSOM) for datasets with overlapping
//! Gaussian structure.

use crate::{core::kmeans::{KMeansBuilder, KMeansInit}, SomError};
use ndarray::{Array1, Array2, ArrayView2};
use rand::{SeedableRng, rngs::StdRng};

/// Minimum per-dimension variance — prevents log(0) and degenerate components.
const MIN_VAR: f64 = 1e-6;
/// ln(2π)
const LOG_2PI: f64 = 1.837_877_066_409_345_3;

// ─────────────────────────────────────────────────────────────────────────────

/// Fitted diagonal-covariance GMM.
pub struct Gmm {
    /// k × d mean matrix (row = component).
    pub means:          Array2<f64>,
    /// k × d per-dimension variances (diagonal covariance).
    pub vars:           Array2<f64>,
    /// k mixing weights (sums to 1).
    pub weights:        Array1<f64>,
    /// Final log-likelihood.
    pub log_likelihood: f64,
    /// BIC = -2·log_L + p·log(n), where p = k·(2d+1) − 1.
    pub bic:            f64,
    pub k:              usize,
    pub d:              usize,
    pub n:              usize,
}

impl Gmm {
    /// Fit a k-component diagonal GMM using EM.
    ///
    /// Uses KMeans++ warm initialization. Runs `n_restarts` independent
    /// attempts and keeps the one with the highest log-likelihood.
    ///
    /// Returns `Err(InsufficientData)` only if every restart fails.
    pub fn fit(
        data:       &ArrayView2<f64>,
        k:          usize,
        max_iters:  usize,
        tol:        f64,
        n_restarts: usize,
    ) -> Result<Self, SomError> {
        let n = data.nrows();
        let d = data.ncols();
        if n < k || k == 0 {
            return Err(SomError::InsufficientData { n });
        }

        let mut best: Option<Gmm> = None;

        for restart in 0..n_restarts {
            let seed = (n as u64)
                .wrapping_add((k as u64).wrapping_mul(31))
                .wrapping_add((restart as u64).wrapping_mul(97));
            let mut rng = StdRng::seed_from_u64(seed);

            let (means0, vars0, weights0) = init_params(data, k, &mut rng);

            if let Ok(gmm) = em_fit(data, means0, vars0, weights0, k, d, n, max_iters, tol) {
                if best.as_ref().map_or(true, |b: &Gmm| gmm.log_likelihood > b.log_likelihood) {
                    best = Some(gmm);
                }
            }
        }

        best.ok_or(SomError::InsufficientData { n })
    }

    /// Hard-assign each point to its highest-responsibility component.
    pub fn predict(&self, data: &ArrayView2<f64>) -> Array1<usize> {
        let log_resp = e_step_matrix(data, &self.means, &self.vars, &self.weights, self.k, self.d);
        Array1::from_shape_fn(data.nrows(), |i| {
            log_resp.row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, _)| c)
                .unwrap_or(0)
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BIC sweep
// ─────────────────────────────────────────────────────────────────────────────

/// Sweep k = 2..=max_k on `data`, fit GMM at each k, return the k with
/// minimum BIC.  Falls back to `k_fallback` if every fit fails.
pub fn gmm_bic_select_k(
    data:       &ArrayView2<f64>,
    max_k:      usize,
    k_fallback: usize,
) -> usize {
    let mut best_k   = k_fallback;
    let mut best_bic = f64::INFINITY;

    for k in 2..=max_k {
        // Skip k if too few points
        if data.nrows() < 2 * k { continue; }

        match Gmm::fit(data, k, 100, 1e-4, 3) {
            Ok(gmm) => {
                let bic = gmm.bic;
                if bic < best_bic {
                    best_bic = bic;
                    best_k   = k;
                }
            }
            Err(_) => continue,
        }
    }

    best_k
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute (n × k) log-responsibility matrix using the log-sum-exp trick.
fn e_step_matrix(
    data:    &ArrayView2<f64>,
    means:   &Array2<f64>,
    vars:    &Array2<f64>,
    weights: &Array1<f64>,
    k:       usize,
    d:       usize,
) -> Array2<f64> {
    let n = data.nrows();
    let mut log_resp = Array2::<f64>::zeros((n, k));

    for i in 0..n {
        let xi = data.row(i);
        let mut log_probs = vec![0.0f64; k];

        for ki in 0..k {
            let log_w = if weights[ki] > 0.0 {
                weights[ki].ln()
            } else {
                f64::NEG_INFINITY
            };
            let mut log_norm = 0.0f64;
            for j in 0..d {
                let v    = vars[[ki, j]].max(MIN_VAR);
                let diff = xi[j] - means[[ki, j]];
                log_norm += -0.5 * LOG_2PI - 0.5 * v.ln() - 0.5 * diff * diff / v;
            }
            log_probs[ki] = log_w + log_norm;
        }

        // log-sum-exp normalisation
        let max_lp = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_sum = if max_lp.is_finite() {
            max_lp + log_probs.iter().map(|&x| (x - max_lp).exp()).sum::<f64>().ln()
        } else {
            f64::NEG_INFINITY
        };

        for ki in 0..k {
            log_resp[[i, ki]] = log_probs[ki] - log_sum;
        }
    }
    log_resp
}

/// Initialise means, variances, and weights from KMeans++ hard assignments.
fn init_params(
    data: &ArrayView2<f64>,
    k:    usize,
    _rng: &mut StdRng, // kept for API symmetry; KMeans++ has its own RNG
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let n = data.nrows();
    let d = data.ncols();

    // KMeans++ for warm start
    let mut km = KMeansBuilder::new()
        .n_clusters(k)
        .method(KMeansInit::PlusPlus)
        .max_iters(50)
        .build();

    let labels: Array1<usize> = if km.fit(data).is_ok() {
        km.predict(data)
            .unwrap_or_else(|_| Array1::from_elem(n, 0))
    } else {
        Array1::from_elem(n, 0)
    };

    let mut means  = Array2::<f64>::zeros((k, d));
    let mut vars   = Array2::<f64>::from_elem((k, d), 1.0);
    let mut counts = vec![0usize; k];

    for i in 0..n {
        let c = labels[i];
        counts[c] += 1;
        for j in 0..d { means[[c, j]] += data[[i, j]]; }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..d { means[[c, j]] /= counts[c] as f64; }
        }
    }

    // Per-cluster, per-dimension sample variance
    for i in 0..n {
        let c = labels[i];
        for j in 0..d {
            let diff = data[[i, j]] - means[[c, j]];
            vars[[c, j]] += diff * diff;
        }
    }
    for c in 0..k {
        let cnt = counts[c].max(1) as f64;
        for j in 0..d {
            vars[[c, j]] = (vars[[c, j]] / cnt).max(MIN_VAR);
        }
    }

    let weights = Array1::from_elem(k, 1.0 / k as f64);
    (means, vars, weights)
}

/// EM loop. Returns a fitted `Gmm` or `InsufficientData` if all iterations diverge.
#[allow(clippy::too_many_arguments)]
fn em_fit(
    data:     &ArrayView2<f64>,
    mut means:   Array2<f64>,
    mut vars:    Array2<f64>,
    mut weights: Array1<f64>,
    k:        usize,
    d:        usize,
    n:        usize,
    max_iters: usize,
    tol:       f64,
) -> Result<Gmm, SomError> {
    let mut prev_ll = f64::NEG_INFINITY;

    for _iter in 0..max_iters {
        // ── E-step ────────────────────────────────────────────────────────── //
        let mut log_resp = Array2::<f64>::zeros((n, k));
        let mut ll       = 0.0f64;

        for i in 0..n {
            let xi = data.row(i);
            let mut log_probs = vec![0.0f64; k];

            for ki in 0..k {
                let log_w = if weights[ki] > 0.0 {
                    weights[ki].ln()
                } else {
                    f64::NEG_INFINITY
                };
                let mut log_norm = 0.0f64;
                for j in 0..d {
                    let v    = vars[[ki, j]].max(MIN_VAR);
                    let diff = xi[j] - means[[ki, j]];
                    log_norm += -0.5 * LOG_2PI - 0.5 * v.ln() - 0.5 * diff * diff / v;
                }
                log_probs[ki] = log_w + log_norm;
            }

            let max_lp  = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = if max_lp.is_finite() {
                max_lp + log_probs.iter().map(|&x| (x - max_lp).exp()).sum::<f64>().ln()
            } else {
                f64::NEG_INFINITY
            };

            if log_sum.is_finite() { ll += log_sum; }

            for ki in 0..k {
                log_resp[[i, ki]] = log_probs[ki] - log_sum;
            }
        }

        // ── Convergence check ─────────────────────────────────────────────── //
        if (ll - prev_ll).abs() < tol * prev_ll.abs().max(1.0) {
            let p   = k * (2 * d + 1) - 1;
            let bic = -2.0 * ll + p as f64 * (n as f64).ln();
            return Ok(Gmm { means, vars, weights, log_likelihood: ll, bic, k, d, n });
        }
        prev_ll = ll;

        // ── M-step ────────────────────────────────────────────────────────── //
        let mut new_means = Array2::<f64>::zeros((k, d));
        let mut new_vars  = Array2::<f64>::zeros((k, d));
        let mut nk        = Array1::<f64>::zeros(k);

        for i in 0..n {
            for ki in 0..k {
                let r = log_resp[[i, ki]].exp();
                nk[ki] += r;
                for j in 0..d {
                    new_means[[ki, j]] += r * data[[i, j]];
                }
            }
        }
        for ki in 0..k {
            let eff = nk[ki].max(1e-10);
            for j in 0..d { new_means[[ki, j]] /= eff; }
        }

        for i in 0..n {
            for ki in 0..k {
                let r = log_resp[[i, ki]].exp();
                for j in 0..d {
                    let diff = data[[i, j]] - new_means[[ki, j]];
                    new_vars[[ki, j]] += r * diff * diff;
                }
            }
        }
        for ki in 0..k {
            let eff = nk[ki].max(1e-10);
            for j in 0..d {
                new_vars[[ki, j]] = (new_vars[[ki, j]] / eff).max(MIN_VAR);
            }
        }

        let total = nk.sum().max(1e-10);
        weights = nk.mapv(|v| v / total);
        means   = new_means;
        vars    = new_vars;
    }

    // Return best-effort result after max_iters
    let p   = k * (2 * d + 1) - 1;
    let bic = -2.0 * prev_ll + p as f64 * (n as f64).ln();
    Ok(Gmm { means, vars, weights, log_likelihood: prev_ll, bic, k, d, n })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Two tight 2D blobs on a grid (rows × cols each), well-separated at +100 offset.
    fn two_blobs_grid(rows: usize, cols: usize) -> Array2<f64> {
        let n_per = rows * cols;
        let mut v = Vec::with_capacity(n_per * 4);
        // Blob 1: 2D grid at origin
        for i in 0..rows {
            for j in 0..cols {
                v.push(i as f64); v.push(j as f64);
            }
        }
        // Blob 2: same grid offset by +100 in both axes
        for i in 0..rows {
            for j in 0..cols {
                v.push(100.0 + i as f64); v.push(100.0 + j as f64);
            }
        }
        Array2::from_shape_vec((n_per * 2, 2), v).unwrap()
    }

    #[test]
    fn gmm_fit_two_blobs() {
        let data = two_blobs_grid(10, 10);
        let gmm  = Gmm::fit(&data.view(), 2, 100, 1e-4, 3).unwrap();
        assert_eq!(gmm.k, 2);
        assert!(gmm.log_likelihood.is_finite() && gmm.log_likelihood < 0.0,
            "log_likelihood must be finite negative, got {}", gmm.log_likelihood);
    }

    #[test]
    fn gmm_predict_separates_blobs() {
        let data   = two_blobs_grid(10, 10); // 200 points
        let gmm    = Gmm::fit(&data.view(), 2, 100, 1e-4, 3).unwrap();
        let labels = gmm.predict(&data.view());
        // Both labels must be used
        let uniq: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(uniq.len(), 2, "should produce 2 distinct labels, got {uniq:?}");
        // All points in the first blob should share a single label
        let label0 = labels[0];
        for i in 1..100 {
            assert_eq!(labels[i], label0, "blob-0 point {i} has wrong label");
        }
    }

    #[test]
    fn gmm_bic_selects_two_for_two_blobs() {
        let data   = two_blobs_grid(10, 10);
        let best_k = gmm_bic_select_k(&data.view(), 6, 2);
        assert_eq!(best_k, 2,
            "BIC should pick k=2 for two well-separated blobs, got k={best_k}");
    }

    #[test]
    fn gmm_bic_is_lower_for_better_fit() {
        let data = two_blobs_grid(10, 10);
        let gmm2 = Gmm::fit(&data.view(), 2, 100, 1e-4, 3).unwrap();
        let gmm5 = Gmm::fit(&data.view(), 5, 100, 1e-4, 3).unwrap();
        assert!(gmm2.bic < gmm5.bic,
            "BIC(k=2)={:.2} should be lower than BIC(k=5)={:.2} for two blobs",
            gmm2.bic, gmm5.bic);
    }
}
