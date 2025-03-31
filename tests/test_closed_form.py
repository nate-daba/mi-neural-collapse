# tests/test_closed_form.py

import unittest
import numpy as np
from src.mi.closed_form import GaussianMI
from numpy.typing import NDArray

class TestGaussianMI(unittest.TestCase):
    """
    Unit tests for the GaussianMI class which computes closed-form mutual information
    for jointly Gaussian variables.
    """

    def setUp(self):
        """
        Create a GaussianMI instance before each test.
        """
        self.mi_calc = GaussianMI()
        self.n_samples = 10000
        self.rng = np.random.default_rng(seed=42)

    def test_bivariate_gaussian(self):
        """
        Estimate MI from samples of a known bivariate Gaussian and compare with analytical result.
        """
        rho = 0.8
        cov = np.array([[1.0, rho],
                        [rho, 1.0]])
        mean = np.zeros(2)
        samples = self.rng.multivariate_normal(mean, cov, size=self.n_samples)

        X = samples[:, [0]]
        Y = samples[:, [1]]

        mi_true = -0.5 * np.log(1 - rho**2)
        mi_est = self.mi_calc.compute_mi(X, Y)

        self.assertAlmostEqual(mi_true, mi_est, places=1,  # <- relaxed precision
            msg=f"Expected MI ≈ {mi_true:.4f}, but got {mi_est:.4f}")

    def test_bivariate_gaussian_cov_form(self):
        """
        Test compute_mi_cov() using known covariance matrices
        for a bivariate Gaussian with known correlation.
        """
        rho = 0.8
        cov_joint = np.array([[1.0, rho],
                            [rho, 1.0]])
        
        cov_x = np.array([[1.0]])
        cov_y = np.array([[1.0]])

        mi_true = -0.5 * np.log(1 - rho**2)
        mi_est = self.mi_calc.compute_mi_cov(cov_x, cov_y, cov_joint)
        self.assertAlmostEqual(mi_true, mi_est, places=4,
            msg=f"Expected MI ≈ {mi_true:.4f}, but got {mi_est:.4f}")
        
    def test_independent_gaussian(self):
        """
        Mutual information of independent Gaussians (rho = 0) should be near zero.
        """
        rho = 0.0
        cov_joint = np.array([[1.0, rho],
                            [rho, 1.0]])
        cov_x = np.array([[1.0]])
        cov_y = np.array([[1.0]])

        mi_true = 0.0
        mi_est = self.mi_calc.compute_mi_cov(cov_x, cov_y, cov_joint)

        self.assertAlmostEqual(mi_true, mi_est, places=5,
            msg=f"Expected MI ≈ 0.0 for rho=0, but got {mi_est:.5f}")
        
    def test_perfect_correlation(self):
        """
        Test that MI blows up (approaches infinity) as rho → ±1.
        """
        rho = 0.9999  # Not exactly 1.0 to avoid singular matrix
        cov_joint = np.array([[1.0, rho],
                            [rho, 1.0]])
        cov_x = np.array([[1.0]])
        cov_y = np.array([[1.0]])

        mi_est = self.mi_calc.compute_mi_cov(cov_x, cov_y, cov_joint)

        self.assertGreater(mi_est, 4.0,  # MI should be very large
            msg=f"Expected MI > 4.0 for rho=0.9999, but got {mi_est:.4f}")
        
    def test_marginal_variance_change(self):
        """
        MI should remain the same if rho is the same, even if variances differ.
        """
        rho = 0.5
        var_x = 2.0
        var_y = 5.0

        cov_joint = np.array([
            [var_x, rho * np.sqrt(var_x * var_y)],
            [rho * np.sqrt(var_x * var_y), var_y]
        ])
        cov_x = np.array([[var_x]])
        cov_y = np.array([[var_y]])

        mi_true = -0.5 * np.log(1 - rho**2)
        mi_est = self.mi_calc.compute_mi_cov(cov_x, cov_y, cov_joint)

        self.assertAlmostEqual(mi_true, mi_est, places=4,
            msg=f"Expected MI ≈ {mi_true:.4f}, got {mi_est:.4f}")
        
    def test_high_dimensional_mi(self):
        """
        Test compute_mi and compute_mi_cov on high-dimensional data (e.g., 50D).
        """
        dim = 50
        samples = self.n_samples

        # Create a joint covariance matrix for [X; Y] where X, Y ∈ ℝ^50
        rho = 0.5
        cov_xy = rho * np.eye(dim)
        cov_x = np.eye(dim)
        cov_y = np.eye(dim)
        cov_joint = np.block([
            [cov_x, cov_xy],
            [cov_xy.T, cov_y]
        ])
        mean = np.zeros(2 * dim)

        # Sample from joint Gaussian
        full = self.rng.multivariate_normal(mean, cov_joint, size=samples)
        X = full[:, :dim]
        Y = full[:, dim:]

        mi_est = self.mi_calc.compute_mi(X, Y)
        mi_true = self.mi_calc.compute_mi_cov(cov_x, cov_y, cov_joint)

        self.assertAlmostEqual(mi_true, mi_est, delta=0.2,
            msg=f"Expected high-dim MI ≈ {mi_true:.4f}, got {mi_est:.4f}")
        
    def test_mi_bounds_gaussian(self):
        """
        Validates that MI(X; Y) lies within the theoretical bounds:
            0 ≤ I(X; Y) ≤ min(H(X), H(Y))
        for jointly Gaussian X and Y.
        """
        dim = 20
        rho = 0.7
        samples = self.n_samples

        # Build joint covariance matrix for [X; Y]
        cov_xy = rho * np.eye(dim)
        cov_x = np.eye(dim)
        cov_y = np.eye(dim)
        cov_joint = np.block([
            [cov_x, cov_xy],
            [cov_xy.T, cov_y]
        ])
        mean = np.zeros(2 * dim)

        # Generate samples
        full = self.rng.multivariate_normal(mean, cov_joint, size=samples)
        X = full[:, :dim]
        Y = full[:, dim:]

        # Compute MI
        mi_calc = self.mi_calc
        mi = mi_calc.compute_mi(X, Y)

        # Compute marginal entropies (in nats)
        def entropy_gaussian(cov: NDArray) -> float:
            d = cov.shape[0]
            det = np.linalg.det(cov + 1e-5 * np.eye(d))  # regularize
            return 0.5 * np.log((2 * np.pi * np.e) ** d * det)

        h_x = entropy_gaussian(cov_x)
        h_y = entropy_gaussian(cov_y)

        # Check MI bounds
        self.assertGreaterEqual(mi, 0.0, "MI must be ≥ 0")
        self.assertLessEqual(mi, min(h_x, h_y), 
            f"MI ({mi:.4f}) exceeds min(H(X), H(Y)) = {min(h_x, h_y):.4f}")

if __name__ == '__main__':
    unittest.main()