# src/mi/closed_form.py

import sys
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from typing import Optional

from src.utils.logger import Logger


class GaussianMI:
    """
    Computes mutual information between two multivariate Gaussian-distributed datasets
    using a closed-form expression based on their covariance matrices according to 
    https://statproofbook.github.io/P/mvn-mi.html
    """

    def __init__(self, regularization: float = 1e-5, 
                 logger: Optional[Logger] = None) -> None:
        """
        Args:
            regularization: Small value added to diagonal of covariances for numerical stability.
        """
        self.eps = regularization
        self.logger = logger

    def compute_mi0(self, X: NDArray, Y: NDArray) -> float:
        """
        Computes the closed-form mutual information I(X; Y) for jointly Gaussian variables.

        Args:
            X: (n_samples, n_features_x) array representing the first variable
            Y: (n_samples, n_features_y) array representing the second variable

        Returns:
            Mutual Information in nats (use log base 2 for bits)
        """
        # Stack data horizontally to get joint matrix
        Z = np.hstack((X, Y)) 

        # Center the data
        Z -= Z.mean(axis=0)
        X -= X.mean(axis=0)
        Y -= Y.mean(axis=0)

        # Compute covariance matrices
        cov_z = np.cov(Z, rowvar=False) + self.eps * np.eye(Z.shape[1])
        cov_x = np.cov(X, rowvar=False) + self.eps * np.eye(X.shape[1])
        cov_y = np.cov(Y, rowvar=False) + self.eps * np.eye(Y.shape[1])
        
        # Compute determinants
        det_joint = np.linalg.det(cov_z)
        det_x = np.linalg.det(cov_x)
        det_y = np.linalg.det(cov_y)

        # Avoid log(0) or det < 0 from numerical issues
        # if det_joint <= 0 or det_x <= 0 or det_y <= 0:
        #     raise ValueError("Invalid determinant value; covariance matrices may be singular.")
        
        # Closed-form MI formula
        mi = 0.5 * np.log(det_x * det_y / det_joint)
        return float(mi)

    def compute_mi(self, X: NDArray, Y: NDArray, 
                   use_loop: bool = True, 
                   class_idx: int = 0) -> float:
        """
        Computes mutual information I(X; Y) using full covariance construction:
        cov_joint = [[cov_x, cov_xy],
                    [cov_yx, cov_y]]
        This works even if X and Y have different number of samples.
        """
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)

        cov_x = np.cov(X, rowvar=False) + self.eps * np.eye(X.shape[1])
        cov_y = np.cov(Y, rowvar=False) + self.eps * np.eye(Y.shape[1])
        cov_xy = self.compute_cross_cov_parallel(X, Y) if use_loop else self.compute_cross_cov(X, Y)
        
        # Joint covariance
        cov_joint = np.block([
            [cov_x, cov_xy],
            [cov_xy.T, cov_y]
        ])

        if self.logger:
            self.logger.log_cov(class_idx, 'cov_x', cov_x)
            self.logger.log_cov(class_idx, 'cov_y', cov_y)
            self.logger.log_cov(class_idx, 'cov_joint', cov_joint)
            self.logger.log_cond_num(class_idx, np.linalg.cond(cov_joint))
            
        return self.compute_mi_cov(cov_x, cov_y, cov_joint, class_idx)

    def compute_mi_cov(
        self,
        cov_x: NDArray,
        cov_y: NDArray,
        cov_joint: NDArray,
        class_idx: int = 0
    ) -> float:
        """
        Computes mutual information I(X; Y) given known covariance matrices for
        X, Y, and their joint distribution. Assumes Gaussian distributions.

        Args:
            cov_x: (n_features_x, n_features_x) covariance matrix of X
            cov_y: (n_features_y, n_features_y) covariance matrix of Y
            cov_joint: (n_features_x + n_features_y, n_features_x + n_features_y) joint covariance of [X; Y]

        Returns:
            Mutual Information in nats.
        """
        # Regularize to ensure numerical stability
        cov_x += self.eps * np.eye(cov_x.shape[0])
        cov_y += self.eps * np.eye(cov_y.shape[0])
        cov_joint += self.eps * np.eye(cov_joint.shape[0])
        print(f"[DEBUG] Condition number of cov_joint: {np.linalg.cond(cov_joint):.2e}")
        # Determinants
        det_joint = np.linalg.det(cov_joint)
        det_x = np.linalg.det(cov_x)
        det_y = np.linalg.det(cov_y)
        print(f"[DEBUG] det_x: {det_x:.3e}, det_y: {det_y:.3e}, det_joint: {det_joint:.3e}")
        if self.logger:
            self.logger.log_det(class_idx, det_x, det_y, det_joint)
        # if det_joint <= 0 or det_x <= 0 or det_y <= 0:
        #     raise ValueError("Invalid determinant value; covariance matrices may be singular.")

        mi = 0.5 * np.log2(det_x * det_y / det_joint)
        
        # sign_joint, logdet_joint = np.linalg.slogdet(cov_joint)
        # sign_x, logdet_x = np.linalg.slogdet(cov_x)
        # sign_y, logdet_y = np.linalg.slogdet(cov_y)
        # print(f"[DEBUG] logdet_x: {logdet_x:.3e}, logdet_y: {logdet_y:.3e}, det_joint: {logdet_joint:.3e}")
        # if sign_joint <= 0 or sign_x <= 0 or sign_y <= 0:
        #     raise ValueError("Invalid sign in log-determinant; covariance may not be positive definite.")

        # mi = 0.5 * (logdet_x + logdet_y - logdet_joint)

        if self.logger:
            self.logger.log_scalar(class_idx, 'mi', mi)

        return float(mi)

    @staticmethod
    def compute_cross_cov(X: NDArray, Y: NDArray) -> NDArray:
        """
        Computes cross-covariance matrix Σ12 = (1 / (n * m)) * sum_{i,j} (x_i - μ_x)(y_j - μ_y)^T
        using vectorized einsum.

        Args:
            X: (n, d)
            Y: (m, d)

        Returns:
            Σ12: (d, d)
        """
        print(f"Computing cross-covariance b/n train (X: {X.shape}) and test (Y: {Y.shape})")
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)
        X_centered = X - mu_x  # (n, d)
        Y_centered = Y - mu_y  # (m, d)

        # Vectorized outer product of all pairs
        cov12 = np.einsum('ni,mj->ij', X_centered, Y_centered) / (X.shape[0] * Y.shape[0])
        return cov12

    @staticmethod
    def compute_cross_cov0(X: NDArray, Y: NDArray) -> NDArray:
        """
        Computes cross-covariance matrix Σ12 = (1 / (n * m)) * sum_{i,j} (x_i - μ_x)(y_j - μ_y)^T
        using two nested for-loops (non-vectorized version).

        Args:
            X: (n, d)
            Y: (m, d)

        Returns:
            Σ12: (d, d)
        """
        n, d = X.shape
        m = Y.shape[0]

        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)

        cov12 = np.zeros((d, d))
        for i in tqdm(range(n), desc="Computing cross-covariance b/n train and test"):
            for j in range(m):
                diff_x = X[i] - mu_x
                diff_y = Y[j] - mu_y
                cov12 += np.outer(diff_x, diff_y)
        cov12 /= (n * m)
        return cov12
    
    @staticmethod
    def compute_cross_cov_parallel(X: NDArray, Y: NDArray, 
                                    center: bool = True,
                                    n_jobs: int = -1) -> NDArray:
        n, d = X.shape
        m = Y.shape[0]

        if center:
            mu_x = X.mean(axis=0)
            mu_y = Y.mean(axis=0)
            X = X - mu_x
            Y = Y - mu_y

        def partial_sum(i):
            return sum(np.outer(X[i], Y[j]) for j in range(m))

        partials = Parallel(n_jobs=n_jobs)(
            delayed(partial_sum)(i) for i in tqdm(range(n), desc="Parallel cross-cov")
        )

        cov = sum(partials) / (n * m)
        return cov
    
    def compute_mi_corr(
        self,
        X: NDArray,
        Y: NDArray,
        class_idx: int = 0,
        use_loop: bool = True
    ) -> float:
        """
        Computes mutual information I(X; Y) using correlation matrices,
        i.e., E[XY^T] instead of centered covariance.

        This method preserves the means of X and Y (i.e., no mean-centering).
        """
        n, d = X.shape
        m = Y.shape[0]

        # Uncentered correlation matrices
        corr_x = X.T @ X / n       # (d, d)
        corr_y = Y.T @ Y / m       # (d, d)
        if use_loop:
            corr_xy = self.compute_cross_cov_parallel(X, Y, center=False)
        else:
            corr_xy = X.T @ Y / (n * m)

        # Full correlation matrix
        corr_joint = np.block([
            [corr_x, corr_xy],
            [corr_xy.T, corr_y]
        ])

        # Regularize all
        corr_x += self.eps * np.eye(d)
        corr_y += self.eps * np.eye(d)
        corr_joint += self.eps * np.eye(2 * d)

        # Logging
        if self.logger:
            self.logger.log_cov(class_idx, 'corr_x', corr_x)
            self.logger.log_cov(class_idx, 'corr_y', corr_y)
            self.logger.log_cov(class_idx, 'corr_xy', corr_xy)
            self.logger.log_cov(class_idx, 'corr_joint', corr_joint)
            self.logger.log_cond_num(class_idx, np.linalg.cond(corr_joint))

        # Determinants
        det_joint = np.linalg.det(corr_joint)
        det_x = np.linalg.det(corr_x)
        det_y = np.linalg.det(corr_y)

        if self.logger:
            self.logger.log_det(class_idx, det_x, det_y, det_joint)

        # MI from determinant ratio
        mi = 0.5 * np.log(det_x * det_y / det_joint)
        if self.logger:
            self.logger.log_scalar(class_idx, 'mi_corr', mi)

        return float(mi)

