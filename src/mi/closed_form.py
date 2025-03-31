# src/mi/closed_form.py

import numpy as np
from numpy.typing import NDArray
from typing import Optional


class GaussianMI:
    """
    Computes mutual information between two multivariate Gaussian-distributed datasets
    using a closed-form expression based on their covariance matrices according to 
    https://statproofbook.github.io/P/mvn-mi.html
    """

    def __init__(self, regularization: float = 1e-5) -> None:
        """
        Args:
            regularization: Small value added to diagonal of covariances for numerical stability.
        """
        self.eps = regularization

    def compute_mi(self, X: NDArray, Y: NDArray) -> float:
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
        if det_joint <= 0 or det_x <= 0 or det_y <= 0:
            raise ValueError("Invalid determinant value; covariance matrices may be singular.")

        # Closed-form MI formula
        mi = 0.5 * np.log(det_x * det_y / det_joint)
        return float(mi)
    
    def compute_mi_cov(
        self,
        cov_x: NDArray,
        cov_y: NDArray,
        cov_joint: NDArray
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

        # Determinants
        det_joint = np.linalg.det(cov_joint)
        det_x = np.linalg.det(cov_x)
        det_y = np.linalg.det(cov_y)

        if det_joint <= 0 or det_x <= 0 or det_y <= 0:
            raise ValueError("Invalid determinant value; covariance matrices may be singular.")

        mi = 0.5 * np.log(det_x * det_y / det_joint)
        return float(mi)