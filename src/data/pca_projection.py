import numpy as np
from typing import Dict, Tuple, Optional
from numpy.typing import NDArray
from .dataloader import load_cifar10
from tqdm import tqdm

from src.utils.logger import Logger

class CIFAR10EigenProjector:
    def __init__(self, num_eigv: int = 1000, 
                 logger: Optional[Logger] = None):
        """
        Initializes the eigen projector.

        Args:
            k: Number of top eigenvectors to retain per class.
        """
        self.num_eigv = num_eigv
        self.logger = logger

    def project_classwise(self, classwise_data: Dict[int, NDArray]) -> Dict[int, NDArray]:
        """
        Performs eigen-projection on each class's data.

        Args:
            classwise_data: Dict[class_id] = (N, 3072) feature vectors

        Returns:
            Dict[class_id] = (N, k) projected feature vectors
        """
        projected = {}

        for class_id, features in classwise_data.items():
            # Mean center
            mean = np.mean(features, axis=0)
            centered = features - mean

            # Covariance and eigendecomposition
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            if self.logger:
                self.logger.log_eigenvalues(class_id, eigvals)
                self.logger.log_covariance(class_id, 'eigenvectors', eigvecs)
                
            # Top-k eigenvectors (last k columns)
            top_k_vecs = eigvecs[:, -self.num_eigv:]  # shape: (3072, k)

            # Project onto top-k
            proj = centered @ top_k_vecs  # (N, k)
            projected[class_id] = proj

        return projected
    
    def fit_project(self, X: NDArray, class_id: int = 0) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Fit PCA on X and project X onto top-k components.

        Returns:
            Tuple of:
            - projected data (N, k)
            - projection matrix (D, k)
            - mean vector (D,)
        """
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        if self.logger:
            self.logger.log_eigenvalues(class_id, eigvals)
            
        top_k = eigvecs[:, -self.num_eigv:]
        X_proj = X_centered @ top_k
        return X_proj, top_k, mean

    def transform(self, X: NDArray, projection_matrix: NDArray, mean: NDArray) -> NDArray:
        """
        Projects new data onto previously computed eigenvectors.

        Args:
            X: (N, D) input data
            projection_matrix: (D, k)
            mean: (D,) mean used to center X

        Returns:
            Projected data (N, k)
        """
        return (X - mean) @ projection_matrix
    
if __name__ == "__main__":
    from dataloader import load_cifar10

    train_data = load_cifar10(root="../../data", train=True)
    test_data  = load_cifar10(root="../../data", train=False)

    projector = CIFAR10EigenProjector(k=1000)
    classwise_split = {}

    
    for cls in tqdm(train_data, desc=f"Projecting train and test data (num_egiv={projector.num_egiv})"):
        X_train = train_data[cls]
        X_test = test_data[cls]

        X_train_proj, P, mean = projector.fit_project(X_train, cls)
        X_test_proj = projector.transform(X_test, projection_matrix=P, mean=mean)

        classwise_split[cls] = {
            "train": X_train_proj,
            "test": X_test_proj
        }
        
    print(f"\nProjected train/test data per class (num_egiv={projector.num_egiv}):")
    for cls, split in classwise_split.items():
        print(f"Class {cls}: train = {split['train'].shape}, test = {split['test'].shape}")