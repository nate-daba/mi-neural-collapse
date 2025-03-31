# src/data/generator.py

import numpy as np
from typing import List, Tuple, Dict, TypedDict
from numpy.typing import NDArray
from src import config

class ClasswiseData(TypedDict):
    train: NDArray
    test: NDArray
    mean: NDArray
    cov: NDArray
    
class GaussianFeatureGenerator:
    """
    Generates synthetic multivariate Gaussian feature vectors for multiple classes.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        num_features: int = config.NUM_FEATURES,
        samples_per_class: int = config.SAMPLES_PER_CLASS,
        mean_range: Tuple[float, float] = config.DEFAULT_MEAN_RANGE,
        cov_scale_range: Tuple[float, float] = config.DEFAULT_COV_SCALE,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        """
        Initialize the generator with parameters for feature vector simulation.
        
        Args:
            num_classes: Number of classes (C)
            num_features: Feature vector length (N)
            samples_per_class: Number of samples per class (M / C)
            mean_range: Tuple of (min, max) for sampling class means
            cov_scale_range: Tuple of (min, max) for scaling identity covariance
            seed: Random seed for reproducibility
        """
        self.num_classes = num_classes
        self.num_features = num_features
        self.samples_per_class = samples_per_class
        self.mean_range = mean_range
        self.cov_scale_range = cov_scale_range
        self.rng = np.random.default_rng(seed)

    def generate_class_distribution(self) -> Tuple[NDArray, NDArray]:
        """
        Generate synthetic data for each class and concatenate them.

        Returns:
            features: (C*M, N) numpy array of feature vectors
            labels: (C*M,) numpy array of integer labels
        """
        all_features = []
        all_labels = []

        for class_idx in range(self.num_classes):
            mean = self.rng.uniform(
                *self.mean_range, size=self.num_features
            )
            cov_scale = self.rng.uniform(*self.cov_scale_range)
            cov = cov_scale * np.eye(self.num_features)

            samples = self.rng.multivariate_normal(
                mean=mean, cov=cov, size=self.samples_per_class
            )
            all_features.append(samples)
            all_labels.append(np.full(self.samples_per_class, class_idx))

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        return features, labels

    def generate_classwise_train_test(
        self,
        train_ratio: float = 0.5,
        return_covariances: bool = True
    ) -> Dict[int, ClasswiseData]:
        """
        Generates synthetic Gaussian feature vectors for each class,
        and splits them into train and test sets.

        Args:
            train_ratio: Proportion of samples to allocate to training
            return_covariances: Whether to include mean and covariance used

        Returns:
            Dictionary where keys are class indices and values are dictionaries with:
                - 'train': (n_train, N) array
                - 'test':  (n_test, N) array
                - 'mean':  (N,) array (optional)
                - 'cov':   (N, N) array (optional)
        """
        class_data: Dict[int, ClasswiseData] = {}

        for class_idx in range(self.num_classes):
            mean = self.rng.uniform(
                *self.mean_range, size=self.num_features
            )
            cov_scale = self.rng.uniform(*self.cov_scale_range)
            cov = cov_scale * np.eye(self.num_features)

            samples = self.rng.multivariate_normal(
                mean=mean, cov=cov, size=self.samples_per_class
            )

            # Shuffle and split
            self.rng.shuffle(samples)
            n_train = int(self.samples_per_class * train_ratio)
            train = samples[:n_train]
            test = samples[n_train:]

            class_data[class_idx] = {
                "train": train,
                "test": test,
                "mean": mean if return_covariances else None,
                "cov": cov if return_covariances else None,
            }

        return class_data