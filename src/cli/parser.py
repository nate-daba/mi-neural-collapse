# src/cli/parser.py

import argparse
from typing import Tuple

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for Gaussian data generation.
    
    Returns:
        argparse.Namespace with attributes for all CLI parameters.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic Gaussian feature vectors")

    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes (C)")
    parser.add_argument("--num_features", type=int, default=50, help="Dimensionality of feature vectors (N)")
    parser.add_argument("--num_eigv", type=int, default=1000, help="Number of dominant eigenvectors to select for projection.")
    parser.add_argument("--samples_per_class", type=int, default=100, help="Samples per class (M / C)")
    parser.add_argument("--mean_low", type=float, default=-2.0, help="Lower bound for class mean sampling")
    parser.add_argument("--mean_high", type=float, default=2.0, help="Upper bound for class mean sampling")
    parser.add_argument("--cov_low", type=float, default=0.5, help="Lower bound for covariance scale")
    parser.add_argument("--cov_high", type=float, default=1.5, help="Upper bound for covariance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data", type=str, default="synthetic", choices=["synthetic", "cifar10", "cifar100", "imagenet"], help="Dataset to use for MI analysis")
    parser.add_argument("--use_loop", action="store_true", help="Enable boolean feature (default: False)")
    parser.add_argument("--use_corr", action="store_true", help="Use correlation matrices instead of covariance")
    parser.add_argument("--log", action="store_true", help="log intermediate results")


    return parser.parse_args()