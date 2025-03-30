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
    parser.add_argument("--samples_per_class", type=int, default=100, help="Samples per class (M / C)")
    parser.add_argument("--mean_low", type=float, default=-2.0, help="Lower bound for class mean sampling")
    parser.add_argument("--mean_high", type=float, default=2.0, help="Upper bound for class mean sampling")
    parser.add_argument("--cov_low", type=float, default=0.5, help="Lower bound for covariance scale")
    parser.add_argument("--cov_high", type=float, default=1.5, help="Upper bound for covariance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()