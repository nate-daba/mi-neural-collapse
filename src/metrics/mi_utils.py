import os
import csv
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from numpy.typing import NDArray
from src.mi.closed_form import GaussianMI

def compute_per_class_mi(
    classwise_data: Dict[int, Dict[str, NDArray]],
    mi_calc: GaussianMI,
    use_loop: bool = False,
    use_corr: bool = False
) -> Dict[int, float]:
    """
    Computes mutual information between train and test for each class.

    Args:
        classwise_data: Output from generate_classwise_train_test()
        mi_calc: An instance of GaussianMI

    Returns:
        Dictionary mapping class index to MI value
    """
    mi_per_class = {}
    for class_idx, data in classwise_data.items():
        X_train = data["train"]
        X_test = data["test"]
        if use_corr:
            mi = mi_calc.compute_mi_corr(X_train, X_test, class_idx)
        else:
            mi = mi_calc.compute_mi(X_train, X_test, use_loop=use_loop, class_idx=class_idx)
        mi_per_class[class_idx] = mi
    return mi_per_class

def save_mi_to_csv(mi_dict: Dict[int, float], 
                   base_dir: str = "results/mi_per_class") -> str:
    """
    Saves per-class mutual information to a timestamped CSV file.

    Args:
        mi_dict: Dictionary of class index to MI value
        base_dir: Base folder to store results in (default: 'results/mi_per_class')

    Returns:
        The full path to the saved CSV file
    """
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Filepath
    filepath = os.path.join(save_dir, "mi_per_class.csv")

    # Save CSV
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "MI"])
        for class_idx, mi in mi_dict.items():
            writer.writerow([class_idx, mi])

    return filepath

def compare_mi_estimators(classwise_data: Dict[int, Dict[str, NDArray]], mi_calc: GaussianMI) -> Dict[int, Tuple[float, float, float]]:
    comparison = {}

    for class_idx, data in classwise_data.items():
        X_train = data["train"]
        X_test = data["test"]
        cov = data["cov"]  # this is Σ

        # Empirical MI from samples
        est_mi = mi_calc.compute_mi(X_train, X_test, class_idx)

        # Ground-truth MI using Σ_X, Σ_Y = Σ and joint = block diagonal
        cov_x = cov
        cov_y = cov
        cov_joint = np.block([
            [cov_x, np.zeros_like(cov)],
            [np.zeros_like(cov), cov_y]
        ])
        
        true_mi = mi_calc.compute_mi_cov(cov_x, cov_y, cov_joint)

        error = abs(true_mi - est_mi)
        comparison[class_idx] = (est_mi, true_mi, error)

    return comparison

