from typing import Dict
import numpy as np
from numpy.typing import NDArray

def print_data_shapes(classwise_data: Dict[int, Dict[str, NDArray]]) -> None:
    """
    Prints the shape of train and test data for each class.
    """
    for class_idx, data in classwise_data.items():
        X_train = data["train"]
        X_test = data["test"]
        print(f"Class {class_idx}: Train shape {X_train.shape}, Test shape {X_test.shape}")

def print_mi_results(mi_per_class: Dict[int, float]) -> None:
    """
    Prints the mutual information for each class.
    """
    print("\nPer-class MI (Train vs Test):")
    for class_idx, mi in mi_per_class.items():
        print(f"  Class {class_idx}: {mi:.4f} nats")