import h5py
import numpy as np
from typing import Dict
from numpy.typing import NDArray

def load_cifar10_features(mat_path: str) -> Dict[int, Dict[str, NDArray]]:
    """
    Loads class-wise train/test features saved in MATLAB v7.3 format using h5py.

    Args:
        mat_path: Path to .mat file saved with -v7.3

    Returns:
        Dictionary: {class_idx: {'train': array, 'test': array}}
    """
    classwise_data = {}

    with h5py.File(mat_path, "r") as f:
        allFTrain = f["allFTrain"]
        allFTest = f["allFTest"]
        num_classes = allFTrain.shape[0]

        for i in range(num_classes):
            # MATLAB 1-based indexing and column-major memory layout
            train_ref = allFTrain[i][0]
            test_ref = allFTest[i][0]

            fTrain = np.array(f[train_ref]).T
            fTest = np.array(f[test_ref]).T

            classwise_data[i] = {
                "train": fTrain,
                "test": fTest,
            }

    return classwise_data

if __name__ == "__main__":
    data_path = "/Users/natnaeldaba/Documents/Documents/Academia/UofA/Research/neural_collapse/data/cifar10_pca_features.mat"
    data = load_cifar10_features(data_path)
    print(f"Loaded {len(data)} classes.")
    for k, v in data.items():
        print(f"Class {k}: train shape {v['train'].shape}, test shape {v['test'].shape}")