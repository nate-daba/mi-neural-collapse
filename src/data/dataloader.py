from torchvision.datasets import CIFAR10
import numpy as np
from typing import Dict
from PIL import Image

def load_cifar10(
    root: str = "../../data",
    train: bool = True
) -> Dict[int, np.ndarray]:
    """
    Loads CIFAR-10 as raw numpy arrays grouped by class.

    Args:
        root: Path to store/download CIFAR-10
        train: Whether to load training data (True) or test data (False)

    Returns:
        A dictionary mapping class_index (0–9) to an array of shape (N, 3072),
        where N is the number of samples in that class.
    """
    dataset = CIFAR10(root=root, train=train, download=True)

    classwise_data: Dict[int, list[np.ndarray]] = {i: [] for i in range(10)}

    for img, label in dataset:
        # Convert PIL to numpy array and flatten to 1D (3072,)
        img_np = np.array(img).astype(np.float32).reshape(-1)  # (3072,)
        classwise_data[label].append(img_np)

    # Convert lists to stacked arrays
    for cls in classwise_data:
        classwise_data[cls] = np.stack(classwise_data[cls], axis=0)  # (N, 3072)

    return classwise_data

if __name__ == "__main__":
    classwise_train = load_cifar10(root="../../data", train=False)

    print(f"Loaded {len(classwise_train)} classes.")
    for cls, data in classwise_train.items():
        print(f"Class {cls}: shape = {data.shape}")

