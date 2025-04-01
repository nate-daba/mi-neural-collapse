from typing import Dict
import numpy as np
from PIL import Image

from torchvision.datasets import CIFAR10

def load_cifar10(
    root: str = "../../data",
    train: bool = True,
    samples_per_class: int = 5000
) -> Dict[int, np.ndarray]:
    """
    Loads CIFAR-10 as raw numpy arrays grouped by class.

    Args:
        root: Path to store/download CIFAR-10
        train: Whether to load training data (True) or test data (False)
        samples_per_class: Number of samples to load per class

    Returns:
        A dictionary mapping class_index (0–9) to an array of shape (samples_per_class, 3072)
    """
    dataset = CIFAR10(root=root, train=train, download=True)

    classwise_data: Dict[int, list[np.ndarray]] = {i: [] for i in range(10)}

    for img, label in dataset:
        if len(classwise_data[label]) < samples_per_class:
            img_np = np.array(img).astype(np.float32).reshape(-1) / 255.0
            classwise_data[label].append(img_np)

        # Early exit if all classes are filled
        if all(len(lst) >= samples_per_class for lst in classwise_data.values()):
            break

    # Convert lists to stacked arrays
    for cls in classwise_data:
        classwise_data[cls] = np.stack(classwise_data[cls], axis=0)  # (samples_per_class, 3072)

    return classwise_data

if __name__ == "__main__":
    classwise_train = load_cifar10(root="../../data", train=False, samples_per_class=100)

    print(f"Loaded {len(classwise_train)} classes.")
    for cls, data in classwise_train.items():
        print(f"Class {cls}: shape = {data.shape}")


