from typing import Dict
import numpy as np
from PIL import Image
from src.data.pca_projection import CIFAR10EigenProjector
from src.data.generator import GaussianFeatureGenerator
from src.utils.logger import Logger
from tqdm import tqdm
from numpy.typing import NDArray

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
        A dictionary mapping class_index (0–9) to an array of shape (samples_per_class, 3072)dddd
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



def get_data(args: , logger: Logger) -> Dict[int, Dict[str, NDArray]]:
    if args.data == "cifar10":
        print("[INFO] Using CIFAR-10 dataset with PCA projection")

        train_data = load_cifar10(root="data", train=True, 
                                  samples_per_class=args.samples_per_class)
        test_data  = load_cifar10(root="data", train=False, 
                                  samples_per_class=args.samples_per_class)

        projector = CIFAR10EigenProjector(num_eigv=args.num_eigv, logger=logger)
        classwise_data = {}

        for cls in tqdm(train_data, desc=f"Projecting train and test data (num_eigv={projector.num_eigv})"):
            X_train = train_data[cls]
            X_test = test_data[cls]
            X_train_proj, P, mean = projector.fit_project(X_train, 
                                                          class_id=cls, 
                                                          use_corr=args.use_corr)
            X_test_proj = projector.transform(X_test, 
                                              projection_matrix=P, 
                                              mean=mean,
                                              use_corr=args.use_corr)

            classwise_data[cls] = {
                "train": X_train_proj,
                "test": X_test_proj
            }

        return classwise_data

    elif args.data == "synthetic":
        print("[INFO] Using synthetic Gaussian data")
        generator = GaussianFeatureGenerator(
            num_classes=args.num_classes,
            num_features=args.num_features,
            samples_per_class=args.samples_per_class,
            mean_range=(args.mean_low, args.mean_high),
            cov_scale_range=(args.cov_low, args.cov_high),
            seed=args.seed,
        )
        return generator.generate_classwise_train_test()

    else:
        raise NotImplementedError(f"Dataset '{args.data}' is not supported yet.")

if __name__ == "__main__":
    classwise_train = load_cifar10(root="../../data", train=False, samples_per_class=100)

    print(f"Loaded {len(classwise_train)} classes.")
    for cls, data in classwise_train.items():
        print(f"Class {cls}: shape = {data.shape}")


