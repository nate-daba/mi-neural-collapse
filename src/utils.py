

def compute_per_class_mi(classwise_data: Dict[int, Dict[str, NDArray]], mi_calc: GaussianMI) -> Dict[int, float]:
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
        mi = mi_calc.compute_mi(X_train, X_test)
        mi_per_class[class_idx] = mi
    return mi_per_class