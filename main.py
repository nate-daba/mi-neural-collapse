# main.py
from tqdm import tqdm

from src.cli.parser import parse_args
from src.data.generator import GaussianFeatureGenerator
from src.mi.closed_form import GaussianMI
from src.metrics.mi_utils import compute_per_class_mi, save_mi_to_csv, compare_mi_estimators
from src.data.dataloader import load_cifar10
from src.data.pca_projection import CIFAR10EigenProjector

def main() -> None:
    args = parse_args()

    if args.data == "cifar10":
        print("[INFO] Using CIFAR-10 dataset with PCA projection")

        train_data = load_cifar10(root="data", 
                                  train=True, 
                                  samples_per_class=args.samples_per_class)
        test_data  = load_cifar10(root="data", 
                                  train=False,
                                  samples_per_class=args.samples_per_class)

        projector = CIFAR10EigenProjector(args.num_eigv)
        classwise_data = {}

        for cls in tqdm(train_data, desc=f"Projecting train and test data (num_eigv={projector.num_eigv})"):
            X_train = train_data[cls]
            X_test = test_data[cls]
            X_train_proj, P, mean = projector.fit_project(X_train)
            X_test_proj = projector.transform(X_test, projection_matrix=P, mean=mean)

            classwise_data[cls] = {
                "train": X_train_proj,
                "test": X_test_proj
            }

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
        classwise_data = generator.generate_classwise_train_test()

    else:
        raise NotImplementedError(f"Dataset '{args.data}' is not supported yet.")

    for class_idx, data in classwise_data.items():
        X_train = data["train"]
        X_test = data["test"]
        print(f"Class {class_idx}: Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    mi_calc = GaussianMI()
    mi_per_class = compute_per_class_mi(classwise_data, mi_calc, use_loop=args.use_loop)

    print("\nPer-class MI (Train vs Test):")
    for class_idx, mi in mi_per_class.items():
        print(f"  Class {class_idx}: {mi:.4f} nats")
    
    filepath = save_mi_to_csv(mi_per_class)
    print(f"\n[INFO] MI results saved to: {filepath}")
    

if __name__ == "__main__":
    main()