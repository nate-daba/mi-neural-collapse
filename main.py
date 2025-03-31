# main.py

from src.cli.parser import parse_args
from src.data.generator import GaussianFeatureGenerator
from src.mi.closed_form import GaussianMI
from src.metrics.mi_utils import compute_per_class_mi, save_mi_to_csv, compare_mi_estimators

def main() -> None:
    args = parse_args()

    generator = GaussianFeatureGenerator(
        num_classes=args.num_classes,
        num_features=args.num_features,
        samples_per_class=args.samples_per_class,
        mean_range=(args.mean_low, args.mean_high),
        cov_scale_range=(args.cov_low, args.cov_high),
        seed=args.seed,
    )
    
    classwise_data = generator.generate_classwise_train_test()

    for class_idx, data in classwise_data.items():
        X_train = data["train"]
        X_test = data["test"]
        print(f"Class {class_idx}: Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    mi_calc = GaussianMI()
    mi_per_class = compute_per_class_mi(classwise_data, mi_calc)

    print("\nPer-class MI (Train vs Test):")
    for class_idx, mi in mi_per_class.items():
        print(f"  Class {class_idx}: {mi:.4f} nats")
    
    filepath = save_mi_to_csv(mi_per_class)
    print(f"\n[INFO] MI results saved to: {filepath}")
    
    comparison = compare_mi_estimators(classwise_data, mi_calc)

    print("\nComparison: Empirical MI vs Ground-Truth MI")
    print("Class |    MI_est    |    MI_true   |   Abs Error")
    print("------|--------------|--------------|-------------")
    for class_idx, (est, truth, err) in comparison.items():
        print(f"{class_idx:>5} | {est:>12.4f} | {truth:>12.4f} | {err:>11.4f}")

if __name__ == "__main__":
    main()