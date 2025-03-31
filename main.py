# main.py

from src.cli.parser import parse_args
from src.data.generator import GaussianFeatureGenerator
from src.mi.closed_form import GaussianMI

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

    features, labels = generator.generate_class_distribution()

    print(f"Generated features shape: {features.shape}")
    print(f"Generated labels shape: {labels.shape}")
    print(f"Unique labels: {set(labels)}")
    
    # Split features into two independent sets (simulate train/test)
    midpoint = features.shape[0] // 2
    X = features[:midpoint]
    Y = features[midpoint:]

    mi_calc = GaussianMI()
    mi = mi_calc.compute_mi(X, Y)

    print(f"Closed-form Mutual Information (train/test split): {mi:.4f} nats")

if __name__ == "__main__":
    main()