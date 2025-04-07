# main.py
from tqdm import tqdm

from src.cli.parser import parse_args
from src.data.generator import GaussianFeatureGenerator
from src.mi.closed_form import GaussianMI
from src.metrics.mi_utils import compute_per_class_mi, save_mi_to_csv, compare_mi_estimators
from src.data.dataloader import get_data
from src.utils.utils import print_data_shapes, print_mi_results
from src.data.pca_projection import CIFAR10EigenProjector
from src.utils.logger import Logger

def main() -> None:
    args = parse_args()
    logger = Logger(args) if args.log else None

    classwise_data = get_data(args, logger)

    print_data_shapes(classwise_data)

    mi_calc = GaussianMI(logger=logger)
    mi_per_class = compute_per_class_mi(classwise_data, mi_calc,
                                    use_loop=args.use_loop,
                                    use_corr=args.use_corr)

    print_mi_results(mi_per_class)

    filepath = save_mi_to_csv(mi_per_class)
    print(f"\n[INFO] MI results saved to: {filepath}")

    summary_df = logger.export_summary()
    print(summary_df)
    logger.save_summary_csv(prefix="corr" if args.use_corr else "cov")

if __name__ == "__main__":
    main()