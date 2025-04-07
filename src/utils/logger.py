# src/utils/logger.py
import os
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


class Logger:
    def __init__(self, args: argparse.Namespace = None, output_dir: str = "results/plots", ):
        self.logs: Dict[int, Dict[str, NDArray]] = {}
        self.output_dir = output_dir
        self.args = args
    def log_cov(self, class_idx: int, cov_type: str, matrix: NDArray) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}
        self.logs[class_idx][cov_type] = matrix
        self.plot_cov_mat(matrix, class_idx, cov_type)
        self.log_offdiag_ratio(class_idx, matrix, f"{cov_type}_offdiag_ratio")

    def log_eigv(self, class_idx: int, eigvals: NDArray) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}
        self.logs[class_idx]['eigvals'] = eigvals

    def log_cond_num(self, class_idx: int, cond: float) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}
        self.logs[class_idx]['cond'] = cond
    
    def save_to_npz(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for class_idx, values in self.logs.items():
            for name, matrix in values.items():
                np.savez(os.path.join(output_dir, f"class_{class_idx}_{name}.npz"), matrix)
                
    def log_det(self, class_idx: int, 
                         det_x: float, 
                         det_y: float, 
                         det_joint: float) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}

        self.logs[class_idx]['det_x'] = det_x
        self.logs[class_idx]['det_y'] = det_y
        self.logs[class_idx]['det_joint'] = det_joint
        
    def log_scalar(self, class_idx: int, key: str, value: float) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}

        # Avoid overwriting if multiple scalars are logged under the same key
        if key in self.logs[class_idx]:
            raise KeyError(f"[Logger] Key '{key}' already logged for class {class_idx}")

        self.logs[class_idx][key] = value
        
    def plot_cov_mat(
        self,
        matrix: NDArray,
        class_idx: int,
        name: str
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",               # show values with 1 decimal point
            annot_kws={"size": 6},   # smaller font size for cell numbers
            cmap="coolwarm",
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False
        )
        plt.title(f"Covariance Matrix: {name} (Class {class_idx})")
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"class_{class_idx}_{name}_covariance.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        
    def log_offdiag_ratio(self, class_idx: int, matrix: NDArray, key: str) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}

        diag = np.diag(np.diag(matrix))
        off_diag_norm = np.linalg.norm(matrix - diag)
        diag_norm = np.linalg.norm(diag)
        ratio = off_diag_norm / (diag_norm + 1e-10)
        self.logs[class_idx][key] = ratio
        
    def export_summary(self) -> pd.DataFrame:
        rows = []
        mi_type = 'mi_corr' if self.args.use_corr else 'mi'
        for cls, vals in self.logs.items():
            row = {
                'class': cls,
                'cond': vals.get('cond'),
                'eigval_max': np.max(vals.get('eigvals', np.array([np.nan]))),
                'eigval_min': np.min(vals.get('eigvals', np.array([np.nan]))),
                'det_x': vals.get('det_x'),
                'det_y': vals.get('det_y'),
                'det_joint': vals.get('det_joint'),
                mi_type: vals.get(mi_type), 
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    def save_summary_csv(self, path: str = "results/summary.csv", prefix: Optional[str] = None) -> None:
        df = self.export_summary()
        
        # Adjust filename if prefix is provided
        if prefix:
            dir_name, base_name = os.path.split(path)
            name, ext = os.path.splitext(base_name)
            path = os.path.join(dir_name, f"{prefix}_{name}{ext}")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    
    def __repr__(self) -> str:
        return f"<Logger with logs for classes: {list(self.logs.keys())}>"