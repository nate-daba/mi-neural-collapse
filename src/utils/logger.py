# src/utils/logger.py

import numpy as np
from typing import Dict, Optional
from numpy.typing import NDArray
import os
import pandas as pd

class Logger:
    def __init__(self):
        self.logs: Dict[int, Dict[str, NDArray]] = {}

    def log_covariance(self, class_idx: int, cov_type: str, matrix: NDArray) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}
        self.logs[class_idx][cov_type] = matrix

    def log_eigenvalues(self, class_idx: int, eigvals: NDArray) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}
        self.logs[class_idx]['eigvals'] = eigvals

    def log_condition_number(self, class_idx: int, cond: float) -> None:
        if class_idx not in self.logs:
            self.logs[class_idx] = {}
        self.logs[class_idx]['cond'] = cond
    
    def save_to_npz(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for class_idx, values in self.logs.items():
            for name, matrix in values.items():
                np.savez(os.path.join(output_dir, f"class_{class_idx}_{name}.npz"), matrix)
                
    def log_determinants(self, class_idx: int, 
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

    def export_summary(self) -> pd.DataFrame:
        rows = []
        for cls, vals in self.logs.items():
            row = {
                'class': cls,
                'cond': vals.get('cond'),
                'eigval_max': np.max(vals.get('eigvals', np.array([np.nan]))),
                'eigval_min': np.min(vals.get('eigvals', np.array([np.nan]))),
                'det_x': vals.get('det_x'),
                'det_y': vals.get('det_y'),
                'det_joint': vals.get('det_joint'),
                'mi': vals.get('mi'),  # You log this using log_scalar(...)
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    def __repr__(self) -> str:
        return f"<Logger with logs for classes: {list(self.logs.keys())}>"