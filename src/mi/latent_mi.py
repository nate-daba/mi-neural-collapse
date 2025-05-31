import numpy as np
import torch
from typing import Dict, Tuple
from latentmi import lmi

def compute_latent_mi(predictions: Dict[int, Dict[str, list]],
                      latent_dim: int = 200,
                      estimate_on_val: bool = False) -> float:
    """
    Computes the mutual information between latent features of train and test
    NN pairs using the LMI framework.

    Args:
        predictions (dict): Dictionary with 'train_feats' and 'test_feats' for each class.
        latent_dim (int): Dimensionality of the latent features.
        estimate_on_val (bool): Whether to use a holdout set during estimation.

    Returns:
        float: Estimated mutual information.
    """
    Xs, Ys = [], []

    for class_idx, data in predictions.items():
        train_feats = data['train_feats']  # List of np arrays
        test_feats = data['test_feats']    # List of np arrays

        for t_feat, s_feat in zip(train_feats, test_feats):
            Xs.append(np.squeeze(s_feat))
            Ys.append(np.squeeze(t_feat))

    Xs = np.stack(Xs)
    Ys = np.stack(Ys)

    # Estimate MI
    print(f"[INFO] Estimating latent MI with {Xs.shape[0]} samples and {latent_dim} latent dimensions.")
    if estimate_on_val:
        print("[INFO] Using validation set for estimation.")
    else:
        print("[INFO] Using training set for estimation.")
    pmis, _, _ = lmi.estimate(Xs, Ys, 
                              N_dims=latent_dim, 
                              estimate_on_val=estimate_on_val, 
                              quiet=True)
    print("\n")
    return np.nanmean(pmis)