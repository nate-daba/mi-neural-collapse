"""
Neural Collapse Analyzer Module

This module provides functionality to compute and track Neural Collapse metrics
during deep learning model training, specifically focusing on the four main
neural collapse phenomena (NC1-NC4).

"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neural_collapse.accumulate import (
    CovarAccumulator, DecAccumulator, MeanAccumulator, VarNormAccumulator
)
from neural_collapse.kernels import kernel_stats, log_kernel
from neural_collapse.measure import (
    clf_ncc_agreement, covariance_ratio, self_duality_error, 
    similarities, simplex_etf_error, variability_cdnv
)


class NeuralCollapseAnalyzer:
    """
    Analyzer class for computing Neural Collapse metrics during training.
    
    This class implements the four main Neural Collapse phenomena:
    - NC1: Within-class variability collapse
    - NC2: Convergence to simplex ETF (Equiangular Tight Frame)
    - NC3: Convergence to self-duality
    - NC4: Simplification to Nearest Class Center (NCC)
    """
    
    def __init__(
        self, 
        num_classes: int, 
        feature_dim: int, 
        device: torch.device,
        tile_size: int = 64
    ) -> None:
        """
        Initialize the Neural Collapse Analyzer.
        
        Args:
            num_classes: Number of classes in the dataset
            feature_dim: Dimension of the feature space (penultimate layer)
            device: PyTorch device (CPU/GPU)
            tile_size: Tile size for efficient computation (default: 64)
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.tile_size = tile_size
        
        # Feature storage for hook
        self.features: Optional[torch.Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    
    def register_feature_hook(self, model: nn.Module) -> None:
        """
        Register a forward hook to capture penultimate layer features.
        
        Args:
            model: PyTorch model with a 'fc' final layer
        """
        def feature_hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """Hook function to capture input features to the final classifier."""
            self.features = input[0].clone().detach()
        
        self._hook_handle = model.fc.register_forward_hook(feature_hook)
    
    def remove_feature_hook(self) -> None:
        """Remove the registered feature hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def compute_nc_metrics(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Compute all Neural Collapse metrics.
        
        Args:
            model: Trained PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            
        Returns:
            Dictionary containing all NC metrics
        """
        model.eval()
        
        with torch.no_grad():
            # Get classifier weights
            weights = model.fc.weight
            
            # Step 1: Collect class means from training data
            means, global_mean = self._collect_class_means(model, train_loader)
            
            # Step 2: Collect within-class statistics
            within_class_stats = self._collect_within_class_stats(
                model, train_loader, means
            )
            
            # Step 3: Collect decision statistics on test data
            decision_stats = self._collect_decision_stats(
                model, test_loader, means, weights
            )
            
            # Step 4: Compute NC metrics
            nc_metrics = self._compute_all_nc_metrics(
                means, global_mean, weights, within_class_stats, decision_stats
            )
        
        return nc_metrics
    
    def _collect_class_means(
        self, 
        model: nn.Module, 
        train_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect class means from training data.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            
        Returns:
            Tuple of (class_means, global_mean)
        """
        mean_accum = MeanAccumulator(self.num_classes, self.feature_dim, self.device)
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            _ = model(images)  # Forward pass to trigger hook
            
            if self.features is not None:
                mean_accum.accumulate(self.features, labels)
        
        means, global_mean = mean_accum.compute()
        return means, global_mean
    
    def _collect_within_class_stats(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        means: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Collect within-class statistics.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            means: Class means tensor
            
        Returns:
            Dictionary containing within-class statistics
        """
        var_norms_accum = VarNormAccumulator(
            self.num_classes, self.feature_dim, self.device, M=means
        )
        covar_accum = CovarAccumulator(
            self.num_classes, self.feature_dim, self.device, M=means
        )
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            _ = model(images)  # Forward pass to trigger hook
            
            if self.features is not None:
                var_norms_accum.accumulate(self.features, labels, means)
                covar_accum.accumulate(self.features, labels, means)
        
        var_norms, _ = var_norms_accum.compute()
        covar_within = covar_accum.compute()
        
        return {
            'var_norms': var_norms,
            'covar_within': covar_within
        }
    
    def _collect_decision_stats(
        self, 
        model: nn.Module, 
        test_loader: DataLoader, 
        means: torch.Tensor, 
        weights: torch.Tensor
    ) -> DecAccumulator:
        """
        Collect decision statistics on test data.
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            means: Class means tensor
            weights: Classifier weights
            
        Returns:
            DecAccumulator with decision statistics
        """
        dec_accum = DecAccumulator(
            self.num_classes, self.feature_dim, self.device, M=means, W=weights
        )
        dec_accum.create_index(means)  # Use FAISS index for efficiency
        
        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            _ = model(images)  # Forward pass to trigger hook
            
            if self.features is not None:
                if dec_accum.index is None:
                    dec_accum.accumulate(self.features, labels, weights, means)
                else:
                    dec_accum.accumulate(self.features, labels, weights)
        
        return dec_accum
    
    def _compute_all_nc_metrics(
        self,
        means: torch.Tensor,
        global_mean: torch.Tensor,
        weights: torch.Tensor,
        within_class_stats: Dict[str, torch.Tensor],
        decision_stats: DecAccumulator
    ) -> Dict[str, float]:
        """
        Compute all Neural Collapse metrics.
        
        Args:
            means: Class means tensor
            global_mean: Global mean tensor
            weights: Classifier weights
            within_class_stats: Within-class statistics
            decision_stats: Decision statistics accumulator
            
        Returns:
            Dictionary containing all NC metrics
        """
        var_norms = within_class_stats['var_norms']
        covar_within = within_class_stats['covar_within']
        
        # Helper function to extract scalar values
        def get_scalar(value: Any) -> float:
            """Extract scalar value from tensor or float."""
            if hasattr(value, 'item'):
                return float(value.item())
            else:
                return float(value)
        
        # Compute NC metrics
        nc_metrics = {
            # NC1: Within-class variability collapse
            "nc1_pinv": get_scalar(covariance_ratio(covar_within, means, global_mean)),
            "nc1_svd": get_scalar(covariance_ratio(covar_within, means, global_mean, "svd")),
            "nc1_quot": get_scalar(covariance_ratio(covar_within, means, global_mean, "quotient")),
            "nc1_cdnv": get_scalar(variability_cdnv(var_norms, means, tile_size=self.tile_size)),
            
            # NC2: Convergence to simplex ETF
            "nc2_etf_err": get_scalar(simplex_etf_error(means, global_mean)),
            "nc2g_dist": get_scalar(kernel_stats(means, global_mean, tile_size=self.tile_size)[1]),
            "nc2g_log": get_scalar(kernel_stats(means, global_mean, kernel=log_kernel, tile_size=self.tile_size)[1]),
            
            # NC3: Convergence to self-duality
            "nc3_dual_err": get_scalar(self_duality_error(weights, means, global_mean)),
            "nc3u_uni_dual": get_scalar(similarities(weights, means, global_mean).var()),
            
            # NC4: Simplification to NCC
            "nc4_agree": get_scalar(clf_ncc_agreement(decision_stats)),
        }
        
        return nc_metrics
    
    def __del__(self) -> None:
        """Cleanup hook on destruction."""
        self.remove_feature_hook()