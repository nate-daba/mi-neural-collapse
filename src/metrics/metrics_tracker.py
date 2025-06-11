"""
Comprehensive Metrics Tracker Module

This module provides a unified interface for tracking various metrics during
deep learning model training, including accuracy, loss, mutual information,
and neural collapse metrics.

"""
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from nc.nc import NeuralCollapseAnalyzer


class MetricsTracker:
    """
    Comprehensive metrics tracker for deep learning experiments.
    
    This class provides a unified interface for computing and tracking:
    - Basic metrics (accuracy, loss)
    - Mutual information metrics
    - Neural collapse metrics
    - Model activations and gradients
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        device: torch.device,
        track_activations: bool = True,
        track_gradients: bool = True,
        track_nc_metrics: bool = True,
        track_mi_metrics: bool = True,
        nc_tile_size: int = 64
    ) -> None:
        """
        Initialize the comprehensive metrics tracker.
        
        Args:
            num_classes: Number of classes in the dataset
            feature_dim: Feature dimension of the model
            device: PyTorch device
            track_activations: Whether to track layer activations
            track_gradients: Whether to track gradient statistics
            track_nc_metrics: Whether to track neural collapse metrics
            track_mi_metrics: Whether to track mutual information metrics
            nc_tile_size: Tile size for neural collapse computations
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.track_activations = track_activations
        self.track_gradients = track_gradients
        self.track_nc_metrics = track_nc_metrics
        self.track_mi_metrics = track_mi_metrics
        
        # Initialize neural collapse analyzer if needed
        if self.track_nc_metrics:
            self.nc_analyzer = NeuralCollapseAnalyzer(
                num_classes=num_classes,
                feature_dim=feature_dim,
                device=device,
                tile_size=nc_tile_size
            )
        else:
            self.nc_analyzer = None
        
        # Storage for activation tracking
        self.activations: Dict[str, torch.Tensor] = {}
        self.activation_hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Storage for features (for MI computation)
        self.features: Optional[torch.Tensor] = None
        self.feature_hook: Optional[torch.utils.hooks.RemovableHandle] = None
    
    def setup_model_hooks(self, model: nn.Module) -> None:
        """
        Set up hooks for tracking model activations and features.
        
        Args:
            model: PyTorch model to instrument
        """
        # Setup neural collapse feature hook
        if self.track_nc_metrics and self.nc_analyzer is not None:
            self.nc_analyzer.register_feature_hook(model)
        
        # Setup feature hook for MI computation
        if self.track_mi_metrics:
            self._setup_feature_hook(model)
        
        # Setup activation hooks
        if self.track_activations:
            self._setup_activation_hooks(model)
    
    def _setup_feature_hook(self, model: nn.Module) -> None:
        """Setup hook to capture penultimate layer features for MI computation."""
        def feature_hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.features = input[0].clone().detach()
        
        self.feature_hook = model.fc.register_forward_hook(feature_hook)
    
    def _setup_activation_hooks(self, model: nn.Module) -> None:
        """Setup hooks to track activations from key layers."""
        # Find key layers to track
        layers_to_track = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Track first conv layer and some intermediate ones
                if any(layer_id in name for layer_id in ['0', '4', '7']):
                    layers_to_track.append((name, module))
        
        # Register hooks for selected layers (limit to avoid overhead)
        for name, module in layers_to_track[:3]:
            hook = module.register_forward_hook(
                lambda m, inp, out, name=name: self._save_activation(name, out)
            )
            self.activation_hooks.append(hook)
    
    def _save_activation(self, name: str, activation: torch.Tensor) -> None:
        """Save activations from forward pass."""
        if isinstance(activation, tuple):
            activation = activation[0]
        self.activations[name] = activation.detach()
    
    def compute_basic_metrics(
        self, 
        model: nn.Module, 
        criterion: nn.Module,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Compute basic metrics (accuracy and loss).
        
        Args:
            model: PyTorch model
            criterion: Loss function
            data_loader: Data loader
            
        Returns:
            Dictionary containing accuracy and loss
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                
                # Handle model outputs (tuple or tensor)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return {
            'accuracy': 100.0 * correct / total,
            'loss': total_loss / len(data_loader)
        }
    
    def compute_gradient_metrics(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute gradient-related metrics.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary containing gradient statistics
        """
        if not self.track_gradients:
            return {}
        
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1.0 / 2)
            return {
                'gradient_norm': total_norm,
                'gradient_params_count': param_count
            }
        else:
            return {'gradient_norm': 0.0, 'gradient_params_count': 0}
    
    def compute_activation_metrics(self) -> Dict[str, Any]:
        """
        Compute activation-related metrics.
        
        Returns:
            Dictionary containing activation statistics
        """
        if not self.track_activations or not self.activations:
            return {}
        
        activation_stats = {}
        
        for name, activation in self.activations.items():
            flat_activation = activation.flatten().cpu().numpy()
            
            activation_stats[f'activations/{name}_mean'] = float(np.mean(flat_activation))
            activation_stats[f'activations/{name}_std'] = float(np.std(flat_activation))
            activation_stats[f'activations/{name}_min'] = float(np.min(flat_activation))
            activation_stats[f'activations/{name}_max'] = float(np.max(flat_activation))
        
        return activation_stats
    
    def compute_mi_metrics(
        self, 
        nn_pairs: Any, 
        model: nn.Module,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute mutual information metrics.
        
        Args:
            nn_pairs: Nearest neighbor pairs for MI computation
            model: PyTorch model
            config: Configuration dictionary with MI parameters
            
        Returns:
            Dictionary containing MI metrics
        """
        if not self.track_mi_metrics:
            return {}
        
        try:
            # Import MI computation functions
            from src.utils.utils import get_nn_pair_predictions, get_pmf_table
            from src.mi.discrete_mi import compute_mi
            from src.mi.latent_mi import compute_latent_mi
            
            # Get predictions for nearest neighbor pairs
            predictions = get_nn_pair_predictions(nn_pairs, model, self.device)
            
            # Compute discrete mutual information
            pmf_table = get_pmf_table(predictions, len(nn_pairs))
            discrete_mi = compute_mi(pmf_table)
            
            # Compute latent mutual information
            latent_mi = compute_latent_mi(
                predictions,
                latent_dim=config.get('lmi_dim', 16),
                estimate_on_val=config.get('estimate_on_val', False)
            )
            
            return {
                'discrete_mutual_information': float(discrete_mi),
                'latent_mutual_information': float(latent_mi)
            }
            
        except ImportError as e:
            print(f"Warning: Could not compute MI metrics due to import error: {e}")
            return {}
        except Exception as e:
            print(f"Warning: Error computing MI metrics: {e}")
            return {}
    
    def compute_nc_metrics(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Compute neural collapse metrics.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            test_loader: Test data loader
            
        Returns:
            Dictionary containing NC metrics
        """
        if not self.track_nc_metrics or self.nc_analyzer is None:
            return {}
        
        try:
            return self.nc_analyzer.compute_nc_metrics(model, train_loader, test_loader)
        except Exception as e:
            print(f"Warning: Error computing NC metrics: {e}")
            return {}
    
    def compute_comprehensive_metrics(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        nn_pairs: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics in one call.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            train_loader: Training data loader
            test_loader: Test data loader
            nn_pairs: Nearest neighbor pairs for MI computation
            config: Configuration dictionary
            
        Returns:
            Dictionary containing all computed metrics
        """
        all_metrics = {}
        
        # Basic metrics on both train and test sets
        train_metrics = self.compute_basic_metrics(model, criterion, train_loader)
        test_metrics = self.compute_basic_metrics(model, criterion, test_loader)
        
        all_metrics.update({
            'train_accuracy': train_metrics['accuracy'],
            'train_loss': train_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss']
        })
        
        # Gradient metrics
        all_metrics.update(self.compute_gradient_metrics(model))
        
        # Activation metrics
        all_metrics.update(self.compute_activation_metrics())
        
        # Neural collapse metrics
        if self.track_nc_metrics:
            nc_metrics = self.compute_nc_metrics(model, train_loader, test_loader)
            all_metrics.update(nc_metrics)
        
        # Mutual information metrics
        if self.track_mi_metrics and nn_pairs is not None and config is not None:
            mi_metrics = self.compute_mi_metrics(nn_pairs, model, config)
            all_metrics.update(mi_metrics)
        
        return all_metrics
    
    def cleanup(self) -> None:
        """Clean up all registered hooks."""
        # Remove activation hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks.clear()
        
        # Remove feature hook
        if self.feature_hook is not None:
            self.feature_hook.remove()
            self.feature_hook = None
        
        # Clean up neural collapse analyzer
        if self.nc_analyzer is not None:
            self.nc_analyzer.remove_feature_hook()
    
    def __del__(self) -> None:
        """Cleanup on destruction."""
        self.cleanup()