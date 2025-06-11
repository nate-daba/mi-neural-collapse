"""
ResNet Trainer with Metrics Tracking

This module provides a complete training framework for ResNet-18 on CIFAR-10
with metrics tracking including mutual information and neural collapse phenomena.

"""

import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, Union

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchinfo import summary
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import matplotlib.figure

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import custom modules
try:
    from src.utils.utils import save_checkpoint, get_pmf_table, plot_pmf_table
    from metrics.metrics_tracker import MetricsTracker
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure all required modules are available in the Python path")


def safe_close_figure(fig: Any) -> None:
    """
    Safely close a matplotlib figure with proper type checking.
    
    Args:
        fig: Figure object to close (or other object)
    """
    try:
        if fig is None:
            return
        
        # Check if it's a matplotlib Figure
        if isinstance(fig, matplotlib.figure.Figure):
            plt.close(fig)
        elif hasattr(fig, 'savefig'):  # Duck typing for Figure-like objects
            plt.close(fig)
        elif isinstance(fig, int):  # Figure number
            plt.close(fig)
        elif isinstance(fig, str):  # Figure name
            plt.close(fig)
        else:
            # If we're not sure what it is, try to close all figures
            print(f"Warning: Unexpected figure type {type(fig)}, closing all figures")
            plt.close('all')
            
    except Exception as e:
        print(f"Warning: Error closing figure: {e}")
        # Last resort - close all figures
        try:
            plt.close('all')
        except:
            pass


def safe_plot_pmf_table(pmf_table, class_names, title="PMF Table"):
    """
    Safely plot PMF table with proper figure management.
    
    Args:
        pmf_table: PMF table as numpy array
        class_names: List of class names for labeling
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    try:
        # Try to use the original plot_pmf_table function first
        fig = plot_pmf_table(pmf_table, class_names)
        
        # Validate that we got a proper Figure object
        if isinstance(fig, matplotlib.figure.Figure) or hasattr(fig, 'savefig'):
            return fig
        else:
            print(f"Original plot_pmf_table returned unexpected type: {type(fig)}")
            raise ValueError("Invalid figure type returned")
            
    except Exception as e:
        print(f"Original plot_pmf_table failed: {e}")
        print("Falling back to safe plotting function")
        
        # Fallback: create a simple heatmap
        try:
            import seaborn as sns
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(
                pmf_table,
                annot=True,
                fmt='.3f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar_kws={'label': 'Probability'}
            )
            
            ax.set_xlabel('Test Predictions')
            ax.set_ylabel('Train Predictions')
            ax.set_title(title)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)
            fig.tight_layout()
            
            return fig
            
        except Exception as fallback_e:
            print(f"Fallback plotting also failed: {fallback_e}")
            # Create a minimal error plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"PMF Plot Error:\n{str(e)}", 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title("PMF Plot - Error Occurred")
            return fig


class ResNetWithFeatures(nn.Module):
    """
    ResNet wrapper that returns both logits and penultimate layer features.

    This is necessary for metrics computation including
    neural collapse and mutual information analysis.
    """
    
    def __init__(self, base_model: nn.Module) -> None:
        """
        Initialize the wrapper.
        
        Args:
            base_model: Base ResNet model
        """
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = base_model.fc
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and features.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (logits, features)
        """
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits, features


class ResNetTrainer:
    """
    ResNet-18 trainer for CIFAR-10 with extensive metrics tracking.

    This trainer implements:
    - Standard training with configurable hyperparameters
    - Neural collapse metrics tracking (NC1-NC4)
    - Mutual information metrics tracking
    - Logging with Weights & Biases
    - Gradient and activation monitoring
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the trainer.

        Args:
            config: Training configuration dictionary
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Initialize components
        self._init_data()
        self._init_model()
        self._init_metrics_tracker()
        
        print(f"Initialized ResNetTrainer on {self.device}")
        print(f"Tracking NC metrics: {self.config.get('track_nc_metrics', True)}")
        print(f"Tracking MI metrics: {self.config.get('track_mi_metrics', True)}")
    
    def _init_data(self) -> None:
        """Initialize CIFAR-10 data loaders with appropriate transforms."""
        # Training transforms
        train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
        
        # Add data augmentation if specified
        if self.config.get('data_augmentation', False):
            train_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ] + train_transforms
        
        train_transform = transforms.Compose(train_transforms)
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=self.config.get('num_workers', 4)
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config.get('num_workers', 4)
        )
        
        # Load nearest neighbor pairs for MI computation if available
        self.nn_pairs = None
        if self.config.get('nn_pairs_path') and os.path.exists(self.config['nn_pairs_path']):
            try:
                self.nn_pairs = torch.load(self.config['nn_pairs_path'], weights_only=False)
                print(f"Loaded NN pairs from {self.config['nn_pairs_path']}")
            except Exception as e:
                print(f"Warning: Could not load NN pairs: {e}")
        
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)
        
        print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
        print(f"Number of classes: {self.num_classes}")
    
    def _init_model(self) -> None:
        """Initialize ResNet-18 model with setup."""
        # Create base ResNet-18 for CIFAR-10
        base_model = models.resnet18(num_classes=self.num_classes, weights=None)
        
        # Modify for CIFAR-10 (remove max pooling for small images)
        base_model.maxpool = nn.Identity()
        
        # Wrap model to return features
        self.model = ResNetWithFeatures(base_model).to(self.device)
        
        # Get feature dimension
        self.feature_dim = self.model.fc.in_features
        print(f"Feature dimension: {self.feature_dim}")
        
        # Print model summary
        summary(self.model, input_size=(1, 3, 32, 32))
        
        # Initialize optimizer
        if self.config.get('optimizer', 'SGD').upper() == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 5e-4)
            )
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not supported")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        if self.config.get('epochs_lr_decay'):
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config['epochs_lr_decay'],
                gamma=self.config.get('lr_decay', 0.1)
            )
        else:
            self.scheduler = None
        
        # Initialize Weights & Biases tracking
        if self.config.get('use_wandb', True):
            wandb.watch(
                self.model,
                log="all",
                log_freq=self.config.get('wandb_log_freq', 100),
                log_graph=True
            )
    
    def _init_metrics_tracker(self) -> None:
        """Initialize the metrics tracker."""
        try:
            self.metrics_tracker = MetricsTracker(
                num_classes=self.num_classes,
                feature_dim=self.feature_dim,
                device=self.device,
                track_activations=self.config.get('track_activations', True),
                track_gradients=self.config.get('track_gradients', True),
                track_nc_metrics=self.config.get('track_nc_metrics', True),
                track_mi_metrics=self.config.get('track_mi_metrics', True),
                nc_tile_size=self.config.get('nc_tile_size', 64)
            )
            
            # Setup model hooks for metrics tracking
            self.metrics_tracker.setup_model_hooks(self.model)
            print("Metrics tracker initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize metrics tracker: {e}")
            print("Continuing without advanced metrics tracking")
            self.metrics_tracker = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing epoch training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config['epochs']}", 
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(epoch_pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, features = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if self.config.get('gradient_clip_norm') is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Log detailed metrics periodically
            if (batch_idx + 1) % self.config.get('log_interval', 100) == 0:
                current_acc = 100.0 * correct / total
                current_loss = running_loss / (batch_idx + 1)
                
                # Update progress bar
                epoch_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
                
                # Log to wandb
                if self.config.get('use_wandb', True):
                    step_metrics = {
                        'batch_loss': loss.item(),
                        'batch_accuracy': 100.0 * (predicted == labels).float().mean().item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    
                    # Add gradient metrics if tracker is available
                    if self.metrics_tracker is not None:
                        grad_metrics = self.metrics_tracker.compute_gradient_metrics(self.model)
                        step_metrics.update(grad_metrics)
                        
                        # Add activation metrics (less frequently)
                        if (batch_idx + 1) % (self.config.get('log_interval', 100) * 2) == 0:
                            activation_metrics = self.metrics_tracker.compute_activation_metrics()
                            step_metrics.update(activation_metrics)
                    
                    wandb.log(step_metrics)
        
        # Compute final epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'train_loss': epoch_loss,
            'train_accuracy': epoch_acc
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.metrics_tracker is not None:
            return self.metrics_tracker.compute_basic_metrics(
                self.model, self.criterion, self.test_loader
            )
        else:
            # Fallback basic evaluation
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs, _ = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            return {
                'accuracy': 100.0 * correct / total,
                'loss': total_loss / len(self.test_loader)
            }
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Returns:
            Dictionary containing metrics
        """
        if self.metrics_tracker is not None:
            # Use the correct method name from your MetricsTracker
            if hasattr(self.metrics_tracker, 'compute_comprehensive_metrics'):
                return self.metrics_tracker.compute_comprehensive_metrics(
                    model=self.model,
                    criterion=self.criterion,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader,
                    nn_pairs=self.nn_pairs,
                    config=self.config
                )
            elif hasattr(self.metrics_tracker, 'compute_metrics'):
                return self.metrics_tracker.compute_metrics(
                    model=self.model,
                    criterion=self.criterion,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader,
                    nn_pairs=self.nn_pairs,
                    config=self.config
                )
            else:
                print("Warning: MetricsTracker doesn't have expected methods")
                return {}
        else:
            return {}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = os.path.join("checkpoints", self.config['timestamp'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'resnet18_epoch_{epoch}_acc_{metrics.get("test_accuracy", metrics.get("accuracy", 0)):.2f}.pth'
        )
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def train(self) -> None:
        """Main training loop with metrics tracking."""
        best_test_acc = 0.0
        best_metrics = {}

        print("Starting training...")
        print("=" * 80)
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Evaluation phase
            eval_metrics = self.evaluate()
            
            # Combine basic metrics
            current_metrics = {**train_metrics, **eval_metrics}
            current_metrics['epoch'] = epoch + 1
            current_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # Compute comprehensive metrics periodically
            if (epoch + 1) % self.config.get('eval_freq', 5) == 0 or epoch == self.config['epochs'] - 1:
                print("Computing comprehensive metrics...")
                comprehensive_metrics = self.compute_metrics()
                current_metrics.update(comprehensive_metrics)

                # Plot and log PMF table if MI metrics are available
                if 'discrete_mutual_information' in comprehensive_metrics and self.nn_pairs is not None:
                    try:
                        from src.utils.utils import get_nn_pair_predictions, get_pmf_table
                        predictions = get_nn_pair_predictions(self.nn_pairs, self.model, self.device)
                        pmf_table = get_pmf_table(predictions, len(self.nn_pairs))
                        
                        # Use safe plotting function
                        fig = safe_plot_pmf_table(pmf_table, self.class_names)
                        
                        if self.config.get('use_wandb', True) and fig is not None:
                            wandb.log({"pmf_table": wandb.Image(fig)})
                        
                        # Safe figure closing
                        safe_close_figure(fig)
                        
                    except Exception as e:
                        print(f"Warning: Could not plot PMF table: {e}")
                        # Ensure we close any potentially open figures
                        plt.close('all')
            
            # Logging
            print(f"\nEpoch [{epoch+1}/{self.config['epochs']}]")
            print(f"Train Loss: {current_metrics['train_loss']:.4f}, Train Acc: {current_metrics['train_accuracy']:.2f}%")
            
            # Handle different metric key names
            test_acc_key = 'accuracy' if 'accuracy' in current_metrics else 'test_accuracy'
            test_loss_key = 'loss' if 'loss' in current_metrics else 'test_loss'
            
            print(f"Test Loss: {current_metrics[test_loss_key]:.4f}, Test Acc: {current_metrics[test_acc_key]:.2f}%")
            
            if 'discrete_mutual_information' in current_metrics:
                print(f"Discrete MI: {current_metrics['discrete_mutual_information']:.4f} bits")
            if 'nc1_pinv' in current_metrics:
                print(f"NC1 (pinv): {current_metrics['nc1_pinv']:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', True):
                wandb.log(current_metrics)
            
            # Save best model
            current_test_acc = current_metrics[test_acc_key]
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                best_metrics = current_metrics.copy()
                
                checkpoint_path = self.save_checkpoint(epoch + 1, current_metrics)
                print(f"New best model saved: {checkpoint_path}")
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        "best_test_accuracy": best_test_acc,
                        "checkpoint_path": checkpoint_path
                    })
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
        
        if best_metrics:
            print("Best model metrics:")
            for key, value in best_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'metrics_tracker') and self.metrics_tracker is not None:
            self.metrics_tracker.cleanup()
        # Close any remaining matplotlib figures
        plt.close('all')
    
    def __del__(self) -> None:
        """Cleanup on destruction."""
        self.cleanup()


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for training.
    
    Returns:
        Default configuration dictionary
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    return {
        # Training hyperparameters (based on neural collapse paper: https://arxiv.org/abs/2008.08186)
        "epochs": 350,
        "batch_size": 256,
        "learning_rate": 0.1,
        "epochs_lr_decay": [150, 250],
        "lr_decay": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "optimizer": "SGD",
        
        # Model and dataset
        "model": "ResNet-18",
        "dataset": "CIFAR-10",
        "data_augmentation": False,  # Set to True to enable
        
        # Metrics tracking configuration
        "track_nc_metrics": True,
        "track_mi_metrics": True,
        "track_activations": True,
        "track_gradients": True,
        "eval_freq": 1,  # Compute comprehensive metrics every N epochs

        # MI-specific parameters
        "lmi_dim": 64,
        "estimate_on_val": False,
        "nn_pairs_path": "data/nn_pairs/class_constrained_nn_pairs.pt",
        
        # NC-specific parameters
        "nc_tile_size": 64,
        
        # Training configuration
        "gradient_clip_norm": 1.0,
        "num_workers": 4,
        "log_interval": 100,
        
        # Logging
        "use_wandb": True,
        "wandb_log_freq": 100,
        "timestamp": timestamp,
        
        # Paths
        "checkpoint_dir": f"checkpoints/{timestamp}",
    }


def run_training() -> None:
    """
    Run the training with comprehensive metrics tracking.
    """
    # Create configuration
    config = create_default_config()
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project="resnet18-cifar10",
            name=f"resnet18-{config['timestamp']}",
            config=config,
            tags=["neural-collapse", "mutual-information"]
        )
        config = dict(wandb.config)  # Use wandb config for hyperparameter sweeps
    
    # Create and run trainer
    trainer = ResNetTrainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        trainer.cleanup()
        if config['use_wandb']:
            wandb.finish()


if __name__ == "__main__":
    run_training()