import os
import sys
from datetime import datetime
from typing import Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchinfo import summary
from tqdm import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.utils import save_checkpoint, get_pmf_table, plot_pmf_table, get_nn_pair_predictions
from src.mi.discrete_mi import compute_mi
from src.mi.latent_mi import compute_latent_mi

class ResNetTrainer:
    """
    Class for training ResNet-18 on CIFAR-10 with mutual information tracking.
    Based on neural collapse paper hyperparameters and methodology.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the trainer with the given configuration.

        Args:
            config (dict): A dictionary containing training configurations.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self._init_data()
        self._init_model()

    def _init_data(self) -> None:
        """
        Initializes the training and validation data loaders for CIFAR-10.
        Uses data augmentation as specified in neural collapse paper.
        """
        # Training transforms: Random crop + horizontal flip + normalization
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Test transforms: Only normalization
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Loading CIFAR-10 training data
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

        # Loading CIFAR-10 test data
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        
        # Loading the nn pairs for the test-train pairs
        nn_pairs = torch.load(self.config.nn_pairs_path, weights_only=False)
        self.nn_pairs = nn_pairs
        
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)
        print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
            
    def _init_model(self) -> None:
        """
        Initializes ResNet-18 model, optimizer, and loss function.
        Uses neural collapse paper hyperparameters.
        """
        # Initialize ResNet-18 for CIFAR-10
        base_model = models.resnet18(num_classes=10, weights=None)
        
        # Modify for CIFAR-10 (32x32 RGB images)
        # Remove the initial max pooling for small images
        base_model.maxpool = nn.Identity()
        
        # Create a wrapper model that returns both logits and features
        class ResNetWithFeatures(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.features = nn.Sequential(*list(base_model.children())[:-1])
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = base_model.fc
                
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                features = torch.flatten(x, 1)
                logits = self.fc(features)
                return logits, features
        
        self.model = ResNetWithFeatures(base_model).to(self.device)
        
        # Get feature dimension for hooks
        self.feature_dim = self.model.fc.in_features
        print(f"Feature dimension: {self.feature_dim}")
        
        # Print model summary
        summary(self.model, input_size=(1, 3, 32, 32))

        # Initialize optimizer (SGD as in neural collapse paper)
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler (MultiStepLR as in neural collapse paper)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config.epochs_lr_decay,
            gamma=self.config.lr_decay
        )
        
        # Use wandb.watch to track parameters and gradients
        wandb.watch(
            self.model,
            log="all",     # Track both gradients and parameters
            log_freq=100,  # Adjust this based on your needs
            log_graph=True # Visualize model architecture
        )
        
        # Set up activation tracking for key layers
        self.activation_hooks = []
        self.activations = {}
        self.features = None  # To store penultimate layer features
        
        # Hook to capture penultimate layer features (input to final classifier)
        def feature_hook(module, input, output):
            self.features = input[0].clone().detach()
            
        self.model.fc.register_forward_hook(feature_hook)
        
        # Track a few strategically important layers
        layers_to_track = []
        
        # Find key layers to track in the feature extractor
        for name, module in self.model.features.named_modules():
            if isinstance(module, nn.Conv2d):
                # Track first conv layer and some intermediate ones
                if '0' in name or '4' in name or '7' in name:  # Adjust based on ResNet structure
                    layers_to_track.append((name, module))
        
        # Register hooks for these layers
        for name, module in layers_to_track[:3]:  # Limit to 3 layers to avoid overhead
            self.activation_hooks.append(
                module.register_forward_hook(
                    lambda m, inp, out, name=name: self._save_activation(name, out)
                )
            )
    
    def _save_activation(self, name, activation):
        """Save activations from forward pass for logging later"""
        if isinstance(activation, tuple):
            activation = activation[0]
        self.activations[name] = activation.detach()

    def compute_accuracy(self, data_loader):
        """Compute accuracy on given data loader"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _ = self.model(images)  # Unpack the tuple
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def train(self) -> None:
        """
        Main method for training the model.
        """
        best_test_acc = 0.0
        best_val_loss = float('inf')

        for epoch in tqdm(range(self.config.epochs), desc="Training epoch"):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs, features = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Log gradient norm before clipping (optional)
                if i % 100 == 0:
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    wandb.log({"gradient_norm_before_clip": total_norm})
                    
                # Gradient clipping
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Log penultimate layer activations periodically
                if i % 100 == 0:
                    wandb.log({"penultimate_activations": wandb.Histogram(features.detach().cpu().numpy())})
                
                # Log other tracked activations less frequently
                if i % 200 == 0:
                    for name, activation in self.activations.items():
                        flat_activation = activation.flatten().cpu().numpy()
                        wandb.log({f"activations/{name}": wandb.Histogram(flat_activation)})
                        
                        # For convolutional layers, visualize feature maps
                        if len(activation.shape) == 4:  # [batch, channels, height, width]
                            num_channels = min(6, activation.shape[1])
                            feature_maps = activation[0, :num_channels].cpu().numpy()
                            
                            images_to_log = []
                            for idx in range(num_channels):
                                fmap = feature_maps[idx]
                                if fmap.max() > fmap.min():
                                    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
                                images_to_log.append(wandb.Image(fmap))
                            
                            wandb.log({f"feature_maps/{name}": images_to_log})

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.config.epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")

            # Update learning rate
            self.scheduler.step()
            
            # Compute metrics
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct / total
            test_acc = self.compute_accuracy(self.test_loader)
            
            print(f"Epoch [{epoch+1}/{self.config.epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

            # Evaluate mutual information (compute every few epochs to save time)
            if (epoch + 1) % self.config.mi_eval_freq == 0 or epoch == self.config.epochs - 1:
                test_loss, mi, latent_mi, pmf_table = self.evaluate_mi()
                
                # Log PMF table
                fig = plot_pmf_table(pmf_table, self.class_names)
                wandb.log({"pmf_table": wandb.Image(fig)})
                
                # Log to wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "mutual_information": mi,
                    "latent_mutual_information": latent_mi,
                })
                
                # Save checkpoint if test accuracy improves
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_val_loss = test_loss
                    ckpt_path = save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch + 1,
                        val_loss=test_loss,
                        train_loss=train_loss,
                        config=self.config,
                        base_dir=os.path.join("checkpoints", self.config.timestamp)
                    )
                    wandb.log({"checkpoint_saved_at": ckpt_path, "best_test_accuracy": best_test_acc})
            else:
                # Log basic metrics without MI computation
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

        print(f"Training completed! Best Test Accuracy: {best_test_acc:.2f}%")
        
        # Clean up hooks to prevent memory leaks
        for hook in self.activation_hooks:
            hook.remove()
    
    def evaluate_mi(self) -> Tuple[float, float, float, any]:
        """
        Evaluates mutual information between train-test predictions.

        Returns:
            tuple: (test_loss, mi, latent_mi, pmf_table)
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _ = self.model(images)  # Unpack the tuple
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_loss = test_loss / len(self.test_loader)
        test_acc = 100 * correct / total
        
        # Get predictions for the nearest neighbor pairs
        predictions = get_nn_pair_predictions(self.nn_pairs, self.model, self.device)
        
        # Compute discrete mutual information between classification vectors
        pmf_table = get_pmf_table(predictions, len(self.nn_pairs))
        mi = compute_mi(pmf_table)
        
        # Compute mutual information between continuous latent features
        latent_mi = compute_latent_mi(
            predictions,
            latent_dim=self.config.lmi_dim,
            estimate_on_val=self.config.estimate_on_val
        )

        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, MI: {mi:.4f} bits, Latent MI: {latent_mi:.4f} bits")
        return test_loss, mi, latent_mi, pmf_table

def run_training() -> None:
    """
    Initializes and runs the training process.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Configuration based on neural collapse paper hyperparameters
    config = {
        "epochs": 350,  # Neural collapse paper uses 350 epochs for CIFAR-10
        "batch_size": 256,  # Neural collapse paper batch size
        "learning_rate": 0.1,  # Initial learning rate
        "epochs_lr_decay": [150, 250],  # Learning rate decay epochs
        "lr_decay": 0.1,  # Decay factor
        "momentum": 0.9,  # SGD momentum
        "weight_decay": 5e-4,  # Weight decay
        "lmi_dim": 64,  # Latent MI dimension
        "optimizer": "SGD",
        "model": "ResNet-18",
        "dataset": "CIFAR-10",
        "data_augmentation": True,
        "timestamp": timestamp,
        "nn_pairs_path": "data/nn_pairs/class_constrained_nn_pairs.pt",
        "estimate_on_val": False,  # LMI estimation setting
        "mi_eval_freq": 1,  # Evaluate MI every 5 epochs to save computation
        "gradient_clip_norm": 1.0,  # Gradient clipping norm
    }
    
    wandb.init(
        project="resnet18-cifar10-mi", 
        name=f"resnet18-neural-collapse-{timestamp}", 
        config=config
    )
    
    trainer = ResNetTrainer(config=wandb.config)
    trainer.train()
    
if __name__ == "__main__":
    run_training()
    wandb.finish()