import os
import sys
from datetime import datetime
from typing import Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
from tqdm import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.aec import AEC  # Update this import based on where you saved the AEC class
from src.utils.utils import save_checkpoint

class AECTrainer:
    """
    Class for training a classifier on top of the encoder of the autoencoder.
    The encoder is pre-trained, and the classifier head is fine-tuned for CIFAR-10 classification.
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
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Loading CIFAR-10 training data
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        # Loading CIFAR-10 validation data
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
    
    def load_pretrained_encoder(self):
        checkpoint = torch.load(self.config.pretrained_model_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint["model_state_dict"])
        for param in self.model.encoder.parameters():
            param.requires_grad = False  # Freeze encoder layers
            
    def _init_model(self) -> None:
        """
        Initializes the model, optimizer, and loss function.
        The encoder part of the autoencoder is loaded and frozen.
        """
        self.model = AEC(latent_dim=self.config.latent_dim,
                         pdrop_2d=self.config.pdrop_2d,
                         pdrop_1d=self.config.pdrop_1d).to(self.device)
        summary(self.model, input_size=(1, 3, 32, 32))
        def init_weights(module):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

        self.model.apply(init_weights)
        # self.load_pretrained_encoder()

        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

    def train(self) -> None:
        """
        Main method for training the model.
        """
        best_val_loss = float('inf')

        for epoch in tqdm(range(self.config.epochs), desc="Training epoch"):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss /= len(self.train_loader)
            train_acc = correct / total
            print(f"Epoch [{epoch + 1}/{self.config.epochs}], Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Compute validation loss
            val_loss, val_acc = self.evaluate()

            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"]
            })

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    val_loss=val_loss,
                    train_loss=train_loss,
                    config=self.config,
                    base_dir=os.path.join("checkpoints", self.config.timestamp)
                )
                wandb.log({"checkpoint_saved_at": ckpt_path})

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

        print(f"Best Validation Loss: {best_val_loss:.4f}")

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluates the model on the validation set.

        Returns:
            float: The validation loss.
        """
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        return val_loss, val_acc

def run_training() -> None:
    """
    Initializes and runs the training process.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="aec-cifar10", name=f"aec-2d-dropout-p=0.1-{timestamp}", config={
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "latent_dim": 200,
        "pretrained_model_path": "path/to/your/pretrained/model",  # specify the path
        "optimizer": "Adam",
        "log_images_every": 5,
        "timestamp": timestamp,
        "weight_decay": 1e-4,
        "pdrop_2d": 0.3,
        "pdrop_1d": 0.5
    })
    trainer = AECTrainer(config=wandb.config)
    trainer.train()

if __name__ == "__main__":
    run_training()