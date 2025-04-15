import os
import sys
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchinfo import summary

import numpy as np
from tqdm import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.ae import AE

def save_checkpoint(model, 
                    optimizer, 
                    scheduler, 
                    epoch, 
                    val_loss, 
                    train_loss,
                    config, 
                    base_dir="checkpoints"):
    """
    Saves a model checkpoint including weights, optimizer state, scheduler state, etc.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_dir = os.path.join(base_dir, f"{timestamp}-val-loss-{val_loss:.6f}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = f"checkpoint_val_loss_{val_loss:.6f}.ckpt"
    filepath = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'config': dict(config)
    }

    torch.save(checkpoint, filepath)
    return filepath

class AETrainer:
    def __init__(self, config: Any, timestamp: str):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.timestamp = timestamp
        self._init_data()
        self._init_model()

    def _init_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Full CIFAR-10 training set (50,000)
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

        # Use test set (10,000) for validation/visualization
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

        self.train_loader = DataLoader(train_dataset,
                                    batch_size=self.config.batch_size,
                                    shuffle=True)
        self.val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.batch_size,
                                    shuffle=False)

    def _init_model(self):
        self.model = AE(latent_dim=self.config.latent_dim).to(self.device)
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
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.config.learning_rate,
                                    weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

    def train(self):
        best_val_loss = float('inf')

        for epoch in tqdm(range(self.config.epochs), desc="Training epoch"):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"]
            })

            if (epoch + 1) % self.config.log_images_every == 0:
                self._log_images()

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
                    base_dir=os.path.join("checkpoints", self.timestamp)
                )
                wandb.log({"checkpoint_saved_at": ckpt_path})

            self.scheduler.step(val_loss)

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = 0
        with torch.set_grad_enabled(train):
            for x, _ in loader:
                x = x.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                x_hat = self.model(x)
                loss = self.criterion(x_hat, x)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)

    def _log_images(self):
        self.model.eval()
        with torch.no_grad():
            val_batch = next(iter(self.val_loader))[0][:3].to(self.device)
            recon = self.model(val_batch)
            images = [wandb.Image(img, caption="Original") for img in val_batch] + \
                     [wandb.Image(img, caption="Reconstructed") for img in recon]
            wandb.log({"val_reconstructions": images})

def run_training():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="ae-cifar10", name=f"ae-xavier-init-{timestamp}", config={
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "latent_dim": 200,
        "optimizer": "Adam",
        "log_images_every": 5
    })
    trainer = AETrainer(config=wandb.config, timestamp=timestamp)
    trainer.train()

if __name__ == "__main__":
    run_training()
