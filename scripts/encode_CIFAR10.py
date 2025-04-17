import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.models.ae import AE


def get_dataloaders(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def encode_dataset(model, loader, device):
    model.eval()
    all_encodings = []

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Encoding dataset"):
            x = x.to(device)
            z = model.encode(x)  # B x latent_dim
            all_encodings.append(z.cpu().numpy())

    return np.concatenate(all_encodings, axis=0)


def evaluate_mse(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Evaluating reconstruction"):
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)

    return total_loss / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/encoded_CIFAR10/train_enc.npy")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE(latent_dim=args.latent_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Optional: Validate MSE on test set
    val_mse = evaluate_mse(model, test_loader, device)
    print(f"Validation MSE: {val_mse:.6f}")
    print(f"Saved MSE: {checkpoint['val_loss']:.6f}")

    # Encode training set
    X_enc = encode_dataset(model, train_loader, device)
    np.save(args.output, X_enc)
    print(f"Saved encoded features to {args.output} with shape {X_enc.shape}")

if __name__ == "__main__":
    main()
