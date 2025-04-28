import sys
import os
from typing import Tuple

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.models.ae import AE
from src.models.aec import AEC
from scripts.whiten import whiten


def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def encode_dataset(model: AE, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_encodings = []

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Encoding dataset"):
            x = x.to(device)
            z = model.encode(x)  # B x latent_dim
            all_encodings.append(z.cpu().numpy())

    return np.concatenate(all_encodings, axis=0)

def encode_labels(model: AEC, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Encodes the labels using the encoder part of the model.

    Args:
        model: The trained autoencoder model.
        loader: DataLoader for the dataset.
        device: The device (CPU or GPU) on which the model and data will be processed.

    Returns:
        np.ndarray: Encoded label feature matrix of shape (n_samples, latent_dim).
    """
    model.eval()
    all_labels_encodings = []

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Encoding labels"):
            x = x.to(device)
            z = model.encode(x)  
            all_labels_encodings.append(z.cpu().numpy())

    return np.concatenate(all_labels_encodings, axis=0)

def evaluate_mse(model: AE, loader: DataLoader, device: int) -> float:
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

def evaluate_ce(model: AEC, loader: DataLoader, device: int) -> float:
    """
    Evaluates the classifier model on the test set using cross-entropy loss.

    Args:
        model: The trained classifier model (AEC).
        loader: DataLoader for the test dataset.
        device: The device (CPU or GPU) on which the model and data will be processed.

    Returns:
        float: The average cross-entropy loss on the test set.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating classification"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    ce_loss = total_loss / total
    accuracy = correct / total
    
    return ce_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="Path to the AE model checkpoint (.ckpt)")
    parser.add_argument("--aec_checkpoint", type=str, required=True, help="Path to the AEC model checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/encoded_CIFAR10/train_data_enc.npy")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(args.batch_size)
    
    # 1. AE model
    # Load model
    ae_model = AE(latent_dim=args.latent_dim).to(device)
    ae_checkpoint = torch.load(args.ae_checkpoint, map_location=device)
    ae_model.load_state_dict(ae_checkpoint["model_state_dict"])

    # Validate MSE on test set
    val_mse = evaluate_mse(ae_model, test_loader, device)
    print(f"Test MSE [Re-computed]: {val_mse:.6f}")
    print(f"Test MSE [Checkpoint]: {ae_checkpoint['val_loss']:.6f}")

    # Encode training set
    train_data_enc = encode_dataset(ae_model, train_loader, device)
    np.save(args.output, train_data_enc)
    print(f"Saved encoded features to {args.output} with shape {train_data_enc.shape}")
    
    # Whitening the encoded features
    train_data_enc_white, W, train_data_enc_mean = whiten(train_data_enc, method="zca", fudge=1e-8)
    white_output_path = args.output.replace("train_data_enc.npy", "train_data_enc_white.npy")
    np.save(white_output_path, train_data_enc_white)
    print(f"Saved whitened encoded features to {white_output_path} with shape {train_data_enc_white.shape}")
    
    # 2. AEC model
    # Load model 
    aec_model = AEC(latent_dim=args.latent_dim, 
                    pdrop_2d=0.3, 
                    pdrop_1d=0.5).to(device)
    aec_checkpoint = torch.load(args.aec_checkpoint, map_location=device)
    
    aec_model.load_state_dict(aec_checkpoint["model_state_dict"])
    # Encode labels using the AEC model (classifier)
    train_labels_enc = encode_labels(aec_model, train_loader, device)

    labels_output_path = args.output.replace("train_data_enc.npy", "train_labels_enc.npy")
    np.save(labels_output_path, train_labels_enc)
    print(f"Saved encoded labels to {labels_output_path} with shape {train_labels_enc.shape}")

    # Center the label encodings baesd on mean of training data encodings
    train_labels_enc = train_labels_enc - train_data_enc_mean
    # Whitening the encoded labels using the same whitening matrix
    train_labels_enc_white = train_labels_enc @ W.T
    labels_white_output_path = labels_output_path.replace("train_labels_enc.npy", "train_labels_enc_white.npy")
    np.save(labels_white_output_path, train_labels_enc_white)
    print(f"Saved whitened encoded labels to {labels_white_output_path} with shape {train_labels_enc_white.shape}")

    # Evaluate classifier performance
    ce_loss, accuracy = evaluate_ce(aec_model, test_loader, device)
    print(f"Test Classifier CE Loss [Re-computed]: {ce_loss:.6f}, Accuracy: {accuracy:.6f}")
    val_acc = aec_checkpoint['val_acc'] if 'val_acc' in aec_checkpoint else "N/A"
    if isinstance(val_acc, int):
        print(f"Test Classifier CE Loss [Checkpoint]: {aec_checkpoint['val_loss']:.6f}, Accuracy: {val_acc:.6f}")
    else:
        print(f"Test Classifier CE Loss [Checkpoint]: {aec_checkpoint['val_loss']:.6f}, Accuracy: {val_acc}")    
    
    

if __name__ == "__main__":
    main()
    
# run this:
# python3 scripts/encode_CIFAR10.py --ae_checkpoint checkpoints/2025-04-14-13-08-30/2025-04-14-13-30-44-val-loss-0.01/checkpoint_val_loss_0.01.ckpt --aec_checkpoint checkpoints/2025-04-22-12-02-45/2025-04-22-13-04-16-val-loss-0.738567/checkpoint_val_loss_0.738567.ckpt
