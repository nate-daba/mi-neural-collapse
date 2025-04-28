import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from src.models.ae import AE  # Your trained AE model
from src.utils.utils import save_checkpoint
from typing import Tuple, Union

import torchvision.utils as vutils

class NearestNeighborMatcher:
    """
    Class for performing nearest neighbor search in the latent space.
    """
    
    def __init__(self, model: AE, device: torch.device, batch_size: int = 512) -> None:
        """
        Initializes the NearestNeighborMatcher.
        
        Args:
            model (AE): The trained model used to generate latent embeddings.
            device (torch.device): Device for running the model (CPU/GPU).
            batch_size (int): Batch size for data loading.
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.train_embeddings = None
        self.nearest_neighbors = None

    def encode_data(self, data_loader: DataLoader) -> np.ndarray:
        """
        Encodes the dataset using the encoder of the model.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to be encoded.

        Returns:
            np.ndarray: Encoded feature matrix.
        """
        self.model.eval()
        all_encodings = []

        with torch.no_grad():
            for images, _ in tqdm(data_loader, desc="Encoding dataset"):
                images = images.to(self.device)
                
                # Unsqueeze to add batch dimension if needed (for single image inference)
                if images.dim() == 3:  # If the image is of shape (C, H, W)
                    images = images.unsqueeze(0)  # Add batch dimension (1, C, H, W)

                encoded = self.model.encode(images)  # Use the encoder part
                
                # # Flatten the encoded tensor if needed
                # encoded = torch.flatten(encoded, start_dim=1)  # Flatten starting from the second dimension

                all_encodings.append(encoded.cpu().numpy())

        return np.concatenate(all_encodings, axis=0)

    def fit(self, train_loader: DataLoader) -> None:
        """
        Fit the nearest neighbor model using the training data embeddings.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
        """
        # Encode the training data
        print("Encoding training data...")
        train_embeddings = self.encode_data(train_loader)
        print("Training embeddings shape:", train_embeddings.shape)
        # Store the training embeddings
        self.train_embeddings = train_embeddings

        # Fit the NearestNeighbors model on the training embeddings
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        self.nearest_neighbors.fit(train_embeddings)

    def find_nearest_neighbors(self, test_loader: DataLoader) -> np.ndarray:
        """
        Find the nearest neighbors for each test sample in the training set.

        Args:
            test_loader (DataLoader): DataLoader for the test set.

        Returns:
            np.ndarray: Indices of the nearest neighbors for each test sample.
        """
        print("Finding nearest neighbors for test data...")
        test_embeddings = self.encode_data(test_loader)
        print("Test embeddings shape:", test_embeddings.shape)
        # Find the nearest neighbors
        distances, indices = self.nearest_neighbors.kneighbors(test_embeddings)
        return indices

if __name__ == "__main__":
    # Ensure the model is loaded into the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the AE model (adjust with your model checkpoint path)
    model = AE(latent_dim=200).to(device)
    ckpt_path = "checkpoints/2025-04-14-13-08-30/2025-04-14-13-30-44-val-loss-0.01/checkpoint_val_loss_0.01.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # DataLoader for CIFAR-10
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Single sample per batch for visualization

    # Initialize the nearest neighbor matcher
    nn_matcher = NearestNeighborMatcher(model, device)

    # Load training data for nearest neighbor search
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)  # Use batch_size for efficiency
    nn_matcher.fit(train_loader)  # Fit nearest neighbor matcher on the training data

    # Now we will visualize the nearest neighbors for each class
    # We will randomly sample one image per class from the test set and its corresponding nearest neighbor

    class_names = test_dataset.classes
    fig, axes = plt.subplots(2, 10, figsize=(15, 8))

    # Randomly select one image per class
    selected_indices = []
    for class_idx in range(10):
        class_images = np.where(np.array(test_dataset.targets) == class_idx)[0]
        selected_idx = np.random.choice(class_images)  # Randomly choose one image from the class
        selected_indices.append(selected_idx)

    test_images = [test_dataset[i][0] for i in selected_indices]  # Selected test images
    test_labels = [test_dataset[i][1] for i in selected_indices]  # Corresponding labels

    # Find nearest neighbors for these selected test images
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_image_loader = torch.utils.data.Subset(test_loader.dataset, selected_indices)

    # Get the nearest neighbors for these test images
    nearest_indices = nn_matcher.find_nearest_neighbors(test_image_loader)

    # Directory to save the results
    output_dir = "results/nn_plots/"
    os.makedirs(output_dir, exist_ok=True)

    # Combine the test images and their nearest neighbors for the plot
    test_images_tensor = torch.stack(test_images)
    nearest_images_tensor = torch.stack([train_dataset[nearest_indices[i][0]][0] for i in range(len(selected_indices))])

    # Create a grid of test images (top row) and nearest images (bottom row)
    images_grid = torch.cat((test_images_tensor, nearest_images_tensor), dim=0)
    grid = vutils.make_grid(images_grid, nrow=10, padding=2, normalize=True)  # Added padding between images

    # Plot the image grid
    fig, ax = plt.subplots(figsize=(15, 8))

    # Show the grid of images
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())  # Convert to numpy for imshow
    ax.axis('off')  # Hide axis

    img_width = 32  # Width of CIFAR-10 image
    padding = 2  # Match the padding in make_grid
    full_width = img_width + padding  # Full width including padding

    # Add class labels for each test image (top row)
    for i, label in enumerate(test_labels):
        ax.text(i * full_width + img_width // 2, -5, 
                f'{class_names[label]}', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='none', pad=1))

    # Add class labels for nearest neighbor images (bottom row)
    for i, label in enumerate(test_labels):
        nn_label = train_dataset[nearest_indices[i][0]][1]  # Get actual label of nearest neighbor
        ax.text(i * full_width + img_width // 2, img_width + padding + img_width + 5, 
                f'{class_names[nn_label]}', 
                ha='center', va='top', fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='none', pad=1))

    # Add row titles for "Test Image" and "NN Training Image"
    ax.text(-10, img_width // 2, 'Test Image', fontsize=12, va='center', ha='right', rotation=90)
    ax.text(-10, img_width + padding + img_width // 2, 'NN Training Image', fontsize=12, va='center', ha='right', rotation=90)

    # Add the title for the entire plot
    plt.suptitle("Nearest Neighbors: Test Images and Their Corresponding Nearest Neighbors", fontsize=14)

    # Save the figure to a file
    fig_path = os.path.join(output_dir, "nearest_neighbors.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"Saved nearest neighbors plot to {fig_path}")