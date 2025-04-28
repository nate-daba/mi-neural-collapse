import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import clip  # pip install clip
import torchvision.utils as vutils


class ClassConstrainedMatcher:
    """
    Class for performing nearest neighbor search using CLIP embeddings,
    constrained to only match within the same class.
    """
    
    def __init__(self, device: torch.device, batch_size: int = 512) -> None:
        """
        Initializes the ClassConstrainedMatcher.
        
        Args:
            device (torch.device): Device for running the model (CPU/GPU).
            batch_size (int): Batch size for data loading.
        """
        # Load the CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        self.batch_size = batch_size
        self.train_embeddings = None
        self.train_indices_by_class = None
        self.nearest_neighbors_by_class = {}

    def encode_data(self, data_loader: DataLoader) -> np.ndarray:
        """
        Encodes the dataset using CLIP's image encoder.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to be encoded.

        Returns:
            np.ndarray: Encoded feature matrix.
        """
        self.model.eval()
        all_encodings = []
        resize_transform = transforms.Resize(224)  # CLIP expects 224x224 images

        with torch.no_grad():
            for images, _ in tqdm(data_loader, desc="Encoding dataset"):
                images = images.to(self.device)
                
                # CLIP expects normalized images in the range [0, 1]
                # Convert from [-1, 1] to [0, 1] range if necessary
                if images.min() < 0:
                    images = (images + 1) / 2
                
                # Resize images to 224x224 for CLIP
                images = resize_transform(images)
                
                # Get image features from CLIP
                image_features = self.model.encode_image(images)
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
                all_encodings.append(image_features.cpu().numpy())

        return np.concatenate(all_encodings, axis=0)

    def fit(self, train_dataset) -> None:
        """
        Fit separate nearest neighbor models for each class.

        Args:
            train_dataset: The training dataset.
        """
        print("Organizing training data by class...")
        # Group training data indices by class
        self.train_indices_by_class = {}
        for i, (_, label) in enumerate(train_dataset):
            if label not in self.train_indices_by_class:
                self.train_indices_by_class[label] = []
            self.train_indices_by_class[label].append(i)
        
        print(f"Found {len(self.train_indices_by_class)} classes in training data")
        
        # Encode training data for each class separately
        for class_idx, indices in self.train_indices_by_class.items():
            print(f"Processing class {class_idx} ({len(indices)} images)...")
            
            # Create a subset of the training data for this class
            class_subset = Subset(train_dataset, indices)
            class_loader = DataLoader(class_subset, batch_size=self.batch_size, shuffle=False)
            
            # Encode the class-specific data
            class_embeddings = self.encode_data(class_loader)
            print(f"Class {class_idx} embeddings shape: {class_embeddings.shape}")
            
            # Fit a NearestNeighbors model for this class
            nn_model = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine')
            nn_model.fit(class_embeddings)
            
            # Store the model and data for this class
            self.nearest_neighbors_by_class[class_idx] = {
                'model': nn_model,
                'embeddings': class_embeddings,
                'indices': indices
            }
        
        print("Finished fitting nearest neighbor models for all classes")

    def find_nearest_neighbors(self, test_dataset, selected_indices):
        """
        Find the nearest neighbors for test samples, constrained to the same class.

        Args:
            test_dataset: The test dataset.
            selected_indices: Indices of selected test samples.

        Returns:
            tuple: Indices of nearest neighbors and distances.
        """
        nearest_indices = []
        distances = []
        
        for test_idx in selected_indices:
            test_image, test_label = test_dataset[test_idx]
            
            # If we don't have a model for this class, skip it
            if test_label not in self.nearest_neighbors_by_class:
                print(f"Warning: No model for class {test_label}")
                nearest_indices.append([0])  # Placeholder
                distances.append([1.0])  # Maximum distance
                continue
            
            # Encode the test image
            test_image = test_image.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Convert from [-1, 1] to [0, 1] range if necessary
            if test_image.min() < 0:
                test_image = (test_image + 1) / 2
            
            # Resize to 224x224 for CLIP
            resize_transform = transforms.Resize(224)
            test_image = resize_transform(test_image)
            
            # Encode the test image
            with torch.no_grad():
                test_embedding = self.model.encode_image(test_image)
                test_embedding = test_embedding / test_embedding.norm(dim=1, keepdim=True)
                test_embedding = test_embedding.cpu().numpy()
            
            # Find the nearest neighbor in the same class
            class_data = self.nearest_neighbors_by_class[test_label]
            dist, idx = class_data['model'].kneighbors(test_embedding)
            
            # Convert the local index to the global index
            global_idx = [class_data['indices'][i] for i in idx[0]]
            
            nearest_indices.append(global_idx)
            distances.append(dist[0])
        
        return nearest_indices, distances


def visualize_nearest_neighbors(test_dataset, train_dataset, selected_indices, nearest_indices, distances, output_dir):
    """
    Visualize the test images and their nearest neighbors.
    
    Args:
        test_dataset: The test dataset
        train_dataset: The training dataset
        selected_indices: Indices of selected test images
        nearest_indices: Indices of nearest neighbors for each test image
        distances: Distances to nearest neighbors
        output_dir: Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names
    class_names = test_dataset.classes
    
    # Get test images and labels
    test_images = [test_dataset[i][0] for i in selected_indices]
    test_labels = [test_dataset[i][1] for i in selected_indices]
    
    # Get nearest neighbor images and labels
    nn_images = [train_dataset[nearest_indices[i][0]][0] for i in range(len(selected_indices))]
    nn_labels = [train_dataset[nearest_indices[i][0]][1] for i in range(len(selected_indices))]
    
    # Combine test and nearest neighbor images
    test_images_tensor = torch.stack(test_images)
    nearest_images_tensor = torch.stack(nn_images)
    
    # Create a grid
    padding = 1    # Padding in grid
    images_grid = torch.cat((test_images_tensor, nearest_images_tensor), dim=0)
    grid = vutils.make_grid(images_grid, nrow=len(selected_indices), padding=padding, normalize=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis('off')
    
    # Calculate image dimensions with padding
    img_width = 32  # Width of CIFAR-10 image
    full_width = img_width + padding
    
    # Add class labels for test images (top row)
    for i, label in enumerate(test_labels):
        ax.text(i * full_width + img_width // 2, -2, 
                f'{class_names[label]}', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='none', pad=0))
    
    # Add class labels for nearest neighbor images (bottom row) and similarity score
    for i, label in enumerate(nn_labels):
        similarity = 1 - distances[i][0]  # Convert cosine distance to similarity
        ax.text(i * full_width + img_width // 2, img_width + padding + img_width + 3, 
                f'{class_names[label]} \n (sim = {similarity:.2f})',  # Add the similarity score
                ha='center', va='top', fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='none', pad=0))
    
    # Add similarity score
    # for i, dist in enumerate(distances):
    #     similarity = 1 - dist[0]  # Convert cosine distance to similarity
    #     ax.text(i * full_width + img_width // 2, img_width + padding + img_width // 2,
    #             f'{similarity:.2f}', ha='center', va='center', fontsize=9,
    #             color='white', bbox=dict(facecolor='black', alpha=0.7, pad=1))
    
    # Add row titles
    ax.text(-5, img_width // 2, 'Test Image', fontsize=12, va='center', ha='right', rotation=90)
    ax.text(-5, img_width + padding + img_width // 2, 'NN Train Image', fontsize=12, va='center', ha='right', rotation=90)
    
    # Add title
    plt.suptitle("Class-Constrained Nearest Neighbors: Test Images and Their Corresponding Nearest Neighbors", fontsize=14)
    
    # Save the figure
    fig_path = os.path.join(output_dir, "class_constrained_nearest_neighbors.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()
    
    print(f"Saved nearest neighbors plot to {fig_path}")
    
    # Print individual matches
    print("\nIndividual matches:")
    for i, (test_idx, nn_idx) in enumerate(zip(selected_indices, nearest_indices)):
        test_label = test_dataset[test_idx][1]
        nn_label = train_dataset[nn_idx[0]][1]
        similarity = 1 - distances[i][0]
        
        assert test_label == nn_label, "Class constraint violation!"
        print(f"✓ Test: {class_names[test_label]} → NN: {class_names[nn_label]} (similarity: {similarity:.3f})")


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define data transforms for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    # Initialize the matcher
    matcher = ClassConstrainedMatcher(device)
    
    # Fit on training data
    matcher.fit(train_dataset)
    
    # Randomly select one image per class from test set
    selected_indices = []
    for class_idx in range(10):
        class_images = np.where(np.array(test_dataset.targets) == class_idx)[0]
        selected_idx = np.random.choice(class_images)
        selected_indices.append(selected_idx)
    
    # Find nearest neighbors with class constraint
    nearest_indices, distances = matcher.find_nearest_neighbors(test_dataset, selected_indices)
    
    # Visualize results
    output_dir = "results/class_constrained_nn_plots/"
    visualize_nearest_neighbors(test_dataset, train_dataset, selected_indices, nearest_indices, distances, output_dir)