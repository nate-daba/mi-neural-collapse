import os
from datetime import datetime
from typing import Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray

def print_data_shapes(classwise_data: Dict[int, Dict[str, NDArray]]) -> None:
    """
    Prints the shape of train and test data for each class.
    """
    for class_idx, data in classwise_data.items():
        X_train = data["train"]
        X_test = data["test"]
        print(f"Class {class_idx}: Train shape {X_train.shape}, Test shape {X_test.shape}")

def print_mi_results(mi_per_class: Dict[int, float]) -> None:
    """
    Prints the mutual information for each class.
    """
    print("\nPer-class MI (Train vs Test):")
    for class_idx, mi in mi_per_class.items():
        print(f"  Class {class_idx}: {mi:.4f} nats")
        
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

def get_nn_pair_predictions(nn_pairs_by_class, model, device):
    """
    Get the predictions of the test-train pairs for each class.

    Args:
        nn_pairs_by_class (dict): Dictionary containing the nearest neighbor pairs for each class.
        model: The trained model used to generate predictions.
        device: The device (CPU/GPU) to run the model on.

    Returns:
        dict: A dictionary containing the predictions for the test and train images in the pairs.
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for class_idx, data in nn_pairs_by_class.items():
            test_preds = []
            train_preds = []

            # Iterate over each pair of test and train images
            pairs = data['pairs']  # Get the pairs of test and train indices

            for i in range(len(pairs)):
                test_img = pairs[i, 0]  # Test image
                train_img = pairs[i, 1]  # Train image

                # Get the model's predictions for both test and train images
                test_img = test_img.unsqueeze(0).to(device)  # Add batch dimension
                train_img = train_img.unsqueeze(0).to(device)  # Add batch dimension

                test_pred = model(test_img)[0]  # Model output for test image
                train_pred = model(train_img)[0]  # Model output for train image

                _, test_class = torch.max(test_pred, dim=1)
                _, train_class = torch.max(train_pred, dim=1)

                test_preds.append(test_class.item())
                train_preds.append(train_class.item())

            predictions[class_idx] = {'test_preds': test_preds, 'train_preds': train_preds}
    
    return predictions

def build_pmf_table(predictions, num_classes):
    """
    Build the PMF table from the predictions for test and train images.

    Args:
        predictions (dict): A dictionary containing the predictions for test and train images in the pairs.
        num_classes (int): The number of classes in the classification task.

    Returns:
        torch.Tensor: PMF table representing the joint probability distribution.
    """
    pmf_table = torch.zeros((num_classes, num_classes), dtype=torch.float32)

    for class_idx, data in predictions.items():
        test_preds = data['test_preds']
        train_preds = data['train_preds']

        # Iterate over test and train predictions to populate the PMF table
        for test_class, train_class in zip(test_preds, train_preds):
            pmf_table[train_class, test_class] += 1

    # Normalize to get probabilities
    total_count = pmf_table.sum()
    if total_count > 0:
        pmf_table /= total_count

    return pmf_table

def get_pmf_table(model, nn_pairs, device, num_classes):
    """
    Update the PMF table after each epoch.

    Args:
        model: The trained model.
        nn_pairs (dict): Dictionary containing the nearest neighbor pairs for each class.
        device: The device (CPU/GPU) to run the model on.
        num_classes (int): The number of classes in the classification task.

    Returns:
        torch.Tensor: The PMF table.
    """

    # Get predictions for the test and train images in the pairs
    predictions = get_nn_pair_predictions(nn_pairs, model, device)

    # Build the PMF table using the predictions
    pmf_table = build_pmf_table(predictions, num_classes)

    return pmf_table

def plot_pmf_table(pmf_table: torch.Tensor, class_names: list) -> plt.Figure:
    """
    Create a color-coded confusion matrix from the PMF table and plot it.
    The matrix will have training class predictions as rows and test class predictions as columns.
    The values in the matrix are the PMF values.

    Args:
        pmf_table (torch.Tensor): The PMF table with shape (num_classes, num_classes).
        class_names (list): List of class names for CIFAR-10.

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    # Convert PMF table to numpy for visualization
    pmf_matrix = pmf_table.cpu().numpy()

    # Set up the plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(pmf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                     cbar_kws={'label': 'PMF Value'}, annot_kws={"size": 10}, linewidths=0.5, linecolor='black')

    ax.set_xlabel('Test Predictions')
    ax.set_ylabel('Train Predictions')
    ax.set_title('PMF Table: Training vs Test Predictions')

    # Adjust layout for better fit
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return plt

