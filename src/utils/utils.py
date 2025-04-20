import os
from datetime import datetime
from typing import Dict
import torch
import numpy as np
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