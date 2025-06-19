# Neural Collapse Mutual Information Analysis

A comprehensive framework for analyzing neural collapse and generalization in deep neural networks using mutual information as a proxy metric. This codebase implements both discrete and latent mutual information estimation for tracking neural collapse dynamics during ResNet-18 training on CIFAR-10.

## 🎯 Overview

This repository provides tools for:
- **Discrete MI Estimation**: Track prediction consistency across semantically similar train-test pairs
- **Latent MI Estimation**: Analyze feature-level mutual information dynamics using dimensionality reduction
- **Neural Collapse Metrics**: Monitor NC1-NC2 phenomena during training

## 📁 Repository Structure

```
mi-neural-collapse/
├── data/                          # Dataset and preprocessed files
│   ├── cifar-10-batches-py/       # CIFAR-10 raw data
│   ├── encoded_CIFAR10/           # CLIP-encoded features
│   └── nn_pairs/                  # Nearest neighbor pairs
├── src/                           # Source code modules
│   ├── cli/                       # Command-line interface
│   ├── data/                      # Data processing utilities
│   ├── metrics/                   # Metrics computation
│   ├── mi/                        # Mutual information estimators
│   ├── models/                    # Neural network models (AE, AEC)
│   ├── nc/                        # Neural collapse metrics
│   ├── train/                     # Training scripts
│   └── utils/                     # Utility functions
├── scripts/                       # Preprocessing scripts
├── tests/                         # Unit tests
├── main.py                        # Main entry point
├── sandbox.ipynb                  # Development notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda or Miniconda

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:nate-daba/mi-neural-collapse.git
   cd mi-neural-collapse
   ```

2. **Create conda environment**
   ```bash
   conda create -n mi-collapse python=3.9
   conda activate mi-collapse
   ```

3. **Install PyTorch with CUDA support via conda**
   ```bash
   # For CUDA 11.8 (adjust based on your CUDA version)
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CPU-only installation
   # conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

4. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```



## 🚀 Usage

### Quick Start

1. **Generate Nearest Neighbor Pairs**
   - Ensure you have the CIFAR-10 dataset downloaded in `data/cifar-10-batches-py/`.
   - Run the nearest neighbor pair generation script (this should save pairs in `data/nn_pairs/` as `class_constrained_nn_pairs.pt`):
   ```bash
   python src/utils/CLIP_matcher.py
   ```

2. **Basic training with default configuration**
   ```bash
   python src/train/train_resnet.py
   ```

3. View training progress and metrics in the Weights & Biases dashboard:
   - Log in to your W&B account: `wandb login`
   - Open the dashboard at link in your terminal after training starts.


### Important Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 350 | Number of training epochs |
| `batch_size` | 256 | Training batch size |
| `learning_rate` | 0.1 | Initial learning rate |
| `track_nc_metrics` | True | Enable neural collapse metrics |
| `track_mi_metrics` | True | Enable mutual information metrics |
| `eval_freq` | 1 | Frequency of comprehensive evaluation |
| `lmi_dim` | 64 | Latent dimension for LMI estimation |
| `use_wandb` | True | Enable Weights & Biases logging |

## 📊 Expected Outputs

### Training Logs
```
Epoch [1/350]
Train Loss: 1.4197, Train Acc: 49.37%
Test Loss: 1.4827, Test Acc: 47.16%
Discrete MI: 0.2560 bits
NC1 (pinv): 5.742773
New best model saved: checkpoints/2025-06-16-15-31-48/resnet18_epoch_1_acc_47.16.pth
Computing comprehensive metrics...
[INFO] Estimating latent MI with 10000 samples and 64 latent dimensions.
[INFO] Using training set for estimation.
epoch 127 (of max 300) 🌻🌻🌻🌻
```

### Saved Artifacts
- **Checkpoints**: `checkpoints/YYYY-MM-DD-HH-MM-SS/checkpoint_val_loss_X.XX.pth`
- **Metrics**: Logged to Weights & Biases dashboard
- **Visualizations**: PMF tables, training accuracy and loss curves, NC progression, discrete and latent MI curves

### Key Metrics
- **Discrete MI**: Prediction consistency measure (bits)
- **Latent MI**: Feature-level information sharing (bits)
- **NC1-NC4**: Neural collapse indicators
- **Test Accuracy**: Classification performance


## 📈 Monitoring and Visualization

### Weights & Biases Integration
The framework automatically logs:
- Training/validation losses and accuracies
- Mutual information dynamics
- Neural collapse progression
- Gradient and activation statistics
- PMF visualization tables

### Key Visualizations
1. **PMF Tables**: Joint probability distributions of train-test predictions
2. **MI Curves**: Discrete and latent MI evolution during training
3. **NC Metrics**: NC1-NC4 progression tracking
4. **Training Curves**: Loss and accuracy over time


## 📚 Key Components

### Core Modules
- **`train_resnet.py`**: Main training framework with comprehensive metrics
- **`metrics_tracker.py`**: Unified metrics computation and tracking
- **`discrete_mi.py`**: Discrete mutual information estimation
- **`latent_mi.py`**: High-dimensional MI estimation via dimensionality reduction
- **`nc.py`**: Neural collapse metrics (NC1-NC4)

### Utilities
- **`CLIP_matcher.py`**: Semantic pairing using CLIP embeddings
- **`nn_pair_processor.py`**: Nearest neighbor pair processing
- **`utils.py`**: General utility functions and plotting

## 🔗 References

- **Neural Collapse**: Papyan, V., Han, X. Y., & Donoho, D. L. (2020). *Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training.* [arXiv:2008.08186](https://arxiv.org/abs/2008.08186)

- **CLIP**: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). *Learning Transferable Visual Models from Natural Language Supervision.* [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

- **Latent MI Estimation**: Gowri, A., Addepalli, S., Babu, R. V. (2024). *Approximating Mutual Information in High-Dimensional Spaces: A Novel Approach Using Variational Autoencoders.* [GitHub Repository](https://github.com/gowriaddepalli/latent-mi)
