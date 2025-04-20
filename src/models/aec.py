import torch
import torch.nn as nn

class AEC(nn.Module):
    """
    Autoencoder with a classifier head for CIFAR-10 classification.

    The encoder is a convolutional autoencoder that is extended with a linear
    classification head that predicts the class labels from the latent space representation.

    Args:
        input_channels (int): Number of input channels, typically 3 for RGB images.
        latent_dim (int): Dimensionality of the latent space representation.
        out_channels (int): Number of output channels for the encoder layers.
        num_classes (int): Number of output classes for classification (default is 10 for CIFAR-10).
    """
    def __init__(self, 
                 input_channels: int = 3, 
                 latent_dim: int = 200, 
                 out_channels: int = 16, 
                 num_classes: int = 10) -> None:
        super(AEC, self).__init__()

        # Existing encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, 3, padding=1),  # (32, 32)
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 2 * out_channels, 3, padding=1, stride=2),  # (16, 16)
            nn.ReLU(),
            nn.Conv2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * out_channels, 4 * out_channels, 3, padding=1, stride=2),  # (8, 8)
            nn.ReLU(),
            nn.Conv2d(4 * out_channels, 4 * out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * out_channels * 8 * 8, latent_dim),  # Latent dimension
            nn.ReLU()
        )

        # Classifier head (Linear layer after the encoder)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder and classifier to obtain logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        encoded = self.encoder(x)  # Output of encoder (latent space)
        logits = self.classifier(encoded)  # Classifier output
        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input through the encoder part of the model to obtain the latent features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Latent space representation of shape (batch_size, latent_dim).
        """
        return self.encoder(x)