import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, 
                 input_channels: int = 3,
                 out_channels: int = 16,
                 latent_dim: int = 64,
                 act_fn=nn.ReLU()):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, 3, padding=1),  # (32, 32)
            act_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, 2 * out_channels, 3, padding=1, stride=2),  # (16, 16)
            act_fn,
            nn.Conv2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(2 * out_channels, 4 * out_channels, 3, padding=1, stride=2),  # (8, 8)
            act_fn,
            nn.Conv2d(4 * out_channels, 4 * out_channels, 3, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear(4 * out_channels * 8 * 8, latent_dim),
            act_fn
        )

        # Decoder
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4 * out_channels * 8 * 8),
            act_fn
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * out_channels, 4 * out_channels, 3, padding=1),  # (8, 8)
            act_fn,
            nn.ConvTranspose2d(4 * out_channels, 2 * out_channels, 3, padding=1, stride=2, output_padding=1),  # (16, 16)
            act_fn,
            nn.ConvTranspose2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(2 * out_channels, out_channels, 3, padding=1, stride=2, output_padding=1),  # (32, 32)
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(out_channels, input_channels, 3, padding=1)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        z = self.linear(z)
        z = z.view(z.size(0), -1, 8, 8)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
if __name__ == "__main__":
    model = AE(input_channels=3, latent_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Assuming CIFAR-10 input shape (batch_size, channels, height, width)
    summary(model, input_size=(1, 3, 32, 32))  # batch size = 1 just for summary
