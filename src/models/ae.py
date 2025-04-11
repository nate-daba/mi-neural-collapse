import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class AE(nn.Module):
    def __init__(self, 
                 input_channels: int = 3,
                 latent_dim: int = 64,
                 hidden_dims: list[int] = [32, 64, 128]):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Encoder
        modules = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # Flatten to latent vector
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(hidden_dims[-1]*4*4, latent_dim)

        # Latent to decoder input
        self.fc_dec = nn.Linear(latent_dim, hidden_dims[-1]*4*4)

        # Decoder
        hidden_dims_rev = hidden_dims[::-1]
        modules = []
        for i in range(len(hidden_dims_rev) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_rev[i], hidden_dims_rev[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)

        # Final layer to reconstruct input
        self.final_layer = nn.ConvTranspose2d(hidden_dims[0], input_channels,
                                              kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)         # B x C x 4 x 4
        x = self.flatten(x)         # B x (C*4*4)
        z = self.fc_enc(x)          # B x latent_dim
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_dec(z)          # B x (C*4*4)
        x = x.view(-1, self.hidden_dims[-1], 4, 4)
        x = self.decoder(x)
        x = torch.sigmoid(self.final_layer(x))  # B x 3 x 32 x 32
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
if __name__ == "__main__":
    model = AE(input_channels=3, latent_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Assuming CIFAR-10 input shape (batch_size, channels, height, width)
    summary(model, input_size=(1, 3, 32, 32))  # batch size = 1 just for summary
