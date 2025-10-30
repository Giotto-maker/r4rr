import torch
import torch.nn as nn

# & SingleMLP backbone for Kandinsky (outputs both shape and colour of single primitive)
class SingleJointMLP(nn.Module):
    NAME = "singlejointmlp"

    def __init__(
        self,
        img_channels=3,
        img_size=64,
        hidden_channels=32,
        latent_dim=6,    # final output size = 6 scores per image
        label_dim=20,
        dropout=0.5,
    ):
        """
        Initializes the SingleJointMLP model.
        Args:
            img_channels (int, optional): Number of channels in the input images. Defaults to 3.
            img_size (int, optional): Height and width of the input images (assumed square). Defaults to 64.
            hidden_channels (int, optional): Number of hidden channels (not directly used in this implementation). Defaults to 32.
            latent_dim (int, optional): Dimension of the latent output (number of scores per image). Defaults to 6.
            label_dim (int, optional): Dimension of the label space. Defaults to 20.
            dropout (float, optional): Dropout rate (not directly used in this implementation). Defaults to 0.5.
        Attributes:
            img_channels (int): Number of channels in the input images.
            img_size (int): Height and width of the input images.
            hidden_channels (int): Number of hidden channels.
            latent_dim (int): Dimension of the latent output.
            label_dim (int): Dimension of the label space.
            backbone (nn.Sequential): Sequential neural network serving as the feature extractor and encoder.
        """
        
        super(SingleJointMLP, self).__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.label_dim = label_dim

        self.backbone = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.img_channels * self.img_size * self.img_size,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.latent_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        assert x.size(1) == self.img_channels and x.size(2) == self.img_size and x.size(3) == self.img_size, (
            f"Expected input (B, {self.img_channels}, {self.img_size}, {self.img_size}), got {tuple(x.shape)}"
        )

        logits = self.backbone(x)   # shape (B, 6)
        return logits
