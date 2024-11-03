import torch
from torchvision.ops import MLP


class MNIST_MLP(torch.nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.mlp = MLP(in_channels=28 * 28, hidden_channels=[128, 64, 10])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.mlp(x)
