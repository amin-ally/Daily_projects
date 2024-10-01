import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Example usage in a simple MLP:
class MLP(nn.Module):
    def __init__(self):
        super(MLP).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.swish = Swish()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input for MNIST
        x = self.swish(self.fc1(x))
        x = self.fc2(x)
        return x