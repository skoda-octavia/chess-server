import torch
import torch.nn as nn


class Heu(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_sizes: list[int],
        dropout: float = 0.1,
    ):
        super(Heu, self).__init__()
        layers = []
        # flat = nn.Flatten(start_dim=1)
        # layers.append(flat)
        old_size = input_size
        for layer in layer_sizes:
            layers.append(nn.Linear(old_size, layer))
            layers.append(nn.BatchNorm1d(layer))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            old_size = layer

        layers.append(nn.Linear(old_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = torch.argmax(x, dim=-1)
        return x
