import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, layer_sizes: list[int], dropout: float=0.1):
        super(MLP, self).__init__()
        layers = []
        flat = nn.Flatten(start_dim=1)
        layers.append(flat)
        old_size = input_size
        for layer in layer_sizes:
            layers.append(nn.Linear(old_size, layer))
            layers.append(nn.BatchNorm1d(layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            old_size = layer

        layers.append(nn.Linear(old_size, output_size))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, src, mask):
        out = self.model(src)
        if mask is not None:
            out *= mask
        return out 
