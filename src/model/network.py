import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # TODO
        pass
