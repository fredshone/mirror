import torch
from torch import Tensor


class UnitWeight:
    def __init__(self):
        self.weights = []

    def add_categorical(self, x: Tensor):
        self.weights.append(torch.ones_like(x))

    def add_numeric(self, x: Tensor):
        self.weights.append(torch.ones_like(x))

    def weights(self, xs: Tensor) -> Tensor:
        return torch.stack(self.weights, dim=-1)


class MarginalWeights:
    def __init__(self):
        self.weights = []

    def add_categorical(self, x: Tensor):
        self.weights.append(torch.ones_like(x))

    def add_numeric(self, x: Tensor):
        self.weights.append(torch.ones_like(x))

    def weights(self, xs: Tensor) -> Tensor:
        return torch.stack(self.weights, dim=-1)
