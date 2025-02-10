from typing import Tuple
from torch import nn, stack, Tensor


class LinearEncoder(nn.Module):
    def __init__(self, encodings: dict, embed_size, hidden_n, hidden_size, latent_size):
        super().__init__()
        embeds = []
        for type, encoding in encodings:
            if type == "numeric":
                embeds.append(NumericEmbed(embed_size))
            if type == "categorical":

                embeds.append(nn.Embedding(len(encoding), embed_size))
        self.embeds = nn.ModuleList(embeds)
        hidden = [nn.Linear(embed_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_n - 1):
            hidden.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU,
                ]
            )
        self.hidden = nn.Sequential(hidden)

        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)

    def forward(self, xs: Tuple[Tensor]):
        assert len(xs) == len(self.embeds)
        x = stack([embed(x) for embed, x in zip(self.embeds, xs)], dim=-1).sum(dim=-1)
        x = self.sequential(x)
        return self.fc_mu(x), self.fc_var(x)


class LinearDecoder(nn.Module):
    def __init__(self, encodings: dict, embed_size, hidden_n, hidden_size, latent_size):
        super().__init__()

        hidden = [nn.Linear(latent_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_n - 1):
            hidden.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                ]
            )
        hidden.append(nn.Linear(hidden_size, embed_size))
        self.hidden = nn.Sequential(hidden)

        self.embeds = []
        for type, encoding in encodings:
            if type == "numeric":
                self.embeds.append(
                    nn.Sequential(nn.Linear(embed_size, 1), nn.Sigmoid())
                )
            if type == "categorical":
                self.embeds.append(
                    nn.Sequential(
                        nn.Linear(embed_size, len(encoding)), nn.Softmax(dim=-1)
                    )
                )

    def forward(self, z: Tensor):
        x = self.hidden(z)
        xs = [embed(x) for embed in self.embeds]
        return xs


class NumericEmbed(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(1, hidden_size)

    def forward(self, x):
        return self.fc(x)
