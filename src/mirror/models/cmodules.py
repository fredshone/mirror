from typing import Tuple

from torch import Tensor, nn, stack


class ConditionalBlock(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        encoder_sizes: list,
        depth: int,
        hidden_size: int,
        activation: bool = True,
        normalize: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = Embedder(encoder_types, encoder_sizes, hidden_size)
        self.ff = FFBlock(
            hidden_size,
            hidden_size,
            depth,
            hidden_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )

    def forward(self, y: Tensor) -> Tensor:
        h = self.embed(y)
        h = self.ff(h)
        return h


class EncoderBlock(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        encoder_sizes: list,
        depth: int,
        hidden_size: int,
        latent_size: int,
        activation: bool = True,
        normalize: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = Embedder(encoder_types, encoder_sizes, hidden_size)
        self.ff1 = FFBlock(
            hidden_size,
            hidden_size,
            depth,
            hidden_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.ff2 = FFBlock(
            hidden_size,
            hidden_size,
            depth,
            hidden_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)

    def forward(
        self, x: Tensor, h_conditional: Tensor
    ) -> Tuple[Tensor, Tensor]:
        h = self.embed(x)
        h = self.ff1(h)
        h = h + h_conditional
        h = self.ff2(h)
        return self.fc_mu(h), self.fc_var(h)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        encoder_sizes: list,
        depth,
        hidden_size,
        latent_size,
        activation: bool = True,
        normalize: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ff1 = FFBlock(
            latent_size,
            hidden_size,
            depth,
            hidden_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.ff2 = FFBlock(
            hidden_size,
            hidden_size,
            depth,
            hidden_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )

        self.embeds = []
        for type, size in zip(encoder_types, encoder_sizes):
            if type == "continuous":
                self.embeds.append(
                    nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
                )
            if type == "categorical":
                self.embeds.append(
                    nn.Sequential(
                        nn.Linear(hidden_size, size), nn.LogSoftmax(dim=-1)
                    )
                )

    def forward(self, z: Tensor, h_conditional: Tensor) -> Tensor:
        h = self.ff1(z)
        h = h + h_conditional
        h = self.ff2(h)
        xs = [embed(h) for embed in self.embeds]
        return xs


class Embedder(nn.Module):
    def __init__(self, encoder_types: list, encoder_sizes: list, embed_size):
        super().__init__()
        self.encoder_types = encoder_types
        embeds = []
        for type, size in zip(encoder_types, encoder_sizes):
            if type == "continuous":
                embeds.append(NumericEmbed(embed_size))
            if type == "categorical":
                embeds.append(nn.Embedding(size, embed_size))
        self.embeds = nn.ModuleList(embeds)

    def forward(self, x: Tensor) -> Tensor:
        xs = []
        for i, (type, embed) in enumerate(zip(self.encoder_types, self.embeds)):
            col = x[:, i]
            if type == "categorical":
                col = col.int()
            xs.append(embed(col))
        # consider splitting categorical and continuous in future
        return stack(xs, dim=-1).sum(dim=-1)


class Noop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class FFBlock(nn.Module):
    # check about removing extra bias
    # add skipping connections
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        output_size: int,
        activation: bool = True,
        normalize: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 0:
            raise ValueError("hidden_n must be non-negative")
        if depth == 0 and input_size == output_size:
            block = [Noop()]
        elif depth < 2:
            block = [nn.Linear(input_size, output_size)]
        else:
            block = [nn.Linear(input_size, hidden_size)]
            for _ in range(depth - 1):
                if activation:
                    block.append(nn.ReLU())
            block.extend([nn.Linear(hidden_size, output_size)])
        if normalize:
            block.append(nn.LayerNorm(hidden_size))
        if dropout > 0:
            block.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class NumericEmbed(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(1, hidden_size)

    def forward(self, x):
        return self.fc(x)
