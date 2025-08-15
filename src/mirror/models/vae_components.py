from torch import Tensor, nn, stack


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        encoder_sizes: list,
        embed_size,
        hidden_n,
        hidden_size,
        latent_size,
    ):
        super().__init__()
        self.encoder_types = encoder_types
        embeds = []
        for type, size in zip(encoder_types, encoder_sizes):
            if type == "continuous":
                embeds.append(NumericEmbed(embed_size))
            if type == "categorical":
                embeds.append(nn.Embedding(size, embed_size))
        self.embeds = nn.ModuleList(embeds)
        hidden = [nn.Linear(embed_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_n - 1):
            hidden.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.hidden = nn.Sequential(*hidden)

        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)

    def forward(self, x: Tensor):
        xs = []
        for i, (type, embed) in enumerate(zip(self.encoder_types, self.embeds)):
            col = x[:, i]
            if type == "categorical":
                col = col.int()
            xs.append(embed(col))
        # consider splitting categorical and continuous in future
        x = stack(xs, dim=-1).sum(dim=-1)  # Add all embeddings together
        x = self.hidden(x)
        return self.fc_mu(x), self.fc_var(x)


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        encoder_sizes: list,
        embed_size,
        hidden_n,
        hidden_size,
        latent_size,
    ):
        super().__init__()

        hidden = [nn.Linear(latent_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_n - 1):
            hidden.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        hidden.append(nn.Linear(hidden_size, embed_size))
        self.hidden = nn.Sequential(*hidden)

        self.embeds = []
        for type, size in zip(encoder_types, encoder_sizes):
            if type == "continuous":
                self.embeds.append(
                    nn.Sequential(nn.Linear(embed_size, 1), nn.Sigmoid())
                )
            if type == "categorical":
                self.embeds.append(
                    nn.Sequential(
                        nn.Linear(embed_size, size), nn.LogSoftmax(dim=-1)
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
