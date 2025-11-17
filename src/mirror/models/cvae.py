from typing import List, Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim


class CVAE(LightningModule):
    def __init__(
        self,
        embedding_names: list,
        embedding_types: list,
        labels_encoder_block: nn.Module,
        encoder_block: nn.Module,
        decoder_block: nn.Module,
        beta: float,
        lr: float,
        embedding_weights: Optional[list] = None,
        verbose: bool = False,
    ):
        super().__init__()
        if not (embedding_names and embedding_types):
            raise ValueError("Embedding names and types must be provided")
        if len(embedding_names) != len(embedding_types):
            raise ValueError(
                "Embedding names and types must have the same length"
            )
        self.embedding_names = embedding_names
        self.embedding_types = embedding_types
        if embedding_weights:
            self.embedding_weights = embedding_weights
        else:
            self.embedding_weights = [None] * len(embedding_names)

        self.labels_encoder_block = labels_encoder_block
        self.encoder_block = encoder_block
        self.decoder_block = decoder_block

        self.beta = beta
        self.lr = lr
        self.verbose = verbose
        self.save_hyperparameters(
            ignore=["labels_encoder_block", "encoder_block", "decoder_block"]
        )
        criterion = []
        for etype, weights in zip(self.embedding_types, self.embedding_weights):
            if etype == "continuous":
                criterion.append(nn.MSELoss())
            # move weights to device
            elif etype == "categorical":
                criterion.append(nn.NLLLoss(weight=weights))
            else:
                raise ValueError(f"Unknown embedding type: {etype}")

        self.criterion = nn.ModuleList(criterion)

    def forward(
        self, x: Tensor, y: Tensor, target=None, **kwargs
    ) -> List[Tensor]:
        h_y = self.labels_encoder_block(y)
        mu, log_var = self.encode(x, h_y)
        z = self.reparameterize(mu, log_var)
        log_probs_x = self.decode(z, h_y)
        return [log_probs_x, mu, log_var, z]

    def encode(self, x: Tensor, hidden_y: Tensor) -> list[Tensor]:
        return self.encoder_block(x, hidden_y)

    def decode(self, z: Tensor, hidden_y: Tensor, **kwargs) -> List[Tensor]:
        return self.decoder_block(z, hidden_y)

    def loss_function(
        self,
        log_probs: List[Tensor],
        mu: Tensor,
        log_var: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> dict:
        verbose_metrics = {}
        recons = []

        for i, (name, etype, lprobs, criterion) in enumerate(
            zip(self.embedding_names, self.embedding_types, log_probs, self.criterion)
        ):
            target = targets[:, i]
            if etype == "continuous":
                loss = criterion(torch.exp(lprobs), target)
                recons.append(loss)
                verbose_metrics[f"recon_mse_{name}"] = loss
            elif etype == "categorical":
                loss = criterion(lprobs, target.long())
                recons.append(loss)
                verbose_metrics[f"recon_nll_{name}"] = loss
            else:
                raise ValueError(f"Unknown encoding for {name}, type: {etype}")
        recon = sum(recons) / len(recons)
        b_recon = (1 - self.beta) * recon

        kld = self.kld(mu, log_var)
        b_kld = self.beta * kld

        loss = b_recon + b_kld

        metrics = {"loss": loss, "kld": b_kld, "recon": b_recon}
        if self.verbose:
            metrics.update(verbose_metrics)

        return metrics

    def kld(self, mu: Tensor, log_var: Tensor) -> Tensor:
        return torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (eps * std) + mu

    def predict(self, z: Tensor, y: Tensor, **kwargs) -> List[Tensor]:
        h_y = self.labels_encoder_block(y)
        prob_samples = [
            torch.exp(probs) for probs in self.decode(z, h_y, **kwargs)
        ]
        return y, prob_samples, z

    def infer(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        log_probs_x, _, _, z = self.forward(x, y, **kwargs)
        prob_samples = torch.exp(log_probs_x)
        return prob_samples, z

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_probs, mu, log_var, _ = self.forward(x, y)
        train_losses = self.loss_function(
            log_probs=log_probs, mu=mu, log_var=log_var, targets=x
        )
        self.log_dict(
            {key: val.item() for key, val in train_losses.items()},
            sync_dist=True,
        )
        return train_losses["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        log_probs, mu, log_var, _ = self.forward(x, y)
        loss = self.loss_function(
            log_probs=log_probs, mu=mu, log_var=log_var, targets=x
        )
        self.log_dict(
            {f"val_{key}": val.item() for key, val in loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

    def predict_step(self, batch):
        z, y = batch
        return self.predict(z, y)
