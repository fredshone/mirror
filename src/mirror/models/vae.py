from typing import List

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim


class VAE(LightningModule):
    def __init__(
        self,
        encoder_names: list,
        encoder_types: list,
        encoder_block: nn.Module,
        decoder_block: nn.Module,
        beta: float,
        lr: float,
        verbose: bool = False,
    ):
        super().__init__()
        self.encoder_names = encoder_names
        self.encoder_types = encoder_types

        self.encoder_block = encoder_block
        self.decoder_block = decoder_block

        assert beta <= 1
        self.beta = beta
        self.lr = lr
        self.verbose = verbose
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def forward(self, x: Tensor, target=None, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        log_probs_x = self.decode(z)
        return [log_probs_x, mu, log_var, z]

    def encode(self, input: Tensor) -> list[Tensor]:
        return self.encoder_block(input)

    def decode(self, z: Tensor, **kwargs) -> List[Tensor]:
        return self.decoder_block(z)

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

        for i, (name, etype, lprobs) in enumerate(
            zip(self.encoder_names, self.encoder_types, log_probs)
        ):
            target = targets[:, i]
            if etype == "continuous":
                loss = nn.functional.mse_loss(lprobs, target)
                recons.append(loss)
                verbose_metrics[f"recon_mse_{name}"] = loss
            elif etype == "categorical":
                loss = nn.functional.nll_loss(lprobs, target.long())
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

    def predict(self, z: Tensor, **kwargs) -> List[Tensor]:
        prob_samples = [torch.exp(probs) for probs in self.decode(z, **kwargs)]
        return prob_samples, z

    def infer(self, x: Tensor, **kwargs) -> Tensor:
        log_probs_x, _, _, z = self.forward(x, **kwargs)
        prob_samples = torch.exp(log_probs_x)
        return prob_samples, z

    def training_step(self, batch, batch_idx):
        log_probs, mu, log_var, _ = self.forward(batch)
        train_losses = self.loss_function(
            log_probs=log_probs, mu=mu, log_var=log_var, targets=batch
        )
        self.log_dict(
            {key: val.item() for key, val in train_losses.items()},
            sync_dist=True,
        )
        return train_losses["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        log_probs, mu, log_var, _ = self.forward(batch)
        loss = self.loss_function(
            log_probs=log_probs, mu=mu, log_var=log_var, targets=batch
        )
        self.log_dict(
            {f"val_{key}": val.item() for key, val in loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def get_scheduler(self, optimizer):
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return scheduler

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)

        scheduler = self.get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def predict_step(self, batch):
        return self.predict(batch)
