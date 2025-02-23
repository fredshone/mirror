from typing import List
import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch import optim


class VAE(LightningModule):
    def __init__(
        self,
        names: list,
        encodings: list,
        encoder: nn.Module,
        decoder: nn.Module,
        beta: float,
        lr: float,
        verbose: bool = False,
    ):
        super().__init__()
        assert len(names) == len(encodings)
        self.names = names
        self.encodings = encodings

        self.encoder = encoder
        self.decoder = decoder

        assert beta <= 1
        self.beta = beta
        self.lr = lr
        self.verbose = verbose

    def forward(
        self,
        x: Tensor,
        target=None,
        **kwargs,
    ) -> List[Tensor]:
        print(x.shape)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        log_probs_x = self.decode(z)
        return [log_probs_x, mu, log_var, z]

    def encode(self, input: Tensor) -> list[Tensor]:
        return self.encoder(input)

    def decode(self, z: Tensor, **kwargs) -> List[Tensor]:
        return self.decoder(z)

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
        mses = []
        nlls = []

        assert len(self.names) == len(log_probs) == len(targets[0])

        for i, (name, (etype, _), lprobs) in enumerate(
            zip(self.names, self.encodings, log_probs)
        ):
            target = targets[:, i]
            if etype == "numeric":
                loss = nn.functional.mse_loss(lprobs, target)
                recons.append(loss)
                mses.append(loss)
                verbose_metrics[f"recon_mse_{name}"] = loss
            elif etype == "categorical":
                loss = nn.functional.nll_loss(lprobs, target)
                recons.append(loss)
                nlls.append(loss)
                verbose_metrics[f"recon_nll_{name}"] = loss
        recon = sum(recons) / len(recons)
        b_recon = (1 - self.beta) * recon
        recon_mse = sum(mses) / len(mses)
        recon_nll = sum(nlls) / len(nlls)

        kld = self.kld(mu, log_var)
        b_kld = self.beta * kld

        loss = b_recon + b_kld

        metrics = {
            "loss": loss,
            "kld": b_kld,
            "recon": b_recon,
            "recon_mse": recon_mse,
            "recon_nll": recon_nll,
        }
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

    def predict(self, z: Tensor, device: int, **kwargs) -> Tensor:
        z = z.to(device)
        prob_samples = torch.exp(self.decode(z, **kwargs))
        return prob_samples

    def infer(self, x: Tensor, device: int, **kwargs) -> Tensor:
        log_probs_x, _, _, z = self.forward(x, **kwargs)
        prob_samples = torch.exp(log_probs_x)
        prob_samples = prob_samples.to(device)
        z = z.to(device)
        return prob_samples, z

    def training_step(self, batch, batch_idx):
        log_probs, mu, log_var, _ = self.forward(batch)
        train_losses = self.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            targets=batch,
        )
        self.log_dict(
            {key: val.item() for key, val in train_losses.items()}, sync_dist=True
        )
        return train_losses["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        log_probs, mu, log_var, _ = self.forward(batch)
        val_loss = self.loss_function(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            targets=batch,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
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

    def predict_step(self, batch, gen: bool = True):
        if gen:  # generative process (from zs)
            return (
                self.predict(batch, device=self.curr_device),
                batch,
            )
        # inference process
        preds, zs = self.infer(batch, device=self.curr_device)
        return batch, preds, zs
