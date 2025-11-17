from typing import Tuple

import polars as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from torch import Tensor, argmax, concat, stack
from torch.utils.data import DataLoader

from mirror.encoders import (
    YZDataset,
    ZDataset,
)
from mirror.models.cvae import CVAE
from mirror.models.cvae_components import (
    CVAEDecoderBlock,
    CVAEEncoderBlock,
    LabelsEncoderBlock,
)
from mirror.runners import defaults


def load_data(config: dict) -> Tuple[pl.DataFrame, pl.DataFrame]:
    # load data
    x_path = config["data"]["train_path"]
    x = pl.read_csv(x_path)

    # rename columns and select as required
    rename = config["data"].get("columns", {})
    if rename:
        x = x.rename(rename)
        x = x.select(list(rename.values()))

    y_cols = config["data"]["controls"]
    y = x.select(y_cols)
    x = x.drop(y_cols)
    return x, y


def build_cvae(config: dict, x_encoder, y_encoder) -> LightningModule:
    model_params = config.get("model", {})

    # encoder block to embed labels into vec with hidden size
    labels_encoder_block = LabelsEncoderBlock(
        encoder_types=y_encoder.types(),
        encoder_sizes=y_encoder.sizes(),
        depth=model_params.get("controls_encoder", {}).get("depth", defaults.DEPTH),
        hidden_size=model_params.get("controls_encoder", {}).get("hidden_size", defaults.WIDTH),
    )

    # encoder and decoder block to process census data
    encoder = CVAEEncoderBlock(
        encoder_types=x_encoder.types(),
        encoder_sizes=x_encoder.sizes(),
        depth=model_params.get("encoder", {}).get("depth", defaults.DEPTH),
        hidden_size=model_params.get("encoder", {}).get("hidden_size", defaults.WIDTH),
        latent_size=model_params.get("latent_size", defaults.LATENT_SIZE),
    )
    decoder = CVAEDecoderBlock(
        encoder_types=x_encoder.types(),
        encoder_sizes=x_encoder.sizes(),
        depth=model_params.get("decoder", {}).get("depth", defaults.DEPTH),
        hidden_size=model_params.get("decoder", {}).get("hidden_size", defaults.WIDTH),
        latent_size=model_params.get("latent_size", defaults.LATENT_SIZE),
    )

    return CVAE(
        embedding_names=x_encoder.names(),
        embedding_types=x_encoder.types(),
        embedding_weights=x_encoder.weights(),
        labels_encoder_block=labels_encoder_block,
        encoder_block=encoder,
        decoder_block=decoder,
        beta=model_params.get("beta", defaults.DEFAULT_BETA),
        lr=model_params.get("lr", defaults.DEFAULT_LR),
    )


def build_callbacks(config: dict):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.get("early_stopping", {}).get("patience", defaults.PATIENCE),
            mode="min"
            ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            # dirpath=Path(log_dir, "checkpoints"),
            save_weights_only=True,
        ),
    ]


def build_gen_loader(config: dict, y_dataset: Tensor) -> DataLoader:
    n = len(y_dataset)
    z_loader = ZDataset(n, latent_size=config.get("model", {}).get("latent_size", defaults.LATENT_SIZE))
    yz_loader = YZDataset(z_loader, y_dataset)
    return DataLoader(
        yz_loader, **config.get("gen_dataloader", {})
    )


def generate(config: dict, dataloader: DataLoader, trainer: Trainer) -> Tuple[Tensor, Tensor, Tensor]:
    ys, xs, zs = zip(*trainer.predict(dataloaders=dataloader))
    ys = concat(ys)
    # todo: currently using argmax to decode categorical variables
    xs = concat([stack([argmax(x, dim=1) for x in xb], dim=-1) for xb in xs], dim=0)
    zs = concat(zs)
    return ys, xs, zs

