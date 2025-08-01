from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch import concat

from mirror.dataloaders.loader import DataModule
from mirror.dataloaders.z_loader import build_latent_dataloader
from mirror.encoders.table_encoder import TableEncoder
from mirror.models.modules import Decoder, Encoder
from mirror.models.vae import VAE

LOGDIR = Path("logs")
SIZE = 10000
BATCH = 1024
LATENT_SIZE = 8
MAX_EPOCHS = 1
PATIENCE = 20

LOGDIR.mkdir(parents=True, exist_ok=True)
log_dir = str(Path(LOGDIR))
print(f"Writting logs to: {log_dir}")

census = pd.read_csv("~/Data/sample.csv")
census = census.drop("resident_id_m", axis=1)
census_encoder = TableEncoder(data=census, verbose=True)

dataset = census_encoder.encode(data=census)
dataloader = DataModule(
    dataset=dataset,
    val_split=0.1,
    test_split=None,
    train_batch_size=BATCH,
    val_batch_size=BATCH,
    test_batch_size=BATCH,
    num_workers=4,
    pin_memory=False,
)

encoder = Encoder(
    encoder_types=census_encoder.types(),
    encoder_sizes=census_encoder.sizes(),
    embed_size=32,
    hidden_n=2,
    hidden_size=64,
    latent_size=LATENT_SIZE,
)
decoder = Decoder(
    encoder_types=census_encoder.types(),
    encoder_sizes=census_encoder.sizes(),
    embed_size=32,
    hidden_n=2,
    hidden_size=64,
    latent_size=LATENT_SIZE,
)
vae = VAE(
    encoder_names=census_encoder.names(),
    encoder_types=census_encoder.types(),
    encoder_block=encoder,
    decoder_block=decoder,
    beta=0.001,
    lr=0.001,
    verbose=True,
)

wandb_logger = WandbLogger(project="census", dir=log_dir)

callbacks = [
    ModelCheckpoint(
        dirpath=Path(log_dir, "checkpoints"),
        monitor="val_loss",
        save_top_k=2,
        save_weights_only=False,
    ),
    EarlyStopping(
        monitor="val_loss", patience=PATIENCE, stopping_threshold=0.0
    ),
    LearningRateMonitor(),
]

trainer = Trainer(
    check_val_every_n_epoch=1,
    max_epochs=MAX_EPOCHS,
    logger=wandb_logger,
    callbacks=callbacks,
)
trainer.fit(model=vae, train_dataloaders=dataloader)
trainer.validate(model=vae, dataloaders=dataloader)

zloader = build_latent_dataloader(SIZE, LATENT_SIZE, BATCH)
synth = trainer.predict(ckpt_path="best", dataloaders=zloader)
synthetics, zs = zip(*synth)
synthetic_schedules = concat(synthetics)
zs = concat(zs)

synthetics.to_csv(log_dir / "synthetics.csv")
pd.DataFrame(zs.cpu().numpy()).to_csv(
    Path(log_dir, "synthetic_zs.csv"), index=False, header=False
)
