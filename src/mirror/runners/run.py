from pathlib import Path

import polars as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.random import seed as seeder

from mirror.dataloaders.loader import DataModule
from mirror.encoders import (
    TableEncoder,
    XYDataset,
)
from mirror.runners import helpers


def run_command(
    config: dict,
    verbose: bool = False,
    gen: bool = True,
    test: bool = False,
    infer=True,
) -> None:
    """
    Train, test and generate.

    Args:
        config (dict): The configuration dictionary.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        gen (bool, optional): Whether to generate synthetic data. Defaults to True.
        test (bool, optional): Whether to test the model. Defaults to False.
        infer (bool, optional): Whether to infer the model. Defaults to True.

    Returns:
        None

    """
    save_dir = Path(config.get("logging", {}).get("dir", "logs"))
    project = str(config.get("logging", {}).get("project"))
    name = str(config.get("logging", {}).get("name"))
    
    # create directories
    save_dir.mkdir(exist_ok=True, parents=True)

    logger = WandbLogger(save_dir=save_dir, project=project, name=name)

    seed = config.pop("seed", seeder())
    torch.manual_seed(seed)
    verbose = config.get("verbose", False)

    x, y = helpers.load_data(config)
    x_encoder = TableEncoder(x, verbose=verbose)
    x_dataset = x_encoder.encode(data=x)
    y_encoder = TableEncoder(y, verbose=verbose)
    y_dataset = y_encoder.encode(data=y)

    xy_dataset = XYDataset(x_dataset, y_dataset)
    datamodule = DataModule(
        dataset=xy_dataset,
        **config.get("datamodule", {})
    )

    model = helpers.build_cvae(config=config, x_encoder=x_encoder, y_encoder=y_encoder)
    
    callbacks = helpers.build_callbacks(config)

    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        **config.get("trainer", {}),
    )

    trainer.fit(model=model, train_dataloaders=datamodule)

    checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)
    data_dir = checkpoint_dir.parent / "data"
    data_dir.mkdir(exist_ok = True) 

    gen_loader = helpers.build_gen_loader(config, y_dataset=y_dataset)
    
    ys, xs, zs = helpers.generate(config, dataloader=gen_loader, trainer=trainer)

    y_synth = y_encoder.decode(ys)
    x_synth = x_encoder.decode(xs).drop("pid")
    synth = pl.concat([y_synth, x_synth], how="horizontal")

    synth.write_csv(
        data_dir / "synthetic.csv"
    )

    pl.DataFrame(zs.detach().numpy()).write_csv(
        data_dir / "zs.csv", include_header=False
    )

    
