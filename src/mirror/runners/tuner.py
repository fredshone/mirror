import copy
import datetime
from pathlib import Path

import optuna
import polars as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch import argmax, concat, stack
from torch.random import seed as seeder
from torch.utils.data import DataLoader

from mirror import cuda_available
from mirror.dataloaders.loader import DataModule
from mirror.encoders import (
    TableEncoder,
    YXDataset,
    YZDataset,
    ZDataset,
)
from mirror.eval.density import mean_mean_absolute_error
from mirror.models.cvae import CVAE
from mirror.models.cvae_components import (
    CVAEDecoderBlock,
    CVAEEncoderBlock,
    LabelsEncoderBlock,
)


def tune_command(
    config: dict,
    db_path: str = None,
    verbose: bool = False,
    gen: bool = True,
    test: bool = False,
    infer=True,
) -> None:
    """
    Tune the hyperparameters of the model using optuna.

    Args:
        config (dict): The configuration dictionary.
        db_path (str, optional): The path to the optuna database. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        gen (bool, optional): Whether to generate synthetic data. Defaults to True.
        test (bool, optional): Whether to test the model. Defaults to False.
        infer (bool, optional): Whether to infer the model. Defaults to True.

    Returns:
        None

    """
    name = str(
        config.get("logging", {}).get(
            "name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )
    log_dir = Path(config.get("logging", {}).get("dir", "logs")) / name
    tune_dir = log_dir / "tune"

    # create directories
    log_dir.mkdir(exist_ok=True, parents=True)
    tune_dir.mkdir(exist_ok=True, parents=True)

    seed = config.pop("seed", seeder())
    torch.manual_seed(seed)
    verbose = config.get("verbose", False)

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

    # encode input data
    x_encoder = TableEncoder(x, verbose=verbose)
    x_dataset = x_encoder.encode(data=x)
    y_encoder = TableEncoder(y, verbose=verbose)
    y_dataset = y_encoder.encode(data=y)

    xy_dataset = YXDataset(y_dataset, x_dataset)
    datamodule = DataModule(
        dataset=xy_dataset,
        **config.get("datamodule", {})
    )

    trials = config.get("tune", {}).get("trials", 20)
    prune = config.get("tune", {}).get("prune", True)
    timeout = config.get("tune", {}).get("timeout", 600)

    if cuda_available():
        torch.set_float32_matmul_precision("medium")
    # torch.cuda.empty_cache()

    def objective(trial: optuna.Trial) -> float:
        trial_config = build_config(trial, config)
        trial_name = build_trial_name(trial.number)
        logger = WandbLogger(dir=tune_dir, name=trial_name)

        # build model
        model_params = trial_config.get("model_params", {})
        
        # encoder block to embed labels into vec with hidden size
        labels_encoder_block = LabelsEncoderBlock(
            encoder_types=y_encoder.types(),
            encoder_sizes=y_encoder.sizes(),
            depth=model_params.get("controls_encoder", {}).get("depth", 2),
            hidden_size=model_params.get("controls_encoder", {}).get("hidden_size", 32),
        )

        # encoder and decoder block to process census data
        encoder = CVAEEncoderBlock(
            encoder_types=x_encoder.types(),
            encoder_sizes=x_encoder.sizes(),
            depth=model_params.get("encoder", {}).get("depth", 2),
            hidden_size=model_params.get("encoder", {}).get("hidden_size", 32),
            latent_size=model_params.get("latent_size", 16),
        )
        decoder = CVAEDecoderBlock(
            encoder_types=x_encoder.types(),
            encoder_sizes=x_encoder.sizes(),
            depth=model_params.get("decoder", {}).get("depth", 2),
            hidden_size=model_params.get("decoder", {}).get("hidden_size", 32),
            latent_size=model_params.get("latent_size", 16),
        )

        # CVAE model
        model = CVAE(
            embedding_names=x_encoder.names(),
            embedding_types=x_encoder.types(),
            embedding_weights=x_encoder.weights(),
            labels_encoder_block=labels_encoder_block,
            encoder_block=encoder,
            decoder_block=decoder,
            beta=model_params.get("beta", 0.1),
            lr=model_params.get("lr", 0.001),
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=trial_config.get("early_stopping", {})["patience"],
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

        trainer = Trainer(
            callbacks=callbacks,
            logger=logger,
            **trial_config.get("trainer", {}),
        )

        trainer.logger.log_hyperparams(trial.params)
        trial.set_user_attr("config", trial_config)

        trainer.fit(model=model, train_dataloaders=datamodule)

        n = len(xy_dataset)
        z_loader = ZDataset(n, latent_size=latent)
        yz_loader = YZDataset(z_loader, y_dataset)
        gen_loader = DataLoader(
            yz_loader, **trial_config.get("gen_dataloader", {})
        )

        ys, xs, zs = zip(*trainer.predict(dataloaders=gen_loader))
        ys = concat(ys)
        # todo: currently using argmax to decode categorical variables
        xs = concat([stack([argmax(x, dim=1) for x in xb], dim=-1) for xb in xs], dim=0)
        zs = concat(zs)

        y_synth = y_encoder.decode(ys)
        x_synth = x_encoder.decode(xs).drop("pid")
        synth = pl.concat([y_synth, x_synth], how="horizontal")

        mmae_first = mean_mean_absolute_error(target=x, synthetic=synth, order=1)
        mmae_second = mean_mean_absolute_error(target=x, synthetic=synth, order=2)
        mmae = (mmae_first + mmae_second) / 2.0

        return mmae

    if prune:
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    if db_path is not None:
        db_name = f"sqlite:///{db_path}"
    else:
        db_name = f"sqlite:///{log_dir}/optuna.db"

    study = optuna.create_study(
        storage=db_name,
        study_name=name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(
        objective, n_trials=trials, timeout=timeout, callbacks=[best_callback]
    )

    best_trial = study.best_trial
    print("Best params:", best_trial.params)
    print("=============================================")

    # config = study.user_attrs["config"]
    # config["logging_params"]["log_dir"] = log_dir
    # config["logging_params"]["name"] = "best_trial"

    # runners.run_command(
    #     config, verbose=verbose, gen=gen, test=test, infer=infer
    # )
    # print("=============================================")
    # print(f"Best ({best_trial.value}) params: {best_trial.params}")


def best_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr("config", trial.user_attrs["config"])


def build_config(trial: optuna.Trial, config: dict) -> dict:
    """Iterate through the config leaves and parse the values"""
    new_config = copy.deepcopy(config)
    new_config = build_suggestions(trial, new_config)
    return new_config


def build_trial_name(number: int) -> str:
    return str(number).zfill(4)


def skey(key: str) -> str:
    ks = key.split("_")
    if len(ks) > 1:
        return "".join([k[0].upper() for k in ks])
    length = len(key)
    if length > 3:
        return key[:4]
    return key


def build_suggestions(trial: optuna.Trial, config: dict):
    for k, v in config.copy().items():
        if isinstance(v, dict):
            config[k] = build_suggestions(trial, v)
        else:
            found, suggestion = parse_suggestion(trial, v)
            if found:
                config.pop(k)
                config[k] = suggestion
    return config


def parse_suggestion(trial, value: str):
    """Execute the value and return the suggested value.
    Or return Nones if not a suggestion.
    """
    if isinstance(value, str) and value.startswith("trial.suggest"):
        return True, eval(value)
    else:
        return False, None