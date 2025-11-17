import copy
import datetime
from pathlib import Path

import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.random import seed as seeder

from mirror.dataloaders.loader import DataModule
from mirror.encoders import (
    TableEncoder,
    XYDataset,
)
from mirror.eval.density import mean_mean_absolute_error
from mirror.runners import helpers


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

    trials = config.get("tune", {}).get("trials", 20)
    prune = config.get("tune", {}).get("prune", True)
    timeout = config.get("tune", {}).get("timeout", 600)

    def objective(trial: optuna.Trial) -> float:
        trial_config = build_config(trial, config)
        trial_name = build_trial_name(trial.number)
        logger = WandbLogger(dir=tune_dir, name=trial_name)

        model = helpers.build_cvae(config=trial_config, x_encoder=x_encoder, y_encoder=y_encoder)
        
        callbacks = helpers.build_callbacks(trial_config)

        trainer = Trainer(
            callbacks=callbacks,
            logger=logger,
            **trial_config.get("trainer", {}),
        )

        trainer.logger.log_hyperparams(trial.params)
        trial.set_user_attr("config", trial_config)

        trainer.fit(model=model, train_dataloaders=datamodule)

        gen_loader = helpers.build_gen_loader(config, y_dataset=y_dataset)
        
        xs, ys, zs = helpers.generate(config, dataloader=gen_loader, trainer=trainer)

        # y_synth = y_encoder.decode(ys)
        x_synth = x_encoder.decode(xs).drop("pid")
        # synth = pl.concat([y_synth, x_synth], how="horizontal")

        mmae_first = mean_mean_absolute_error(target=x, synthetic=x_synth, order=1)
        mmae_second = mean_mean_absolute_error(target=x, synthetic=x_synth, order=2)
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