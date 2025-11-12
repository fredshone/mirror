"""Console script for caveat."""

from typing import Optional

import click
import yaml

from mirror.runners.tuner import tune_command


@click.version_option(package_name="mirror")
@click.group()
def cli():
    """Console script for mirror."""
    pass


@cli.command(name="tune")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--db-path", "-db", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--no-infer", "-ni", is_flag=True)
@click.option("--no-gen", "-ng", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def tune(
    config_path: click.Path,
    db_path: Optional[click.Path],
    test: bool,
    no_gen: bool,
    no_infer: bool,
    verbose: bool,
):
    """Train and report on an encoder and model as per the given configuration file."""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        tune_command(
            config,
            db_path=db_path,
            verbose=verbose,
            test=test,
            gen=not no_gen,
            infer=not no_infer,
        )
