from itertools import combinations
from typing import Iterator, Tuple

import polars as pl


def mean_mean_absolute_error(
    target: pl.DataFrame, synthetic: pl.DataFrame, order = 1) -> float:
    """Compute the Mean Mean Absolute Error (MMAE) between target and synthetic DataFrames.
    """
    maes = []
    for _, _, x, xhat in iter_joint_probs(target, synthetic, order=order):
        mae = calc_mae(x, xhat)
        maes.append(mae)
    mmae = sum(maes) / len(maes)
    return mmae


def frequencies(df: pl.DataFrame, cols: list[str], alias: str) -> pl.DataFrame:
    """Compute frequencies of combinations of columns in a DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame.
        cols (list[str]): List of column names to group by.
        alias (str): Name for the frequency column.

    Returns:
        pl.DataFrame: DataFrame with frequencies of combinations.
    """
    frequencies = df.group_by(cols).agg(pl.len().alias(alias))
    new_index = pl.concat_str(cols, separator="_").alias("index")
    frequencies = frequencies.with_columns(new_index).drop(cols)
    return frequencies


def iter_joint_probs(
    target, synthetic, order=2
) -> Iterator[Tuple[str, pl.DataFrame, pl.DataFrame]]:
    """Compute joint frequencies for combinations of columns in two DataFrames.

    Args:
        target (pl.DataFrame): Target DataFrame.
        synthetic (pl.DataFrame): Synthetic DataFrame.
        order (int, optional): The size of column combinations to consider. Defaults to 2.

    Yields:
        tuple: A tuple containing the name of the combination, the target frequencies,
               and the synthetic frequencies.
    """
    if not set(target.columns) == set(synthetic.columns):
        xor = set(target.columns).symmetric_difference(set(synthetic.columns))
        print(f"Column mismatch: {xor}")
        raise ValueError("Target and synthetic DataFrames must have the same columns.")
    for cols in combinations(target.columns, order):
        name = "_".join(cols)
        target_freq = frequencies(target, list(cols), "target")
        synthetic_freq = frequencies(synthetic, list(cols), "synthetic")
        joined = join_probs(target_freq, synthetic_freq)
        yield (
            name,
            joined["index"],
            joined["target"],
            joined["synthetic"],
        )


def join_probs(target: pl.DataFrame, synthetic: pl.DataFrame) -> pl.DataFrame:
    """Join two DataFrames on their indices and compute probabilities.

    Args:
        target (pl.DataFrame): Target DataFrame with frequency counts.
        synthetic (pl.DataFrame): Synthetic DataFrame with frequency counts.

    Returns:
        pl.DataFrame: DataFrame with normalized probabilities for target and synthetic.
    """
    joined = target.join(
        synthetic, on="index", how="full", coalesce=True
    ).fill_null(0)
    joined = joined.select(
        pl.col("index"),
        pl.col("target") / pl.sum("target"),
        pl.col("synthetic") / pl.sum("synthetic"),
    )
    return joined


def absolute_errors(target: pl.Series, synthetic: pl.Series) -> pl.Series:
    return (target - synthetic).abs()


def calc_mae(target: pl.Series, synthetic: pl.Series) -> float:
    return (target - synthetic).abs().mean()


def calc_mnae(target: pl.Series, synthetic: pl.Series) -> float:
    abs = (target - synthetic).abs()
    sum = target + synthetic
    return (abs / sum).mean()


def calc_cross_entropy(
    target: pl.Series, synthetic: pl.Series, epsilon: float = 1e-10
) -> float:
    synthetic = synthetic + epsilon
    log_synthetic = synthetic.log()
    cross_entropy = -(target * log_synthetic).sum()
    return cross_entropy
