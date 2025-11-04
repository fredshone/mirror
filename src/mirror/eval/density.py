from itertools import combinations
from typing import Iterator, Tuple, Callable, Optional

import polars as pl


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


def compute_joint_frequencies(
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
    assert set(target.columns) == set(synthetic.columns)
    for cols in combinations(target.columns, order):
        name = "_".join(cols)
        yield (
            name,
            frequencies(target, list(cols), "target"),
            frequencies(synthetic, list(cols), "synthetic"),
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
    assert joined["target"].sum() == joined["synthetic"].sum()
    joined = joined.select(
        pl.col("target") / pl.sum("target"),
        pl.col("synthetic") / pl.sum("synthetic"),
    )
    return joined


def absolute_errors(target: pl.Series, synthetic: pl.Series) -> pl.Series:
    return (target - synthetic).abs()


def calc_mae(joined: pl.DataFrame) -> float:
    return (joined["target"] - joined["synthetic"]).abs().mean()


def calc_mnae(joined: pl.DataFrame) -> float:
    abs = (joined["target"] - joined["synthetic"]).abs()
    sum = joined["target"] + joined["synthetic"]
    return (abs / sum).mean()


# def evaluate_density(
#         target: pl.DataFrame,
#         synthetic: pl.DataFrame,
#         order: int = 1,
#         metric: Optional[Callable[[pl.DataFrame], float]] = None
# )


# results = {}
# for order in range(1, 2):
#     print(f"Order {order}:")
#     maes = {}
#     for name, target, synthetic in compute_joint_frequencies(
#         census, df, order=order
#     ):
#         join = join_probs(target, synthetic)
#         maes[name] = calc_mae(join)
#     results[order] = maes
