from typing import Optional, Union

import pandas as pd
import polars as pl
from torch import Tensor


def tokenize(data: Union[pd.Series, pl.Series], encoding_map: Optional[dict] = None) -> Tensor:
    """
    Tokenize a Series into a Tensor. If no encoding map is provided,
    a new encoding map will be created. If an encoding map is provided,
    the function will check if the data matches the existing encoding.
    If not, a UserWarning will be raised.
    Args:
        data (Union[pd.Series, pl.Series]): input Series to tokenize.
        encoding_map (dict, optional): existing encodings to use. Defaults to None.
    Returns:
        Tensor: Tensor of encodings and encoding map.
    """
    if encoding_map:
        missing = set(data.unique()) - set(encoding_map.keys())
        if missing:
            raise UserWarning(
                f"""
                Categories in data do not match existing categories.
                {missing} not found.
                Please specify the new categories in the encoding.
                Your existing encodings are: {encoding_map}.
"""
            )
        encoded = data.replace(encoding_map).cast(pl.Int8)
    else:
        cats = list(data.unique().sort())
        encoding_map = {v: i for i, v in enumerate(cats)}
        encoded = data.replace(encoding_map).cast(pl.Int8)
    encoded = Tensor(encoded.to_numpy()).int()
    return encoded, encoding_map
