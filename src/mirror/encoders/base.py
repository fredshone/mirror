from typing import Iterable, Optional

import polars as pl
from torch import Tensor

from mirror.encoders.utils import tokenize


class BaseEncoder:

    def __init__(self, data: Iterable, name: Optional[str] = None):
        raise NotImplementedError(
            "BaseEncoder is an abstract class. Please use a concrete encoder."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def read_polars(self, data: Iterable, name: Optional[str] = None) -> pl.Series:
        if not isinstance(data, pl.Series):
            print(f"Attempting to convert data ({type(data)}) to polars Series.")
            data = pl.Series(data)

        return data
    
    def get_weights(self) -> Optional[Tensor]:
        return None

    def encode(self, data: Iterable) -> Tensor:
        raise NotImplementedError("Encode method not implemented.")

    def decode(self, data: Iterable) -> pl.Series:
        raise NotImplementedError("Decode method not implemented.")


class ContinuousEncoder(BaseEncoder):
    def __init__(
        self, data: Iterable, name: Optional[str] = None, verbose: bool = False
    ):
        """ContinuousEncoder is used to encode continuous data to a range between 0 and 1.

        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
        Raises:
            UserWarning: If the data is not of type int or float.
        """
        if not data.dtype.is_numeric():
            raise UserWarning(
                "ContinuousEncoder only supports numeric data types."
            )
        self.mini = data.min()
        self.maxi = data.max()
        self.range = self.maxi - self.mini
        self.mean = data.mean()
        self.std = data.std()
        if self.range == 0:
            raise UserWarning("Data has no range. Cannot encode.")
        self.dtype = data.dtype

        self.encoding = "continuous"
        self.size = 1

        print(
            f"{self.__class__.__name__}: min: {self.mini}, max: {self.maxi}, range: {self.range}, dtype: {self.dtype}"
        )

    def encode(self, data: Iterable) -> Tensor:
        data = Tensor(data)
        return (data - self.mini) / self.range

    def decode(self, data: Iterable) -> pl.Series:
        data = pl.Series(data)
        new = data * self.range + self.mini
        return new.astype(self.dtype)


class TimeEncoder(BaseEncoder):
    def __init__(
        self,
        data: Iterable,
        name: Optional[str] = None,
        min_value: float = 0,
        cycle: float = 1440,
        verbose: bool = False,
    ):
        """TimeEncoder is used to encode continuous data to a range between 0 and 1.

        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            min_value (float, optional): minimum value of the encoder. Defaults to 0.
            cycle (float, optional): range of the encoder. Defaults to 1440.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
        Raises:
            UserWarning: If the data is not of type int or float.
        """
        if not data.dtype.is_numeric():
            raise UserWarning(
                "TimeEncoder only supports numeric data types."
            )
        if cycle <= 0:
            raise UserWarning("Cycle must be larger than 0.")
        self.mini = min_value
        self.cycle = cycle
        self.mean = data.mean()
        self.std = data.std()
        self.dtype = data.dtype

        self.encoding = "time"
        self.size = 1

        if verbose:
            print(
                f"{self.__class__.__name__}: min: {self.mini}, range: {self.cycle}, dtype: {self.dtype}"
            )

    def encode(self, data: Iterable) -> Tensor:
        data = Tensor(data)
        return (data - self.mini) / self.cycle

    def decode(self, data: Iterable) -> pl.Series:
        data = pl.Series(data)
        new = data * self.cycle + self.mini
        return new.astype(self.dtype)


class CategoricalTokeniser(BaseEncoder):
    def __init__(
        self, data: Iterable, name: Optional[str] = None, verbose: bool = False
    ):
        """CategoricalEncoder is used to encode categorical data as integers from 0 to N.
        Where N is the number of unique categories.
        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
        Raises:
            UserWarning: If the data is not of type int or object.
        """
        self.dtype = data.dtype
        self.encoded, self.mapping = tokenize(data)

        self.encoding = "categorical"
        self.size = len(self.mapping)

        if verbose:
            print(
                f"{self.__class__.__name__}: size: {self.size}, categories: {self.mapping}, dtype: {self.dtype}"
            )
            if self.size > 20:
                print(
                    f">>> Warning: CategoricalEncoder has more than 20 categories ({self.size})). <<<"
                )

    def get_weights(self) -> Tensor:
        """Calculate weights for each category based on their frequency in the data.
        Weights are calculated as the inverse of the frequency, normalized by the mean frequency.
        So that the mean weight is 1.
        
        Returns:
            Tensor: weights for each category."""
        freq = self.encoded.bincount().float()
        freq = freq / freq.mean()
        freq = 1.0 / freq
        return freq

    def encode(self, data: Iterable) -> Tensor:
        return tokenize(data, self.mapping)[0]

    def decode(self, data: Iterable, safe: bool = True) -> pl.Series:
        data = pl.Series(data).cast(pl.Int8)
        reverse_mapping = {v: k for k, v in self.mapping.items()}
        if safe:
            missing = set(data.unique()) - set(reverse_mapping.keys())
            if missing:
                raise UserWarning(
                    f"Missing categories in data: {missing}. Please check your encoding."
                )
        data = data.replace_strict(reverse_mapping, return_dtype=self.dtype)
        return data
