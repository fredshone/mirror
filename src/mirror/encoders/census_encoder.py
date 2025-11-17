from typing import Dict, List, Optional

import pandas as pd
import pandas.api.types as ptypes
import torch
from torch.utils.data import Dataset


class XDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        """Dataset for the input features.

        Args:
            data (torch.Tensor): The input features.
        """
        self.data = data

    def __repr__(self):
        return f"{super().__repr__()}: {self.data.shape}"

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class XYDataset(Dataset):
    def __init__(self, x: Dataset, y: Dataset):
        """Dataset for the input features and target labels.

        Args:
            x (Dataset): The input features.
            y (Dataset): The target labels.

        Raises:
            TypeError: If y or x is not a Dataset.
            ValueError: If y and x are not of the same length.
        """
        if not isinstance(y, Dataset) or not isinstance(x, Dataset):
            raise TypeError("y and x must be instances of Dataset")
        if not len(y) == len(x):
            raise ValueError("y and x must be of the same length")
        self.y = y
        self.x = x

    def __repr__(self):
        return f"{super().__repr__()}: {self.x.data.shape}, {self.y.data.shape}"

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class ZDataset(Dataset):
    def __init__(self, num_samples: int, latent_size: int):
        self.z = torch.randn(num_samples, latent_size)

    def __repr__(self):
        return f"{super().__repr__()}: {self.z.shape}"

    def __getitem__(self, index):
        return self.z[index]

    def __len__(self):
        return len(self.z)


class YZDataset(Dataset):
    def __init__(self, z: Dataset, y: Dataset):
        """Dataset for the latent variables and target labels.

        Args:
            z (Dataset): The latent variables.
            y (Dataset): The target labels.

        Raises:
            TypeError: If y or z is not a Dataset.
            ValueError: If y and z are not of the same length.
        """
        if not isinstance(y, Dataset) or not isinstance(z, Dataset):
            raise TypeError("y and z must be instances of Dataset")
        if not len(y) == len(z):
            raise ValueError("y and z must be of the same length")
        self.y = y
        self.z = z

    def __repr__(self):
        return f"{super().__repr__()}: {self.z.data.shape}, {self.y.data.shape}"

    def __getitem__(self, index):
        return self.z[index], self.y[index]

    def __len__(self):
        return len(self.z)


class TableEncoder:
    def __init__(
        self,
        data: pd.DataFrame,
        cols: Optional[List[str]] = None,
        data_types: Optional[Dict[str, str]] = None,
        auto: bool = True,
        verbose: bool = False,
    ):
        """Table encoder for tabular data.

        Args:
            data (pd.DataFrame): The input data.
            cols (Optional[List[str]], optional): The columns to encode. Defaults to None.
            data_types (Optional[Dict[str, str]], optional): The data types for each column. Defaults to None.
            auto (bool, optional): Whether to automatically infer data types. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        self.verbose = verbose
        self.cols = cols
        if cols is not None:
            data = data[cols]
        if auto:
            # infer data types
            self.data_types = self._build_dtypes(data, data_types)
        else:
            self.data_types = data_types

        self._validate_dtypes(data)
        self.config = self._configure(data)
        if self.verbose:
            print(f"{self} initiated census encoder:")
            for name, (etype, encoder), dtype in zip(
                self.names(), self.encodings(), self.dtypes()
            ):
                print(f"\t>{name}: {etype} {dtype}")

    def encode(self, data: pd.DataFrame, **kwargs: dict) -> XDataset:
        encoded = []
        if self.cols is not None:
            data = data[self.cols]
        self._validate_dtypes

        for cnfg in self.config:
            name = cnfg["name"]
            encoder_type = cnfg["type"]
            dtype = cnfg["dtype"]
            column = data[name].astype(dtype)

            if encoder_type == "categorical":
                categories = cnfg["encoding"]
                nominals = pd.Categorical(column, categories=categories.keys())
                encoded.append(torch.tensor(nominals.codes).long())

            elif encoder_type == "numeric":
                mini, maxi = cnfg["encoding"]
                numeric = torch.tensor(column.values).unsqueeze(-1)
                numeric -= mini
                numeric /= maxi - mini
                encoded.append(numeric.float())

        if not encoded:
            raise UserWarning("No encodings found.")

        encoded = torch.stack(encoded, dim=-1)
        dataset = XDataset(encoded)
        if self.verbose:
            print(f"{self} encoded -> {dataset}")
        return dataset  # todo: weights

    def names(self) -> list:
        return [cnfg["name"] for cnfg in self.config]

    def types(self) -> list:
        return [cnfg["type"] for cnfg in self.config]

    def sizes(self) -> list:
        return [cnfg["size"] for cnfg in self.config]

    def encodings(self) -> list:
        return [(cnfg["type"], cnfg["encoding"]) for cnfg in self.config]

    def dtypes(self) -> list:
        return [cnfg["dtype"] for cnfg in self.config]

    def len(self) -> int:
        return len(self.config)

    def _build_dtypes(
        self, data: pd.DataFrame, data_types: Optional[dict] = None
    ) -> Dict[str, str]:
        """
        Build data types for the encoder.

        Args:
            data (pd.DataFrame): The input data.
            data_types (Optional[dict], optional): Predefined data types. Defaults to None.

        Raises:
            UserWarning: Unrecognized data type found.

        Returns:
            Dict[str, str]: A dictionary mapping column names to their data types.
        """
        if data_types is None:
            data_types = {}
        for c in data.columns:
            if c not in data_types:
                if ptypes.is_string_dtype(data[c]):
                    data_types[c] = "categorical"
                elif -8 in data[c]:
                    data_types[c] = "categorical"
                elif len(set(data[c])) < 100:
                    data_types[c] = "categorical"
                elif ptypes.is_numeric_dtype(data[c]):
                    data_types[c] = "numeric"
                else:
                    raise UserWarning(
                        f"Unrecognised dtype '{data[c].dtype}' at column '{c}'."
                    )
        return data_types

    def _validate_dtypes(self, data):
        """Validate the data types of the input DataFrame.

        Args:
            data (pd.DataFrame): The input data.

        Raises:
            UserWarning: Too many categories found.
            UserWarning: Invalid data type found.
        """
        # check for bad columns (ie too many categories)
        non_numerics = [
            k for k, v in self.data_types.items() if not v == "numeric"
        ]
        n = len(data)
        for c in non_numerics:
            if len(set(data[c])) == n:
                raise UserWarning(
                    f"Categorical column '{c}' appears to have non-categorical data (too many categories)."
                )

        # check numeric and ordinal
        numerics = [k for k, v in self.data_types.items() if v == "numeric"]
        for c in numerics:
            if not ptypes.is_numeric_dtype(data[c]):
                raise UserWarning(
                    f"Numeric column '{c} does not appear to have numeric type."
                )

    def _configure(self, data: pd.DataFrame) -> dict:
        """Configure the encoder.

        Args:
            data (pd.DataFrame): The input data.

        Raises:
            UserWarning: Cannot find column.
            UserWarning: Invalid encoding found.

        Returns:
            dict: A dictionary mapping column names to their configurations.
        """
        config = []
        for i, (c, v) in enumerate(self.data_types.items()):
            if c not in data.columns:
                raise UserWarning(f"Data '{c}' not found in columns.")
            if v == "categorical":
                encodings = tokenize(data[c])
                config.append(
                    {
                        "name": c,
                        "type": "categorical",
                        "size": len(encodings),
                        "encoding": encodings,
                        "dtype": data[c].dtype,
                    }
                )
            elif v == "numeric":
                mini = data[c].min()
                maxi = data[c].max()
                config.append(
                    {
                        "name": c,
                        "type": "numeric",
                        "size": 1,
                        "encoding": (mini, maxi),
                        "dtype": data[c].dtype,
                    }
                )
            else:
                raise UserWarning(
                    f"Unrecognised encoding in configuration: {v}"
                )
        return config


def tokenize(data: pd.Series, encodings: Optional[dict] = None) -> dict:
    nominals = pd.Categorical(data)
    encodings = {e: i for i, e in enumerate(nominals.categories)}
    return encodings
