from typing import Dict, List, Optional

import pandas as pd
import pandas.api.types as ptypes
import torch
from torch.utils.data import Dataset


class CensusDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __repr__(self):
        return f"{super().__repr__()}: {self.data.shape}"

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CensusEncoder:
    def __init__(
        self,
        data: pd.DataFrame,
        cols: Optional[List[str]] = None,
        data_types: Optional[Dict[str, str]] = None,
        auto: bool = True,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.cols = cols
        if cols is not None:
            data = data[cols]
        if auto:
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

    def encode(self, data: pd.DataFrame, **kwargs: dict) -> CensusDataset:
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
        dataset = CensusDataset(encoded)
        if self.verbose:
            print(f"{self} encoded -> {dataset}")
        return dataset  # todo: weights

    def names(self) -> list:
        return [cnfg["name"] for cnfg in self.config]

    def encodings(self) -> list:
        return [(cnfg["type"], cnfg["encoding"]) for cnfg in self.config]

    def dtypes(self) -> list:
        return [cnfg["dtype"] for cnfg in self.config]

    def len(self) -> int:
        return len(self.config)

    def _build_dtypes(
        self, data: pd.DataFrame, data_types: dict
    ) -> Dict[str, str]:
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
