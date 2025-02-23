import pandas as pd
import torch
import pandas.api.types as ptypes
from typing import Optional, List, Dict, Union
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class CensusDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data
        print(data.shape)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: CensusDataset,
        val_split: float = 0.1,
        test_split: Optional[float] = None,
        train_batch_size: int = 1024,
        val_batch_size: int = 1024,
        test_batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs,
    ):
        """Torch DataModule.

        Args:
            data (Dataset): Data
            val_split (float, optional): _description_. Defaults to None.
            test_split (Optional[float], optional): _description_. Defaults to 0.1.
            train_batch_size (int, optional): _description_. Defaults to 1024.
            val_batch_size (int, optional): _description_. Defaults to 1024.
            test_batch_size (int, optional): _description_. Defaults to 1024.
            num_workers (int, optional): _description_. Defaults to 0.
            pin_memory (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        self.dataset = dataset
        self.val_split = val_split
        self.test_split = test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mapping = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.test_split is None:
            (self.train_dataset, self.val_dataset) = torch.utils.data.random_split(
                self.dataset, [1 - self.val_split, self.val_split]
            )
            self.test_dataset = self.val_dataset
        else:
            (self.train_dataset, self.val_dataset, self.test_dataset) = (
                torch.utils.data.random_split(
                    self.dataset,
                    [
                        1 - self.val_split - self.test_split,
                        self.val_split,
                        self.test_split,
                    ],
                )
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def test_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )


class CensusEncoder:
    def __init__(
        self,
        data: pd.DataFrame,
        cols: Optional[List[str]] = None,
        data_types: Optional[Dict[str, str]] = None,
        auto: bool = True,
    ):
        self.cols = cols
        if cols is not None:
            data = data[cols]
        if auto:
            self.data_types = self._build_dtypes(data, data_types)
        else:
            self.data_types = data_types

        self._validate_dtypes(data)
        self.config = self._configure(data)

    def encode(self, data: pd.DataFrame, **kwargs: dict) -> DataModule:
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
                encoded.append(torch.tensor(nominals.codes).int())

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
        return DataModule(dataset=dataset, **kwargs)  # todo: weights

    def names(self) -> list:
        return [cnfg["name"] for cnfg in self.config]

    def encodings(self) -> list:
        return [(cnfg["type"], cnfg["encoding"]) for cnfg in self.config]

    def dtypes(self) -> list:
        return [cnfg["dtype"] for cnfg in self.config]

    def len(self) -> int:
        return len(self.config)

    def _build_dtypes(self, data: pd.DataFrame, data_types: dict) -> Dict[str, str]:
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
        non_numerics = [k for k, v in self.data_types.items() if not v == "numeric"]
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
                raise UserWarning(f"Unrecognised encoding in configuration: {v}")
        return config


def tokenize(data: pd.Series, encodings: Optional[dict] = None) -> dict:
    nominals = pd.Categorical(data)
    encodings = {e: i for i, e in enumerate(nominals.categories)}
    return encodings
