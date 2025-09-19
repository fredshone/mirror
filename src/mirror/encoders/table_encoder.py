from typing import List, Optional, Union

import pandas as pd
import pandas.api.types as ptypes
import polars as pl
from torch import Tensor, stack
from torch.utils.data import Dataset

from mirror.encoders.base import CategoricalTokeniser, ContinuousEncoder


class CensusDataset(Dataset):
    def __init__(self, data: Tensor):
        self.data = data

    def __repr__(self):
        return f"{super().__repr__()}: {self.data.shape}"

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TableEncoder:

    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame],
        include: Optional[list] = None,
        exclude: Optional[list] = None,
        verbose: bool = False,
    ):
        """Tokenise a dataframe into a Tensor,
        and initialise mapping for further encoding and decoding.
        Args:
            data (Union[pl.DataFrame, pd.DataFrame]): input dataframe to tokenise.
            include (list, optional): columns to include. Defaults to None.
            exclude (list, optional): columns to exclude. Defaults to None.
        """
        
        self.verbose = verbose
        columns = data.columns
        columns = [col for col in columns if col not in ["pid", "iid", "hid"]]
        if include is not None:
            columns = [col for col in columns if col in include]
        if exclude is not None:
            columns = [col for col in columns if col not in exclude]

        if not columns:
            raise UserWarning("No columns found to encode in table.")

        self.columns = columns

        self.mode = type(data)
        if isinstance(data, pd.DataFrame):
            self.configure_pandas(data, verbose=verbose)
        elif isinstance(data, pl.DataFrame):
            self.configure_polars(data, verbose=verbose)
        else:
            raise ValueError("Data must be a pandas or polars dataframe")

    def configure_polars(self, data: pl.DataFrame, verbose: bool = False) -> None:
        """Configure the tokeniser by encoding the dataframe columns.
        Args:
            data (pl.DataFrame): input dataframe to configure.
            verbose (bool, optional): print the configuration. Defaults to False.
        """

        self.encoders = {}

        for column in self.columns:
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in attributes")
            values = data[column]
            dtype = values.dtype
            if (
                dtype == pl.Utf8
                or dtype == pl.Object
                or dtype == pl.Categorical
                or dtype == pl.Boolean
                or dtype == pl.Enum
            ):
                self.encoders[column] = CategoricalTokeniser(
                    data[column], column, verbose=verbose
                )
            elif -8 in values or values.n_unique() < 25:
                self.encoders[column] = CategoricalTokeniser(
                    data[column], column, verbose=verbose
                )
            elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                self.encoders[column] = ContinuousEncoder(
                    data[column], column, verbose=verbose
                )
            else:
                raise UserWarning(
                    f"Column '{column}' not supported for encoding: {values.dtype}"
                )

    def configure_pandas(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Configure the tokeniser by encoding the dataframe columns.
        Args:
            data (pd.DataFrame): input dataframe to configure.
            verbose (bool, optional): print the configuration. Defaults to False.
        """
        self.encoders = {}
        for column in self.columns:
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in attributes")
            values = data[column]
            if (
                ptypes.is_string_dtype(values)
                or ptypes.is_object_dtype(values)
                or ptypes.is_categorical_dtype(values)
            ):
                self.encoders[column] = CategoricalTokeniser(
                    data[column], column, verbose=verbose
                )
            elif -8 in values or values.nunique() < 25:
                self.encoders[column] = CategoricalTokeniser(
                    data[column], column, verbose=verbose
                )
            elif ptypes.is_numeric_dtype(values):
                self.encoders[column] = ContinuousEncoder(
                    data[column], column, verbose=verbose
                )
            else:
                raise UserWarning(
                    f"Column '{column}' not supported for encoding: {values.dtype}"
                )

    def encode(self, data: Union[pl.DataFrame, pd.DataFrame]) -> Tensor:
        """Encode the dataframe into a Tensor.
        Args:
            data (Union[pl.DataFrame, pd.DataFrame]): input dataframe to encode.
        Returns:
            Tensor: encoded dataframe.
        """
        encoded = []
        for column, encoder in self.encoders.items():
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in data")
            column_encoded = encoder.encode(data[column])
            encoded.append(column_encoded)

        if not encoded:
            raise UserWarning("No encodings found.")

        encoded = stack(encoded, dim=-1).float()
        dataset = CensusDataset(encoded)
        if self.verbose:
            print(f"{self} encoded -> {dataset}")
        return dataset  # todo: weights

    def encode_series(self, data: pd.Series) -> Tensor:
        """Encode a pandas series into a 1d Tensor.
        Args:
            data (pd.Series): input series to encode.
        Returns:
            Tensor: encoded series.
        """
        if data.name not in self.encoders.keys():
            raise UserWarning(f"'{data.name}' not found in available encoders")
        encoder = self.encoders[data.name]
        column_encoded = encoder.encode(data)
        return column_encoded

    def names(self) -> List[str]:
        """Get the names of the encoders.
        Returns:
            List[str]: list of encoder names.
        """
        return list(self.encoders.keys())

    def types(self) -> List[str]:
        """Get the types of the embeddings.
        Returns:
            List[str]: list of types of the embeddings.
        """
        return [encoder.encoding for encoder in self.encoders.values()]

    def sizes(self) -> List[int]:
        """Get the sizes of the embeddings.
        Returns:
            List[int]: list of sizes of the embeddings.
        """
        return [encoder.size for encoder in self.encoders.values()]
    
    def weights(self) -> List[List[Optional[float]]]:
        """Get the weights of the embeddings.
        Returns:
            List[List[Optional[float]]]: list of weights of the embeddings.
        """
        return [encoder.get_weights() for encoder in self.encoders.values()]

    def decode(self, data: List[Tensor]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Decode Tensor of tokens back into dataframe.

        Args:
            data (List[Tensor]): input Tensor of tokens to decode.

        Returns:
            Union[pd.DataFrame, pl.DataFrame]: decoded dataframe.
        """
        decoded = {"pid": list(range(data.shape[0]))}
        for i, (name, encoder) in enumerate(self.encoders.items()):
            tokens = data[:, i]
            decoded[name] = encoder.decode(tokens)
        decoded = pd.DataFrame(decoded) if self.mode == pd.DataFrame else pl.DataFrame(decoded)
        return decoded

    def argmax_decode(self, data: List[Tensor]) -> Union[pd.DataFrame, pl.DataFrame]:
        argmaxed = [d.argmax(dim=-1) for d in data]
        return self.decode(argmaxed)

    def multinomial_decode(self, data: List[Tensor]) -> Union[pd.DataFrame, pl.DataFrame]:
        sampled = [d.multinomial(1).squeeze() for d in data]
        return self.decode(sampled)
