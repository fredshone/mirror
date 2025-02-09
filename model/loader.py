import numpy as np
import pandas as pd
import torch
import pandas.api.types as ptypes
from typing import Optional, List, Dict
from torch.utils.data import Dataset, DataLoader


class CensusEncoder:
    def __init__(
            self,
            data: pd.DataFrame,
            cols: Optional[List[str]],
            data_types: Optional[Dict[str, str]],
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

    def encode(self, data: pd.DataFrame):
        encoded = []
        if self.cols is not None:
            data = data[self.cols]
        self._validate_dtypes

        for i, cnfg in self.config.items():
            c = cnfg["name"]
            encoder_type = cnfg["type"]
            dtype = cnfg["dtype"]
            column = data[c].astype(dtype)

            if encoder_type == "categorical":
                categories = cnfg["encodings"]
                nominals = pd.Categorical(column, categories=categories.keys())
                encoded.append(torch.tensor(nominals.codes).int())

            elif encoder_type == "numeric":
                mini, maxi = cnfg["minmax"]
                numeric = torch.tensor(column.values).unsqueeze(-1)
                numeric -= mini
                numeric /= (maxi - mini)
                encoded.append(numeric.float())

        if not encoded:
            raise UserWarning("No encodings found.")

        return encoded  # todo: weights
        

    def _build_dtypes(self, data: pd.DataFrame, data_types: dict) -> Dict[str, str]:
        if data_types is None:
            data_types = {}
        for c in data.columns:
            if c not in data_types:
                if ptypes.is_numeric_dtype(data[c]):
                    data_types[c] = "numeric"
                elif ptypes.is_string_dtype(data[c]):
                    data_types[c] = "categorical"
                else:
                    raise UserWarning(f"Unrecognised dtype '{data[c].dtype}' at column '{c}'.")
        return data_types
                
    def _validate_dtypes(self, data):
        # check for bad columns (ie too many categories)
        non_numerics = [k for k, v in self.data_types.items() if not v == "numeric"]
        n = len(data)
        for c in non_numerics:
            if len(set(data[c])) == n:
                raise UserWarning(f"Categorical column '{c}' appears to have non-categorical data (too many categories).")

        # check numeric and ordinal
        numerics = [k for k, v in self.data_types.items() if v == "numeric"]
        for c in numerics:
            if not ptypes.is_numeric_dtype(data[c]):
                raise UserWarning(f"Numeric column '{c} does not appear to have numeric type.")

        
    def _configure(self, data: pd.DataFrame) -> dict:
        config = []
        for i, (c, v) in enumerate(self.data_types.items()):
            if c not in data.columns:
                raise UserWarning(f"Data '{c}' not found in columns.")
            if v == "categorical":
                encodings = tokenize(data[c])
                config.append({
                    "name": c,
                    "type": "categorical",
                    "encodings": encodings,
                    "dtype": data[c].dtype,
                })
            elif v == "numeric":
                mini = data[c].min()
                maxi = data[c].max()
                config.append({
                    "name": c,
                    "type": "numeric",
                    "minmax": (mini, maxi),
                    "dtype": data[c].dtype,
                })
            else:
                raise UserWarning(
                    f"Unrecognised encoding in configuration: {v}"
                )
        return config





class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[1:]
        label = row[0]
        return features, label

    def __len__(self):
        return len(self.dataframe)
    

def tokenize(data: pd.Series, encodings: Optional[dict] = None) -> dict:
    nominals = pd.Categorical(data)
    encodings = {e: i for i, e in enumerate(nominals.categories)}
    return encodings