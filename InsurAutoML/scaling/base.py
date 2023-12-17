from __future__ import annotations
from typing import Union
import numpy as np
import pandas as pd


class BaseScaling:
    def __init__(self) -> None:
        pass

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> BaseScaling:
        raise NotImplementedError

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        _X = X.copy(deep=self.deep_copy)

        self.fit(_X, y)

        return self.transform(_X)

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError
