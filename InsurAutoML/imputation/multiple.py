"""
File Name: multiple.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/imputation/multiple.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 23rd December 2025 10:35:51 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2025, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import Union, List
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from ..constant import UNI_CLASS
from ..utils import random_index, random_list
from .base import SimpleImputer, BaseImputer


class ExpectationMaximization(BaseImputer):
    """
    Use Expectation Maximization (EM) to impute missing data[1]

    [1] Impyute.imputation.cs.em

    Parameters
    ----------
    iterations: maximum number of iterations for single imputation, default = 50

    threshold: threshold to early stop iterations, default = 0.01
    only early stop when iterations < self.iterations and change in the imputation < self.threshold

    seed: random seed, default = 1
    """

    def __init__(
        self, iterations: int = 50, threshold: float = 0.01, seed: int = None
    ) -> None:
        self.iterations = iterations
        self.threshold = threshold
        self.seed = seed

        super().__init__()
        self._fitted = False  # whether the imputer has been fitted

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:
        self.iterations = int(self.iterations)
        self.threshold = float(self.threshold)

        _X = X.copy(deep=True)
        n = _X.shape[0]

        if _X.isnull().values.any():
            _X = self._fill(_X)

        self._fitted = True

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:
        features = list(X.columns)
        np.random.seed(self.seed)

        self._missing_table = pd.DataFrame()  # dataframe to mark the missing index
        # create _missing_table with _missing_feature
        # missing index will be 1, existed index will be 0
        for _column in features:
            if X[_column].isnull().values.any():
                self._missing_table[_column] = X[_column].isnull().astype(int)

        for _column in list(self._missing_table.columns):
            for _index in np.where(self._missing_table[_column] == 1)[0]:
                X.loc[_index, _column] = self._EM_iter(X, _index, _column)

        return X

    def _EM_iter(self, X: pd.DataFrame, index: Union[int, str], column: str):
        _mark = 1
        for _ in range(self.iterations):
            _mu = np.nanmean(X.loc[:, column])
            _std = np.nanstd(X.loc[:, column])
            _tmp = np.random.normal(loc=_mu, scale=_std)
            _delta = np.abs(_tmp - _mark) / _mark
            if _delta < self.threshold and self.iterations > 10:
                return _tmp
            X.loc[index, column] = _tmp
            _mark = _tmp
        return _tmp


class KNNImputer(BaseImputer):
    """
    Use KNN to impute the missing values, further update: use cross validation to select best k [1]

    [1] Stekhoven, D.J. and Bühlmann, P., 2012. MissForest—non-parametric missing value imputation
    for mixed-type data. Bioinformatics, 28(1), pp.112-118.

    Parameters
    ----------
    n_neighbors: list of k, default = None
    default will set to 1:10

    method: method to initaillay impute missing values, default = "mean"

    fold: cross validation number of folds, default = 10

    uni_class: unique class to be considered as categorical columns, default = 31

    seed: random seed, default = 1
    """

    def __init__(
        self,
        n_neighbors: int = None,
        method: str = "mean",
        fold: int = 10,
        uni_class: int = UNI_CLASS,
        seed: int = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.method = method
        self.fold = fold
        self.uni_class = uni_class
        self.seed = seed

        super().__init__()
        self._fitted = False  # whether the imputer has been fitted

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:
        features = list(X.columns)
        for _column in features:
            if len(X[_column].unique()) <= min(0.1 * len(X), self.uni_class):
                raise ValueError("KNN Imputation not supported for categorical data!")

        _X = X.copy(deep=True)
        if _X.isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No nan values found, no change.")

        self._fitted = True

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:
        features = list(X.columns)

        self._missing_feature = []  # features contains missing values
        self._missing_vector = (
            []
        )  # vector with missing values, to mark the missing index
        # create _missing_table with _missing_feature
        # missing index will be 1, existed index will be 0

        for _column in features:
            if X[_column].isnull().values.any():
                self._missing_feature.append(_column)
                self._missing_vector.append(
                    X[_column].loc[X[_column].isnull()].index.astype(int)
                )

        self._missing_vector = np.array(self._missing_vector).T
        self._missing_table = pd.DataFrame(
            self._missing_vector, columns=self._missing_feature
        )

        X = SimpleImputer(method=self.method).fill(
            X
        )  # initial filling for missing values

        random_features = random_list(
            self._missing_feature, self.seed
        )  # the order to regress on missing features
        # _index = random_index(len(X.index))  # random index for cross
        # validation
        _err = []

        # if assigned n_neighbors, use it, otherwise use k-fold cross
        # validation
        if self.n_neighbors is None:
            for i in range(self.fold):
                _test = X.iloc[
                    i * int(len(X.index) / self.fold) : int(len(X.index) / self.fold), :
                ]
                _train = X
                _train.drop(labels=_test.index, axis=0, inplace=True)
                _err.append(self._cross_validation_knn(_train, _test, random_features))

            # mean of cross validation error
            _err = np.mean(np.array(_err), axis=0)
            self.optimial_k = np.array(_err).argmin()[0] + 1  # optimal k

            X = self._knn_impute(X, random_features, self.optimial_k)
        else:
            X = self._knn_impute(X, random_features, self.n_neighbors)

        return X

    def _cross_validation_knn(
        self, _train: pd.DataFrame, _test: pd.DataFrame, random_features: List[str]
    ) -> List[Union[float, np.ndarray]]:  # cross validation to return error
        from sklearn.neighbors import KNeighborsRegressor

        if self.n_neighbors is None:
            n_neighbors = [i + 1 for i in range(10)]
        else:
            n_neighbors = (
                self.n_neighbors
                if isinstance(self.n_neighbors, list)
                else [self.n_neighbors]
            )

        _test_mark = _test.copy(deep=True)
        _err = []

        for _k in n_neighbors:
            _test = _test_mark.copy(deep=True)
            for _feature in random_features:
                _subfeatures = list(_train.columns)
                _subfeatures.remove(_feature)

                fit_model = KNeighborsRegressor(n_neighbors=_k)
                fit_model.fit(_train.loc[:, _subfeatures], _train.loc[:, _feature])
                _test.loc[:, _feature] = fit_model.predict(_test.loc[:, _subfeatures])
            _err.append(((_test - _test_mark) ** 2).sum())

        return _err

    def _knn_impute(
        self, X: pd.DataFrame, random_features: List[str], k: int
    ) -> pd.DataFrame:
        from sklearn.neighbors import KNeighborsRegressor

        features = list(X.columns)
        for _column in random_features:
            _subfeature = features.copy()
            _subfeature.remove(_column)
            X.loc[self._missing_table[_column], _column] = np.nan
            fit_model = KNeighborsRegressor(n_neighbors=k)
            fit_model.fit(
                X.loc[~X[_column].isnull(), _subfeature],
                X.loc[~X[_column].isnull(), _column],
            )
            X.loc[X[_column].isnull(), _column] = fit_model.predict(
                X.loc[X[_column].isnull(), _subfeature]
            )

        return X


class MissForestImputer(BaseImputer):
    """
    MissForest imputer for mixed data types (numerical and categorical).

    Parameters
    ----------
    max_iter : int, default=10
        Maximum number of imputation iterations.
    n_estimators : int, default=100
        Number of trees in the random forest models.
    random_state : int, default=None
        Random seed for reproducibility.
    categorical_threshold : int, default=UNI_CLASS
        Maximum number of unique values for a feature to be treated as categorical.
    """

    def __init__(
        self,
        max_iter: int = 10,
        n_estimators: int = 100,
        random_state: int = None,
        categorical_threshold: int = UNI_CLASS,
    ) -> None:

        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.categorical_threshold = categorical_threshold
        self._fitted = False
        self.feature_types_ = {}

        super().__init__()
        self._fitted = False  # whether the imputer has been fitted

    def _identify_feature_types(self, X: pd.DataFrame) -> None:
        """Identify numerical and categorical features."""
        numerical_features = []
        categorical_features = []

        for column in X.columns:
            if X[column].isna().all():
                continue

            non_null = X[column].dropna()
            if len(non_null) == 0:
                continue

            n_unique = non_null.nunique()

            if (
                non_null.dtype == "object"
                or non_null.dtype.name == "category"
                or n_unique <= self.categorical_threshold
            ):
                categorical_features.append(column)
            else:
                numerical_features.append(column)

        self.feature_types_ = {
            "numerical": numerical_features,
            "categorical": categorical_features,
        }

    def _initial_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform initial simple imputation."""
        X_imputed = X.copy()

        # Numerical
        if self.feature_types_["numerical"]:
            num_data = X_imputed[self.feature_types_["numerical"]]
            if num_data.isna().any().any():
                num_imputer = SimpleImputer(method="mean")
                X_imputed[self.feature_types_["numerical"]] = num_imputer.fill(num_data)

        # Categorical
        if self.feature_types_["categorical"]:
            cat_data = X_imputed[self.feature_types_["categorical"]]
            if cat_data.isna().any().any():
                cat_imputer = SimpleImputer(method="most frequent")
                X_imputed[self.feature_types_["categorical"]] = cat_imputer.fill(
                    cat_data
                )

        return X_imputed

    def _get_missing_info(self, X: pd.DataFrame):
        """Identify missing features and their missing row indices."""
        missing_features = []
        missing_indices = {}

        for col in X.columns:
            mask = X[col].isna()
            if mask.any():
                missing_features.append(col)
                missing_indices[col] = X[mask].index.tolist()

        missing_features_sorted = sorted(
            missing_features, key=lambda col: X[col].isna().sum()
        )

        return missing_features_sorted, missing_indices

    def _calculate_delta(
        self, X_new: pd.DataFrame, X_old: pd.DataFrame, missing_features: list
    ) -> float:
        """Compute iteration difference for convergence checking."""
        num_error = 0
        cat_error = 0

        for col in missing_features:
            mask = ~X_new[col].isna() & ~X_old[col].isna()

            # Numerical
            if col in self.feature_types_["numerical"]:
                if mask.any():
                    diff = (X_new.loc[mask, col] - X_old.loc[mask, col]) ** 2
                    var = X_new.loc[mask, col].var()
                    if var > 0:
                        num_error += diff.sum() / var

            # Categorical
            else:
                if mask.any():
                    misclassified = (X_new.loc[mask, col] != X_old.loc[mask, col]).sum()
                    cat_error += misclassified / mask.sum()

        return num_error + cat_error

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the MissForest imputer and return the imputed DataFrame."""
        if X.isnull().sum().sum() == 0:
            warnings.warn("No missing values found. Returning the original dataset.")
            self._fitted = True
            return X

        self._identify_feature_types(X)
        missing_features, missing_indices = self._get_missing_info(X)

        X_imputed = self._initial_imputation(X)
        _dtypes = X.dtypes
        prev_error = float("inf")

        self.models_ = {}

        for iter in range(self.max_iter):
            X_old = X_imputed.copy()

            for col in missing_features:
                if col not in missing_indices or not missing_indices[col]:
                    continue

                missing_idx = missing_indices[col]
                other_cols = [c for c in X.columns if c != col]

                # Model selection
                if col in self.feature_types_["numerical"]:
                    model = RandomForestRegressor(
                        n_estimators=self.n_estimators, random_state=self.random_state
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=self.n_estimators, random_state=self.random_state
                    )

                train_mask = ~X_imputed.index.isin(missing_idx)

                if train_mask.sum() > 0:
                    X_train = X_imputed.loc[train_mask, other_cols]
                    y_train = X_imputed.loc[train_mask, col]

                    try:
                        model.fit(X_train, y_train)
                        self.models_[col + "_iter_" + str(iter)] = model
                        X_missing = X_imputed.loc[missing_idx, other_cols]

                        if not X_missing.empty:
                            predictions = model.predict(X_missing)
                            X_imputed.loc[missing_idx, col] = predictions

                    except Exception as e:
                        warnings.warn(f"Error imputing '{col}': {e}")
                        continue

            # Restore original data types
            X_imputed = X_imputed.astype(_dtypes)

            # Convergence checking
            current_error = self._calculate_delta(X_imputed, X_old, missing_features)

            if current_error > prev_error or current_error < 1e-6:
                break

            prev_error = current_error

        self._fitted = True
        return X_imputed

    def refill(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Refill missing values in new data using the fitted MissForest imputer.

        Parameters
        ----------
        X : pd.DataFrame
            New data with missing values to be imputed.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values imputed.
        """
        if not self._fitted:
            raise RuntimeError("The imputer must be fitted before calling refill.")

        missing_features, missing_indices = self._get_missing_info(X)
        X_imputed = self._initial_imputation(X)
        _dtypes = X.dtypes
        prev_error = float("inf")

        for iter in range(self.max_iter):
            X_old = X_imputed.copy()

            for col in missing_features:
                if col not in missing_indices or not missing_indices[col]:
                    continue

                missing_idx = missing_indices[col]
                other_cols = [c for c in X.columns if c != col]

                model_key = col + "_iter_" + str(iter)
                if model_key not in self.models_:
                    continue

                model = self.models_[model_key]
                X_missing = X_imputed.loc[missing_idx, other_cols]

                if not X_missing.empty:
                    predictions = model.predict(X_missing)
                    X_imputed.loc[missing_idx, col] = predictions

            # Restore original data types
            X_imputed = X_imputed.astype(_dtypes)

            # Convergence checking
            current_error = self._calculate_delta(X_imputed, X_old, missing_features)

            if current_error > prev_error or current_error < 1e-6:
                break

            prev_error = current_error

        return X_imputed


class MICE(BaseImputer):
    """
    Multiple Imputation by chained equations (MICE)
    using single imputation to initialize the imputation step, and iteratively build regression/
    classification model to impute features with missing values [1]

    [1] Azur, M.J., Stuart, E.A., Frangakis, C. and Leaf, P.J., 2011. Multiple imputation by
    chained equations: what is it and how does it work?. International journal of methods in
    psychiatric research, 20(1), pp.40-49.

    Parameters
    ----------
    cycle: how many runs of regression/imputation to build the complete data, default = 10

    method: the method to initially fill nan values, default = 'mean'
    supproted methods ['mean', 'zero', 'median', 'most frequent', constant]
    'mean' : fill columns with nan values using mean of non nan values
    'zero': fill columns with nan values using 0
    'median': fill columns with nan values using median of non nan values
    'most frequent': fill columns with nan values using most frequent of non nan values
    constant: fill columns with nan values using predefined values

    seed: random seed, default = 1
    every random draw from the minority class will increase the random seed by 1
    """

    def __init__(self, cycle: int = 10, method: str = "mean", seed: int = None) -> None:
        self.method = method
        self.cycle = cycle
        self.seed = seed

        super().__init__()
        self._fitted = False  # whether the imputer has been fitted

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:
        self.cycle = int(self.cycle)

        _X = X.copy(deep=True)

        if _X.isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No nan values found, no change.")

        self._fitted = True

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:
        features = list(X.columns)

        for _column in features:
            if (X[_column].dtype == object) or (str(X[_column].dtype) == "category"):
                raise ValueError(
                    "MICE can only handle numerical filling, run encoding first!"
                )

        self._missing_feature = []  # features contains missing values
        self._missing_table = pd.DataFrame()  # dataframe to mark the missing index
        # create _missing_table with _missing_feature
        # missing index will be 1, existed index will be 0
        for _column in features:
            if X[_column].isnull().values.any():
                self._missing_table[_column] = X[_column].isnull().astype(int)
                self._missing_feature.append(_column)

        X = SimpleImputer(method=self.method).fill(
            X
        )  # initial filling for missing values

        random_features = random_list(
            self._missing_feature, self.seed
        )  # the order to regress on missing features

        for _ in range(self.cycle):
            X = self._cycle_impute(X, random_features)

        return X

    def _cycle_impute(
        self, X: pd.DataFrame, random_features: List[str]
    ) -> pd.DataFrame:
        from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV

        features = list(X.columns)

        for _column in random_features:
            _subfeature = features.copy()
            _subfeature.remove(_column)
            X.loc[self._missing_table[_column] == 1, _column] = np.nan
            if len(X[_column].unique()) == 2:
                fit_model = LogisticRegression()
            elif len(features) <= 15:
                fit_model = LinearRegression()
            else:
                fit_model = LassoCV()
            fit_model.fit(
                X.loc[~X[_column].isnull(), _subfeature],
                X.loc[~X[_column].isnull(), _column],
            )
            X.loc[X[_column].isnull(), _column] = fit_model.predict(
                X.loc[X[_column].isnull(), _subfeature]
            )

        return X
