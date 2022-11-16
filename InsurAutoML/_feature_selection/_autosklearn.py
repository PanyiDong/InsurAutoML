"""
File Name: _autosklearn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_feature_selection/_autosklearn.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 7:01:09 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2022, Panyi Dong

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

import numpy as np
import pandas as pd
from typing import Union
import autosklearn.pipeline.components.feature_preprocessing as askfs

########################################################################################
# wrap autosklearn feature selection methods
# can be initialized without specifying hyperparameters


class densifier(askfs.densifier.Densifier):
    def __init__(self) -> None:
        super().__init__()

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> densifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class extra_trees_preproc_for_classification(
    askfs.extra_trees_preproc_for_classification.ExtraTreesPreprocessorClassification
):
    def __init__(
        self,
        n_estimators: int = 5,
        criterion: str = "entropy",
        min_samples_leaf: int = 5,
        min_samples_split: int = 5,
        max_features: float = 0.5,
        bootstrap: bool = False,
        max_leaf_nodes: int = None,
        max_depth: int = None,
        min_weight_fraction_leaf: float = 0.0,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=bootstrap,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> extra_trees_preproc_for_classification:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class extra_trees_preproc_for_regression(
    askfs.extra_trees_preproc_for_regression.ExtraTreesPreprocessorRegression
):
    def __init__(
        self,
        n_estimators: int = 5,
        criterion: str = "mse",
        min_samples_leaf: int = 5,
        min_samples_split: int = 5,
        max_features: float = 0.5,
        bootstrap: bool = False,
        max_leaf_nodes: int = None,
        max_depth: int = None,
        min_weight_fraction_leaf: float = 0.0,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=bootstrap,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> extra_trees_preproc_for_regression:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class fast_ica(askfs.fast_ica.FastICA):
    def __init__(
        self,
        algorithm: str = "parallel",
        whiten: bool = False,
        fun: str = "logcosh",
        n_components: int = 5,
    ) -> None:
        super().__init__(
            algorithm=algorithm,
            whiten=whiten,
            fun=fun,
            n_components=n_components,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> fast_ica:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class feature_agglomeration(askfs.feature_agglomeration.FeatureAgglomeration):
    def __init__(
        self,
        n_clusters: int = 5,
        affinity: str = "euclidean",
        linkage: str = "ward",
        pooling_func: str = "mean",
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            affinity=affinity,
            linkage=linkage,
            pooling_func=pooling_func,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> feature_agglomeration:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class kernel_pca(askfs.kernel_pca.KernelPCA):
    def __init__(
        self,
        n_components: int = 5,
        kernel: str = "rbf",
        gamma: float = 0.1,
        degree: int = 3,
        coef0: float = 0.5,
    ) -> None:
        super().__init__(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> kernel_pca:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class kitchen_sinks(askfs.kitchen_sinks.RandomKitchenSinks):
    def __init__(
        self,
        gamma: float = 0.1,
        n_components: int = 50,
    ) -> None:
        super().__init__(
            gamma=gamma,
            n_components=n_components,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> kitchen_sinks:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class liblinear_svc_preprocessor(
    askfs.liblinear_svc_preprocessor.LibLinear_Preprocessor
):
    def __init__(
        self,
        penalty: str = "l1",
        loss: str = "squared_hinge",
        dual: bool = False,
        tol: float = 0.0001,
        C: float = 1.0,
        multi_class: str = "ovr",
        fit_intercept: bool = True,
        intercept_scaling: int = 1,
    ) -> None:
        super().__init__(
            penalty=penalty,
            loss=loss,
            dual=dual,
            tol=tol,
            C=C,
            multi_class=multi_class,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> liblinear_svc_preprocessor:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class nystroem_sampler(askfs.nystroem_sampler.Nystroem):
    def __init__(
        self,
        kernel: str = "rbf",
        n_components: int = 50,
        gamma: float = 0.1,
        degree: int = 3,
        coef0: float = 0.5,
    ) -> None:
        super().__init__(
            kernel=kernel,
            n_components=n_components,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> nystroem_sampler:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class pca(askfs.pca.PCA):
    def __init__(
        self,
        keep_variance: float = 0.5,
        whiten: bool = True,
    ) -> None:
        super().__init__(
            keep_variance=keep_variance,
            whiten=whiten,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> pca:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class polynomial(askfs.polynomial.PolynomialFeatures):
    def __init__(
        self,
        degree: int = 3,
        interaction_only: bool = False,
        include_bias: bool = True,
    ) -> None:
        super().__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> polynomial:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class random_trees_embedding(askfs.random_trees_embedding.RandomTreesEmbedding):
    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 3,
        min_samples_split: int = 5,
        min_samples_leaf: int = 5,
        min_weight_fraction_leaf: float = 1.0,
        max_leaf_nodes: int = None,
        bootstrap: bool = True,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> random_trees_embedding:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class select_percentile_classification(
    askfs.select_percentile_classification.SelectPercentileClassification
):
    def __init__(
        self,
        percentile: int = 90,
        score_func: str = "chi2",
    ) -> None:
        super().__init__(
            percentile=percentile,
            score_func=score_func,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> select_percentile_classification:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class select_percentile_regression(
    askfs.select_percentile_regression.SelectPercentileRegression
):
    def __init__(
        self,
        percentile: int = 90,
        score_func: str = "f_regression",
    ) -> None:
        super().__init__(
            percentile=percentile,
            score_func=score_func,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> select_percentile_regression:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class select_rates_classification(
    askfs.select_rates_classification.SelectClassificationRates
):
    def __init__(
        self,
        alpha: float = 0.3,
        score_func: str = "chi2",
        mode: str = "fpr",
    ) -> None:
        super().__init__(
            alpha=alpha,
            score_func="mutual_info_classif"
            if score_func == "mutual_info"
            else score_func,
            # deal with sklearn/autosklearn incosistency
            mode=mode,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> select_rates_classification:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class select_rates_regression(askfs.select_rates_regression.SelectRegressionRates):
    def __init__(
        self,
        alpha: float = 0.3,
        score_func: str = "f_regression",
        mode: str = "fpr",
    ) -> None:
        super().__init__(
            alpha=alpha,
            score_func=score_func,
            mode=mode,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> select_rates_regression:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)


class truncatedSVD(askfs.truncatedSVD.TruncatedSVD):
    def __init__(
        self,
        target_dim: int = 5,
    ) -> None:
        super().__init__(
            target_dim=target_dim,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> truncatedSVD:

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().transform(X)
