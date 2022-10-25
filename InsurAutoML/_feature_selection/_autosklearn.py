"""
File: _autosklearn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: \My_AutoML\_feature_selection\_autosklearn.py
File Created: Friday, 15th April 2022 2:55:10 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 10th May 2022 11:45:12 pm
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

import autosklearn.pipeline.components.feature_preprocessing as askfs

########################################################################################
# wrap autosklearn feature selection methods
# can be initialized without specifying hyperparameters


class densifier(askfs.densifier.Densifier):
    def __init__(self):
        super().__init__()

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class extra_trees_preproc_for_classification(
    askfs.extra_trees_preproc_for_classification.ExtraTreesPreprocessorClassification
):
    def __init__(
        self,
        n_estimators=5,
        criterion="entropy",
        min_samples_leaf=5,
        min_samples_split=5,
        max_features=0.5,
        bootstrap=False,
        max_leaf_nodes=None,
        max_depth=None,
        min_weight_fraction_leaf=0.0,
        min_impurity_decrease=0.0,
    ):
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

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class extra_trees_preproc_for_regression(
    askfs.extra_trees_preproc_for_regression.ExtraTreesPreprocessorRegression
):
    def __init__(
        self,
        n_estimators=5,
        criterion="mse",
        min_samples_leaf=5,
        min_samples_split=5,
        max_features=0.5,
        bootstrap=False,
        max_leaf_nodes=None,
        max_depth=None,
        min_weight_fraction_leaf=0.0,
    ):
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

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class fast_ica(askfs.fast_ica.FastICA):
    def __init__(
        self,
        algorithm="parallel",
        whiten=False,
        fun="logcosh",
        n_components=5,
    ):
        super().__init__(
            algorithm=algorithm,
            whiten=whiten,
            fun=fun,
            n_components=n_components,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class feature_agglomeration(askfs.feature_agglomeration.FeatureAgglomeration):
    def __init__(
        self,
        n_clusters=5,
        affinity="euclidean",
        linkage="ward",
        pooling_func="mean",
    ):
        super().__init__(
            n_clusters=n_clusters,
            affinity=affinity,
            linkage=linkage,
            pooling_func=pooling_func,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class kernel_pca(askfs.kernel_pca.KernelPCA):
    def __init__(
        self,
        n_components=5,
        kernel="rbf",
        gamma=0.1,
        degree=3,
        coef0=0.5,
    ):
        super().__init__(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class kitchen_sinks(askfs.kitchen_sinks.RandomKitchenSinks):
    def __init__(
        self,
        gamma=0.1,
        n_components=50,
    ):
        super().__init__(
            gamma=gamma,
            n_components=n_components,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class liblinear_svc_preprocessor(
    askfs.liblinear_svc_preprocessor.LibLinear_Preprocessor
):
    def __init__(
        self,
        penalty="l1",
        loss="squared_hinge",
        dual=False,
        tol=0.0001,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
    ):
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

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class nystroem_sampler(askfs.nystroem_sampler.Nystroem):
    def __init__(
        self,
        kernel="rbf",
        n_components=50,
        gamma=0.1,
        degree=3,
        coef0=0.5,
    ):
        super().__init__(
            kernel=kernel,
            n_components=n_components,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class pca(askfs.pca.PCA):
    def __init__(
        self,
        keep_variance=0.5,
        whiten=True,
    ) -> None:
        super().__init__(
            keep_variance=keep_variance,
            whiten=whiten,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class polynomial(askfs.polynomial.PolynomialFeatures):
    def __init__(
        self,
        degree=3,
        interaction_only=False,
        include_bias=True,
    ):
        super().__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class random_trees_embedding(askfs.random_trees_embedding.RandomTreesEmbedding):
    def __init__(
        self,
        n_estimators=5,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=5,
        min_weight_fraction_leaf=1.0,
        max_leaf_nodes=None,
        bootstrap=True,
    ):
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

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class select_percentile_classification(
    askfs.select_percentile_classification.SelectPercentileClassification
):
    def __init__(
        self,
        percentile=90,
        score_func="chi2",
    ):
        super().__init__(
            percentile=percentile,
            score_func=score_func,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class select_percentile_regression(
    askfs.select_percentile_regression.SelectPercentileRegression
):
    def __init__(
        self,
        percentile=90,
        score_func="f_regression",
    ):
        super().__init__(
            percentile=percentile,
            score_func=score_func,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class select_rates_classification(
    askfs.select_rates_classification.SelectClassificationRates
):
    def __init__(
        self,
        alpha=0.3,
        score_func="chi2",
        mode="fpr",
    ):
        super().__init__(
            alpha=alpha,
            score_func="mutual_info_classif"
            if score_func == "mutual_info"
            else score_func,
            # deal with sklearn/autosklearn incosistency
            mode=mode,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class select_rates_regression(askfs.select_rates_regression.SelectRegressionRates):
    def __init__(
        self,
        alpha=0.3,
        score_func="f_regression",
        mode="fpr",
    ):
        super().__init__(
            alpha=alpha,
            score_func=score_func,
            mode=mode,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)


class truncatedSVD(askfs.truncatedSVD.TruncatedSVD):
    def __init__(
        self,
        target_dim=5,
    ):
        super().__init__(
            target_dim=target_dim,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        return super().transform(X)
