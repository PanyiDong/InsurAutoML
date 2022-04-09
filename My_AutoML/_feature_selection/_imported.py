"""
File: _imported.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_feature_selection/_imported.py
File Created: Tuesday, 5th April 2022 11:38:17 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 8th April 2022 10:21:05 pm
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

import warnings
import numpy as np
from functools import partial

######################################################################################################################
# Modified Feature Selection from autosklearn


class Densifier:

    """
    from autosklearn.pipeline.components.feature_preprocessing.densifier import Densifier

    Parameters
    ----------
    seed: random seed, default = 1
    """

    def __init__(self, seed=1):
        self.seed = seed
        self.preprocessor = None

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        from scipy import sparse

        if sparse.issparse(X):
            return X.todense().getA()
        else:
            return X


class ExtraTreesPreprocessorClassification:

    """
    from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
    using sklearn.ensemble.ExtraTreesClassifier

    Parameters
    ----------
    n_estimators: Number of trees in forest, default = 100

    criterion: Function to measure the quality of a split, default = 'gini'
    supported ("gini", "entropy")

    min_samples_leaf: Minimum number of samples required to be at a leaf node, default = 1

    min_samples_split: Minimum number of samples required to split a node, default = 2

    max_features: Number of features to consider, default = 'auto'
    supported ("auto", "sqrt", "log2")

    bootstrap: Whether bootstrap samples, default = False

    max_leaf_nodes: Maximum number of leaf nodes accepted, default = None

    max_depth: Maximum depth of the tree, default = None

    min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights, default = 0.0

    min_impurity_decrease: Threshold to split if this split induces a decrease of the impurity, default = 0.0

    oob_score: Whether to use out-of-bag samples, default = False

    n_jobs: Parallel jobs to run, default = 1

    verbose: Controls the verbosity, default = 0

    class weight: Weights associated with classes, default = None
    supported ("balanced", "balanced_subsample"), dict or list of dicts

    seed: random seed, default = 1
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="auto",
        bootstrap=False,
        max_leaf_nodes=None,
        max_depth=None,
        min_weight_fraction_leaf=0.0,
        min_impurity_decrease=0.0,
        oob_score=False,
        n_jobs=1,
        verbose=0,
        class_weight=None,
        seed=1,
    ):

        self.n_estimators = n_estimators
        self.estimator_increment = 10
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.class_weight = class_weight
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y, sample_weight=None):

        import sklearn.ensemble
        import sklearn.feature_selection

        self.n_estimators = int(self.n_estimators)
        self.max_leaf_nodes = (
            None if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
        )
        self.max_depth = None if self.max_depth is None else int(self.max_depth)

        self.bootstrap = True if self.bootstrap is True else False
        self.n_jobs = int(self.n_jobs)
        self.min_impurity_decrease = float(self.min_impurity_decrease)
        self.max_features = self.max_features
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_samples_split = int(self.min_samples_split)
        self.verbose = int(self.verbose)
        max_features = int(X.shape[1] ** float(self.max_features))

        estimator = sklearn.ensemble.ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            max_features=max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.seed,
            class_weight=self.class_weight,
        )
        estimator.fit(X, y, sample_weight=sample_weight)

        self.preprocessor = sklearn.feature_selection.SelectFromModel(
            estimator=estimator, threshold="mean", prefit=True
        )

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError

        return self.preprocessor.transform(X)


class ExtraTreesPreprocessorRegression:

    """
    from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_regression import ExtraTreesPreprocessorRegression
    using sklearn.ensemble.ExtraTreesRegressor

    Parameters
    ----------
    n_estimators: Number of trees in forest, default = 100

    criterion: Function to measure the quality of a split, default = 'squared_error'
    supported ("squared_error", "mse", "absolute_error", "mae")

    min_samples_leaf: Minimum number of samples required to be at a leaf node, default = 1

    min_samples_split: Minimum number of samples required to split a node, default = 2

    max_features: Number of features to consider, default = 'auto'
    supported ("auto", "sqrt", "log2")

    bootstrap: Whether bootstrap samples, default = False

    max_leaf_nodes: Maximum number of leaf nodes accepted, default = None

    max_depth: Maximum depth of the tree, default = None

    min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights, default = 0.0

    oob_score: Whether to use out-of-bag samples, default = False

    n_jobs: Parallel jobs to run, default = 1

    verbose: Controls the verbosity, default = 0

    seed: random seed, default = 1
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="auto",
        bootstrap=False,
        max_leaf_nodes=None,
        max_depth=None,
        min_weight_fraction_leaf=0.0,
        oob_score=False,
        n_jobs=1,
        verbose=0,
        seed=1,
    ):

        self.n_estimators = n_estimators
        self.estimator_increment = 10
        if criterion not in ("mse", "friedman_mse", "mae"):
            raise ValueError(
                "'criterion' is not in ('mse', 'friedman_mse', "
                "'mae'): %s" % criterion
            )
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y):

        import sklearn.ensemble
        import sklearn.feature_selection

        self.n_estimators = int(self.n_estimators)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_samples_split = int(self.min_samples_split)
        self.max_features = float(self.max_features)
        self.bootstrap = True if self.bootstrap is True else False
        self.n_jobs = int(self.n_jobs)
        self.verbose = int(self.verbose)

        self.max_leaf_nodes = (
            None if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
        )
        self.max_depth = None if self.max_depth is None else int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

        num_features = X.shape[1]
        max_features = int(float(self.max_features) * (np.log(num_features) + 1))

        # Use at most half of the features
        max_features = max(1, min(int(X.shape[1] / 2), max_features))
        estimator = sklearn.ensemble.ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            max_features=max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            random_state=self.seed,
        )

        estimator.fit(X, y)
        self.preprocessor = sklearn.feature_selection.SelectFromModel(
            estimator=estimator, threshold="mean", prefit=True
        )

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError

        return self.preprocessor.transform(X)


class FastICA:

    """
    from autosklearn.pipeline.components.feature_preprocessing.fast_ica import FastICA
    using

    Parameters
    ----------
    algorithm: Apply parallel or deflational algorithm, default = 'parallel'
    supported ('parallel', 'deflation')

    whiten: If false, no whitening is performed, default = True

    fun: Functional form of the G function used, default = 'logcosh'
    supported ('logcosh', 'exp', 'cube') or callable

    n_components: Number of components to retain, default = None

    seed: random seed, default = 1
    """

    def __init__(
        self,
        algorithm="parallel",
        whiten=True,
        fun="logcosh",
        n_components=None,
        seed=1,
    ):
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.n_components = n_components
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y=None):

        import sklearn.decomposition

        self.n_components = (
            None if self.n_components is None else int(self.n_components)
        )

        self.preprocessor = sklearn.decomposition.FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            fun=self.fun,
            whiten=self.whiten,
            random_state=self.seed,
        )

        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="array must not contain infs or NaNs"
            )
            try:
                self.preprocessor.fit(X)
            except ValueError as e:
                if "array must not contain infs or NaNs" in e.args[0]:
                    raise ValueError(
                        "Bug in scikit-learn: "
                        "https://github.com/scikit-learn/scikit-learn/pull/2738"
                    )

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)


class FeatureAgglomeration:

    """
    from autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration import FeatureAgglomeration
    using

    Parameters
    ----------
    n_clusters: Number of clusters, default = 2

    affinity: Metric used to compute the linkage, default = 'euclidean'
    supported ("euclidean", "l1", "l2", "manhattan", "cosine", or 'precomputed')

    linkage: Linkage criterion, default = 'ward'
    supported ("ward", "complete", "average", "single")

    pooling_func: Combines the values of agglomerated features into a single value, default = np.mean

    seed: random seed, default = 1
    """

    def __init__(
        self,
        n_clusters=2,
        affinity="euclidean",
        linkage="ward",
        pooling_func=np.mean,
        seed=1,
    ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.seed = seed

        self.pooling_func_mapping = dict(mean=np.mean, median=np.median, max=np.max)

        self.preprocessor = None

    def fit(self, X, y=None):

        import sklearn.cluster

        self.n_clusters = int(self.n_clusters)

        n_clusters = min(self.n_clusters, X.shape[1])

        if not callable(self.pooling_func):
            self.pooling_func = self.pooling_func_mapping[self.pooling_func]

        self.preprocessor = sklearn.cluster.FeatureAgglomeration(
            n_clusters=n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            pooling_func=self.pooling_func,
        )
        self.preprocessor.fit(X)

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)


class KernelPCA:

    """
    from autosklearn.pipeline.components.feature_preprocessing.kernel_pca import KernelPCA
    using sklearn.decomposition.KernelPCA

    Parameters
    ----------
    n_components: number of features to retain, default = None

    kernel: kernel used, default = 'linear'
    supported (linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed')

    degree: Degree for poly kernels, default = 3

    gamma: Kernel coefficient, default = 0.25

    coef0: Independent term in poly and sigmoid kernels, default = 0.0

    seed: random seed, default = 1
    """

    def __init__(
        self,
        n_components=None,
        kernel="linear",
        degree=3,
        gamma=0.25,
        coef0=0.0,
        seed=1,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y=None):

        import sklearn.decomposition

        self.n_components = (
            None if self.n_components is None else int(self.n_components)
        )
        self.degree = int(self.degree)
        self.gamma = float(self.gamma)
        self.coef0 = float(self.coef0)

        self.preprocessor = sklearn.decomposition.KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            remove_zero_eig=True,
            random_state=self.seed,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            self.preprocessor.fit(X)

        if len(self.preprocessor.alphas_ / self.preprocessor.lambdas_) == 0:
            raise ValueError("All features removed.")

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            _X = self.preprocessor.transform(X)

            if _X.shape[1] == 0:
                raise ValueError("KernelPCA removed all features!")

            return _X


class RandomKitchenSinks:

    """
    from autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks import RandomKitchenSinks
    using sklearn.kernel_approximation.RBFSampler

    Parameters
    ----------
    gamma: use to determine standard variance of random weight table, default = 1
    Parameter of RBF kernel: exp(-gamma * x^2).

    n_components: number of samples per original feature, default = 100

    seed: random seed, default = 1
    """

    def __init__(self, gamma=1.0, n_components=100, seed=1):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y=None):

        import sklearn.kernel_approximation

        self.n_components = int(self.n_components)
        self.gamma = float(self.gamma)

        self.preprocessor = sklearn.kernel_approximation.RBFSampler(
            self.gamma, self.n_components, self.seed
        )
        self.preprocessor.fit(X)

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)


class LibLinear_Preprocessor:

    """
    from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor
    using import sklearn.svm, from sklearn.feature_selection import SelectFromModel

    Parameters
    ----------
    penalty: Norm used in the penalization, default = 'l2'
    supported ('l1', 'l2')

    loss: Loss function, default = 'squared_hinge'
    supported ('hinge', 'squared_hinge')

    dual: Whether to solve the dual or primal, default = True

    tol: Stopping criteria, default = 1e-4

    C: Regularization parameter, default = 1.0

    multi_class: Multi-class strategy, default = 'ovr'
    supported ('ovr', 'crammer_singer')

    fit_intercept: Whether to calculate the intercept, default = True

    intercept_scaling: Intercept scaling rate, default = 1

    class_weight: Class weight, default = None
    supported dict or 'balanced'

    seed: random seed, default = 1
    """

    def __init__(
        self,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        tol=1e-4,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        seed=1,
    ):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y, sample_weight=None):

        import sklearn.svm
        from sklearn.feature_selection import SelectFromModel

        self.C = float(self.C)
        self.tol = float(self.tol)
        self.intercept_scaling = float(self.intercept_scaling)

        estimator = sklearn.svm.LinearSVC(
            penalty=self.penalty,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            class_weight=self.class_weight,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            multi_class=self.multi_class,
            random_state=self.seed,
        )
        estimator.fit(X, y)

        self.preprocessor = SelectFromModel(
            estimator=estimator, threshold="mean", prefit=True
        )

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)


class Nystroem:

    """
    from autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler import Nystroem
    using sklearn.kernel_approximation.Nystroem

    Parameters
    ----------
    kernel: Kernel map to be approximated, default = 'rbf'
    supported: ('additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian',
    'sigmoid', 'cosine' )

    n_components: Number of features to retain, default = 100

    gamma: Gamma parameter, default = 1.0

    degree: Degree of the polynomial kernel, default = 3

    coef0: Zero coefficient for polynomial and sigmoid kernels, default = 1

    seed: random seed, default = 1
    """

    def __init__(
        self, kernel="rbf", n_components=100, gamma=1.0, degree=3, coef0=1, seed=1
    ):
        self.kernel = kernel
        self.n_components = (n_components,)
        self.gamma = (gamma,)
        self.degree = (degree,)
        self.coef0 = (coef0,)
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y=None):

        import sklearn.kernel_approximation

        self.n_components = int(self.n_components)
        self.gamma = float(self.gamma)
        self.degree = int(self.degree)
        self.coef0 = float(self.coef0)

        if self.kernel == "chi2":
            X[X < 0] = 0.0

        self.preprocessor = sklearn.kernel_approximation.Nystroem(
            kernel=self.kernel,
            n_components=self.n_components,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            random_state=self.seed,
        )
        self.preprocessor.fit(X)

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        if self.kernel == "chi2":
            X[X < 0] = 0.0

        return self.preprocessor.transform(X)


class PCA:

    """
    from autosklearn.pipeline.components.feature_preprocessing.pca import PCA
    using sklearn.decomposition.PCA

    Parameters
    ----------
    n_components: numer of features to retain, default = None
    all features will be retained

    whiten: default = False
    if True, the `components_` vectors will be modified to ensure uncorrelated outputs

    seed: random seed, default = 1
    """

    def __init__(self, n_components=None, whiten=False, seed=1):
        self.n_components = n_components
        self.whiten = whiten
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y=None):

        import sklearn.decomposition

        self.n_components = (
            None if self.n_components is None else int(self.n_components)
        )

        self.preprocessor = sklearn.decomposition.PCA(
            n_components=self.n_components, whiten=self.whiten, copy=True
        )
        self.preprocessor.fit(X)

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)


class PolynomialFeatures:

    """
    from autosklearn.pipeline.components.feature_preprocessing.polynomial import PolynomialFeatures
    using sklearn.preprocessing.PolynomialFeatures

    Parameters
    ----------
    degree: degree of polynomial features, default = 2

    interaction_only: if to only to conclude interaction terms, default = False

    include_bias: if to conclude bias term, default = True

    seed: random seed, default = 1
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True, seed=1):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y):

        import sklearn.preprocessing

        self.degree = int(self.degree)

        self.preprocessor = sklearn.preprocessing.PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        self.preprocessor.fit(X, y)

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)


class RandomTreesEmbedding:

    """
    from autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding import RandomTreesEmbedding
    using sklearn.ensemble.RandomTreesEmbedding

    Parameters
    ----------
    n_estimators: Number of trees in the forest to train, deafult = 100

    max_depth: Maximum depth of the tree, default = 5

    min_samples_split: Minimum number of samples required to split a node, default = 2

    min_samples_leaf: Minimum number of samples required to be at a leaf node, default = 1

    min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights, default = 0.

    max_leaf_nodes: Maximum number of leaf nodes, deafult = None

    bootstrap: Mark if bootstrap, default = False

    sparse_output: If output as sparse format (with False), default = False
    True for dense output

    n_jobs: Number of jobs run in parallel, default = 1

    seed: random seed, default = 1
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        bootstrap=False,
        sparse_output=False,
        n_jobs=1,
        seed=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y=None):

        import sklearn.ensemble

        self.n_estimators = int(self.n_estimators)
        self.max_depth = int(self.max_depth)
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.bootstrap = True if self.bootstrap is None else False

        self.preprocessor = sklearn.ensemble.RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            sparse_output=self.sparse_output,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )

        self.preprocessor.fit(X)

        return self

    def transform(self, X):

        if self.preprocessor == None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)


"""
from autosklearn.pipeline.components.feature_preprocessing.select_percentile import SelectPercentileBase
using sklearn.feature_selection.SelectPercentile
"""


class SelectPercentileClassification:

    """
    from autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification import SelectPercentileClassification
    using sklearn.feature_selection.SelectPercentile

    Parameters
    ----------
    percentile: Percent of features to keep, default = 10

    score_func: default = 'chi2'
    supported mode ('chi2', 'f_classif', 'mutual_info_classif')

    seed: random seed, default = 1
    """

    def __init__(self, percentile=10, score_func="chi2", seed=1):
        self.percentile = int(float(percentile))
        self.seed = seed
        import sklearn.feature_selection

        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info_classif":
            self.score_func = partial(
                sklearn.feature_selection.mutual_info_classif, random_state=self.seed
            )
        else:
            raise ValueError(
                'Not recognizing score_func, supported ("chi2", "f_classif", "mutual_info_classif", \
                get {})'.format(
                    score_func
                )
            )

        self.preprocessor = None

    def fit(self, X, y):

        import sklearn.feature_selection

        if self.score_func == sklearn.feature_selection.chi2:
            X[X < 0] = 0

        self.preprocessor = sklearn.feature_selection.SelectPercentile(
            score_func=self.score_func, percentile=self.percentile
        )
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X):

        import sklearn.feature_selection

        if self.preprocessor is None:
            raise NotImplementedError()

        if self.score_func == sklearn.feature_selection.chi2:
            X[X < 0] = 0

        _X = self.preprocessor.transform(X)

        if _X.shape[1] == 0:
            raise ValueError("All features removed.")

        return _X


class SelectPercentileRegression:

    """
    from autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression import SelectPercentileRegression
    using sklearn.feature_selection.SelectPercentile

    Parameters
    ----------
    percentile: Percent of features to keep, default = 10

    score_func: default = 'f_regression'
    supported mode ('f_regression', 'mutual_info_regression')

    seed: random seed, default = 1
    """

    def __init__(self, percentile=10, score_func="f_regression", seed=1):
        self.percentile = int(float(percentile))
        self.seed = seed
        import sklearn.feature_selection

        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info_regression":
            self.score_func = partial(
                sklearn.feature_selection.mutual_info_regression, random_state=self.seed
            )
            self.mode = "percentile"
        else:
            raise ValueError(
                'Not recognizing score_func, only support ("f_regression", "mutual_info_regression"), \
                get {}'.format(
                    score_func
                )
            )

        self.preprocessor = None

    def fit(self, X, y):

        import sklearn.feature_selection

        self.preprocessor = sklearn.feature_selection.SelectPercentile(
            score_func=self.score_func, percentile=self.percentile
        )
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X):

        import sklearn.feature_selection

        if self.preprocessor is None:
            raise NotImplementedError()

        _X = self.preprocessor.transform(X)

        if _X.shape[1] == 0:
            warnings.warn("All features removed.")

        return _X


class SelectClassificationRates:

    """
    from autosklearn.pipeline.components.feature_preprocessing.select_rates_classification import SelectClassificationRates
    using sklearn.feature_selection.GenericUnivariateSelect

    Parameters
    ----------
    alpha: parameter of corresponding mode, default = 1e-5

    mode: Feature selection mode, default = 'fpr'
    supported mode ('percentile', 'k_best', 'fpr', 'fdr', 'fwe')

    score_func: default = 'chi2'
    supported mode ('chi2', 'f_classif', 'mutual_info_classif')

    seed: random seed, default = 1
    """

    def __init__(self, alpha=1e-5, mode="fpr", score_func="chi2", seed=1):
        self.alpha = alpha
        self.mode = mode
        self.seed = seed
        import sklearn.feature_selection

        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info_classif":
            self.score_func = partial(
                sklearn.feature_selection.mutual_info_classif, random_state=self.seed
            )
            self.mode = "percentile"
        else:
            raise ValueError(
                'Not recognizing score_func, supported ("chi2", "f_classif", "mutual_info_classif", \
                get {})'.format(
                    score_func
                )
            )

        self.preprocessor = None

    def fit(self, X, y):
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

        if self.score_func == sklearn.feature_selection.chi2:
            X[X < 0] = 0

        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
            score_func=self.score_func, param=self.alpha, mode=self.mode
        )
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X):

        import sklearn.feature_selection

        if self.score_func == sklearn.feature_selection.chi2:
            X[X < 0] = 0

        if self.preprocessor is None:
            raise NotImplementedError()

        _X = self.preprocessor.transform(X)

        if _X.shape[1] == 0:
            warnings.warn("All features removed.")

        return _X


class SelectRegressionRates:

    """
    from autosklearn.pipeline.components.feature_preprocessing.select_rates_regression import SelectRegressionRates
    using sklearn.feature_selection.GenericUnivariateSelect

    Parameters
    ----------
    alpha: parameter of corresponding mode, default = 1e-5

    mode: Feature selection mode, default = 'percentile'
    supported mode ('percentile', 'k_best', 'fpr', 'fdr', 'fwe')

    score_func: default = 'f_regression'
    supported mode ('f_regression', 'mutual_info_regression')

    seed: random seed, default = 1
    """

    def __init__(
        self, alpha=1e-5, mode="percentile", score_func="f_regression", seed=1
    ):
        self.alpha = alpha
        self.mode = mode
        self.seed = seed
        import sklearn.feature_selection

        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info_regression":
            self.score_func = partial(
                sklearn.feature_selection.mutual_info_regression, random_state=self.seed
            )
            self.mode = "percentile"
        else:
            raise ValueError(
                'Not recognizing score_func, only support ("f_regression", "mutual_info_regression"), \
                get {}'.format(
                    score_func
                )
            )

        self.preprocessor = None

    def fit(self, X, y):

        import sklearn.feature_selection

        alpha = float(self.alpha)
        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
            score_func=self.score_func, param=alpha, mode=self.mode
        )

        self.preprocessor.fit(X, y)

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        _X = self.preprocessor.transform(X)

        if _X.shape[1] == 0:
            warnings.warn("All features removed.")

        return _X


class TruncatedSVD:

    """
    from autosklearn.pipeline.components.feature_preprocessing.truncatedSVD import TruncatedSVD
    Truncated SVD using sklearn.decomposition.TruncatedSVD

    Parameters
    ----------
    n_components: Number of features to retain, default = None
    will be set to p - 1, and capped at p -1 for any input

    seed: random seed, default = 1
    """

    def __init__(self, n_components=None, seed=1):
        self.n_components = n_components
        self.seed = seed
        self.preprocessor = None

    def fit(self, X, y):

        if self.n_components == None:
            self.n_components = X.shape[1] - 1
        else:
            self.n_components = int(self.n_components)
        n_components = min(self.n_components, X.shape[1] - 1)  # cap n_components

        from sklearn.decomposition import TruncatedSVD

        self.preprocessor = TruncatedSVD(
            n_components, algorithm="randomized", random_state=self.seed
        )
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)


######################################################################################################################
