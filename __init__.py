from ._base import no_processing, load_data
from ._utils import (
    random_guess,
    random_index,
    random_list,
    is_date,
    feature_rounding,
    train_test_split,
    minloc,
    maxloc,
    True_index,
    nan_cov,
    class_means,
    empirical_covariance,
    class_cov,
    Pearson_Corr,
    MI,
    t_score,
    ANOVA,
)
from ._imputation import (
    SimpleImputer,
    DummyImputer,
    JointImputer,
    ExpectationMaximization,
    KNNImputer,
    MissForestImputer,
    MICE,
    GAIN,
)
from ._encoding import DataEncoding
from ._scaling import (
    MinMaxScale,
    Standardize,
    Normalize,
    RobustScale,
    PowerTransformer,
    QuantileTransformer,
    Winsorization,
)
from ._balancing import (
    SimpleRandomOverSampling,
    SimpleRandomUnderSampling,
    TomekLink,
    EditedNearestNeighbor,
    CondensedNearestNeighbor,
    OneSidedSelection,
    CNN_TomekLink,
    Smote,
    Smote_TomekLink,
    Smote_ENN,
)
from ._feature_selection import (
    PCA_FeatureSelection,
    LDASelection,
    FeatureFilter,
    RBFSampler,
    ASFFS,
    GeneticAlgorithm,
)
from ._feature_selection import (
    Densifier,
    ExtraTreesPreprocessorClassification,
    ExtraTreesPreprocessorRegression,
    FastICA,
    FeatureAgglomeration,
    KernelPCA,
    RandomKitchenSinks,
    LibLinear_Preprocessor,
    Nystroem,
    PCA,
    PolynomialFeatures,
    RandomTreesEmbedding,
    SelectPercentileClassification,
    SelectPercentileRegression,
    SelectClassificationRates,
    SelectRegressionRates,
    TruncatedSVD,
)

from ._model_selection import AutoML, AutoClassifier, AutoRegressor

__all__ = [
    "load_data",  # _base
    "no_processing",
    "SimpleImputer",  # _missing
    "DummyImputer",
    "JointImputer",
    "ExpectationMaximization",
    "KNNImputer",
    "MissForestImputer",
    "MICE",
    "GAIN",
    "DataEncoding",  # _encoding
    "MinMaxScale",  # _scaling
    "Standardize",
    "Normalize",
    "RobustScale",
    "PowerTransformer",
    "QuantileTransformer",
    "Winsorization",
    "SimpleRandomOverSampling",  # _imbalance
    "SimpleRandomUnderSampling",
    "TomekLink",
    "EditedNearestNeighbor",
    "CondensedNearestNeighbor",
    "OneSidedSelection",
    "CNN_TomekLink",
    "Smote",
    "Smote_TomekLink",
    "Smote_ENN",
    "LDASelection",  # _feature_selection
    "PCA_FeatureSelection",
    "RBFSampler",
    "FeatureFilter",
    "ASFFS",
    "GeneticAlgorithm",
    "densifier",  # from autosklearn
    "extra_trees_preproc_for_classification",
    "extra_trees_preproc_for_regression",
    "fast_ica",
    "feature_agglomeration",
    "kernel_pca",
    "kitchen_sinks",
    "liblinear_svc_preprocessor",
    "nystroem_sampler",
    "pca",
    "polynomial",
    "random_trees_embedding",
    #    'select_percentile',
    "select_percentile_classification",
    "select_percentile_regression",
    "select_rates_classification",
    "select_rates_regression",
    "truncatedSVD",
    "AutoClassifier",
]

base = {"load_data": load_data}

encoders = {"DataEncoding": DataEncoding}

imputers = {
    "SimpleImputer": SimpleImputer,
    #    'DummyImputer' : DummyImputer,
    "JointImputer": JointImputer,
    "ExpectationMaximization": ExpectationMaximization,
    "KNNImputer": KNNImputer,
    "MissForestImputer": MissForestImputer,
    "MICE": MICE,
    "GAIN": GAIN,
}
# Markov Chain Monte Carlo (MCMC)

scalings = {
    "no_processing": no_processing,
    "MinMaxScale": MinMaxScale,
    "Standardize": Standardize,
    "Normalize": Normalize,
    "RobustScale": RobustScale,
    "PowerTransformer": PowerTransformer,
    "QuantileTransformer": QuantileTransformer,
    "Winsorization": Winsorization,
}

balancing = {
    "no_processing": no_processing,
    "SimpleRandomOverSampling": SimpleRandomOverSampling,
    "SimpleRandomUnderSampling": SimpleRandomUnderSampling,
    "TomekLink": TomekLink,
    "EditedNearestNeighbor": EditedNearestNeighbor,
    "CondensedNearestNeighbor": CondensedNearestNeighbor,
    "OneSidedSelection": OneSidedSelection,
    "CNN_TomekLink": CNN_TomekLink,
    "Smote": Smote,
    "Smote_TomekLink": Smote_TomekLink,
    "Smote_ENN": Smote_ENN,
}

import autosklearn
import autosklearn.pipeline.components.feature_preprocessing

feature_selection = {
    "no_processing": no_processing,
    # "LDASelection": LDASelection,
    # "PCA_FeatureSelection": PCA_FeatureSelection,
    "RBFSampler": RBFSampler,
    "FeatureFilter": FeatureFilter,
    "ASFFS": ASFFS,
    "GeneticAlgorithm": GeneticAlgorithm,
    # 'densifier' : autosklearn.pipeline.components.feature_preprocessing.densifier.Densifier,  # from autosklearn
    "extra_trees_preproc_for_classification": autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification.ExtraTreesPreprocessorClassification,
    "extra_trees_preproc_for_regression": autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_regression.ExtraTreesPreprocessorRegression,
    "fast_ica": autosklearn.pipeline.components.feature_preprocessing.fast_ica.FastICA,
    "feature_agglomeration": autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration.FeatureAgglomeration,
    "kernel_pca": autosklearn.pipeline.components.feature_preprocessing.kernel_pca.KernelPCA,
    # "kitchen_sinks": autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks.RandomKitchenSinks,
    "liblinear_svc_preprocessor": autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor.LibLinear_Preprocessor,
    "nystroem_sampler": autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler.Nystroem,
    "pca": autosklearn.pipeline.components.feature_preprocessing.pca.PCA,
    "polynomial": autosklearn.pipeline.components.feature_preprocessing.polynomial.PolynomialFeatures,
    "random_trees_embedding": autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding.RandomTreesEmbedding,
    # 'select_percentile' : autosklearn.pipeline.components.feature_preprocessing.select_percentile.SelectPercentileBase,
    "select_percentile_classification": autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification.SelectPercentileClassification,
    "select_percentile_regression": autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression.SelectPercentileRegression,
    "select_rates_classification": autosklearn.pipeline.components.feature_preprocessing.select_rates_classification.SelectClassificationRates,
    "select_rates_regression": autosklearn.pipeline.components.feature_preprocessing.select_rates_regression.SelectRegressionRates,
    "truncatedSVD": autosklearn.pipeline.components.feature_preprocessing.truncatedSVD.TruncatedSVD,
}

model_selection = {
    "AutoML" : AutoML,
    "AutoClassifier": AutoClassifier, 
    "AutoRegressor": AutoRegressor
}