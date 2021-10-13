from ._base import no_processing, load_data
from ._utils import random_guess, random_index, random_list, train_test_split, minloc, \
    is_date
from ._missing import SimpleImputer, DummyImputer, JointImputer, ExpectationMaximization, \
    KNNImputer, MissForestImputer, MICE, GAIN
from ._encoding import DataEncoding
from ._scaling import MinMaxScale, Standardize, Normalize, RobustScale, PowerTransformer, \
    QuantileTransformer, Winsorization
from ._imbalance import SimpleRandomOverSampling, SimpleRandomUnderSampling, TomekLink, \
    EditedNearestNeighbor, CondensedNearestNeighbor, OneSidedSelection, CNN_TomekLink, \
    Smote, Smote_TomekLink, Smote_ENN
from ._feature_selection import PCA_FeatureSelection, RBFSampler, FeatureFilter, \
    ASFFS

from autosklearn.pipeline.components.feature_preprocessing.densifier import Densifier
from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_regression import ExtraTreesPreprocessorRegression
from autosklearn.pipeline.components.feature_preprocessing.fast_ica import FastICA
from autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration import FeatureAgglomeration
from autosklearn.pipeline.components.feature_preprocessing.kernel_pca import KernelPCA
from autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks import RandomKitchenSinks
from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor
from autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler import Nystroem
from autosklearn.pipeline.components.feature_preprocessing.pca import PCA
from autosklearn.pipeline.components.feature_preprocessing.polynomial import PolynomialFeatures
from autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding import RandomTreesEmbedding
from autosklearn.pipeline.components.feature_preprocessing.select_percentile import SelectPercentileBase
from autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification import SelectPercentileClassification
from autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression import SelectPercentileRegression
from autosklearn.pipeline.components.feature_preprocessing.select_rates_classification import SelectClassificationRates
from autosklearn.pipeline.components.feature_preprocessing.select_rates_regression import SelectRegressionRates
from autosklearn.pipeline.components.feature_preprocessing.truncatedSVD import TruncatedSVD

__all__ = [
    'load_data', # _base
    'no_processing',
    'SimpleImputer', # _missing
    'DummyImputer',
    'JointImputer',
    'ExpectationMaximization',
    'KNNImputer',
    'MissForestImputer',
    'MICE',
    'GAIN',
    'DataEncoding', # _encoding
    'MinMaxScale', # _scaling
    'Standardize',
    'Normalize',
    'RobustScale',
    'PowerTransformer',
    'QuantileTransformer',
    'SimpleRandomOverSampling', # _imbalance
    'SimpleRandomUnderSampling', 
    'TomekLink',
    'EditedNearestNeighbor', 
    'CondensedNearestNeighbor', 
    'OneSidedSelection', 
    'CNN_TomekLink', 
    'Smote', 
    'Smote_TomekLink', 
    'Smote_ENN',
    'PCA_FeatureSelection', # _feature_selection
    'RBFSampler' 
]

base = {'load_data' : load_data}

missing = {
    'SimpleImputer' : SimpleImputer,
    'DummyImputer' : DummyImputer,
    'JointImputer' : JointImputer,
    'ExpectationMaximization' : ExpectationMaximization,
    'KNNImputer' : KNNImputer,
    'MissForestImputer' : MissForestImputer,
    'MICE' : MICE,
    'GAIN' : GAIN
}
# Markov Chain Monte Carlo (MCMC)

enoding = {'DataEncoding' : DataEncoding}

scaling = {
    'no_processing' : no_processing,
    'MinMaxScale' : MinMaxScale,
    'Standardize' : Standardize,
    'Normalize' : Normalize,
    'RobustScale' : RobustScale,
    'PowerTransformer' : PowerTransformer,
    'QuantileTransformer' : QuantileTransformer,
    'Winsorization' : Winsorization
}
    
imbalance = {
    'no_processing' : no_processing,
    'SimpleRandomOverSampling' : SimpleRandomOverSampling,
    'SimpleRandomUnderSampling' : SimpleRandomUnderSampling, 
    'TomekLink' : TomekLink,
    'EditedNearestNeighbor' : EditedNearestNeighbor, 
    'CondensedNearestNeighbor' : CondensedNearestNeighbor, 
    'OneSidedSelection' : OneSidedSelection, 
    'CNN_TomekLink' : CNN_TomekLink, 
    'Smote' : Smote, 
    'Smote_TomekLink' : Smote_TomekLink, 
    'Smote_ENN' : Smote_ENN
}

feature_selection = {
    'no_processing' : no_processing,
    'PCA_FeatureSelection' : PCA_FeatureSelection,
    'RBFSampler' : RBFSampler,
    'FeatureFilter' : FeatureFilter,
    'ASFFS' : ASFFS,
    'densifier' : Densifier,  # from autosklearn
    'extra_trees_preproc_for_classification' : ExtraTreesPreprocessorClassification, 
    'extra_trees_preproc_for_regression' : ExtraTreesPreprocessorRegression, 
    'fast_ica' : FastICA, 
    'feature_agglomeration' : FeatureAgglomeration,
    'kernel_pca' : KernelPCA, 
    'kitchen_sinks' : RandomKitchenSinks, 
    'liblinear_svc_preprocessor' : LibLinear_Preprocessor, 
    'nystroem_sampler' : Nystroem, 
    'pca' : PCA, 
    'polynomial' : PolynomialFeatures, 
    'random_trees_embedding' : RandomTreesEmbedding,
    'select_percentile' : SelectPercentileBase, 
    'select_percentile_classification' : SelectPercentileClassification, 
    'select_percentile_regression' : SelectPercentileRegression, 
    'select_rates_classification' : SelectClassificationRates,
    'select_rates_regression' : SelectRegressionRates, 
    'truncatedSVD' : TruncatedSVD
}