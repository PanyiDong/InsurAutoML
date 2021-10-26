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
from ._feature_selection import PCA_FeatureSelection, LDASelection, FeatureFilter, \
    RBFSampler, ASFFS, GeneticAlgorithm
from ._feature_selection import Densifier, ExtraTreesPreprocessorClassification, ExtraTreesPreprocessorRegression, \
    FastICA, FeatureAgglomeration, KernelPCA, RandomKitchenSinks, LibLinear_Preprocessor, Nystroem, PCA, \
    PolynomialFeatures, RandomTreesEmbedding, SelectPercentileClassification, SelectPercentileRegression, \
    SelectClassificationRates, SelectRegressionRates, TruncatedSVD
from ._model_selection import AutoClassifier

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
    'Winsorization',
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
    'LDASelection', # _feature_selection
    'PCA_FeatureSelection',
    'RBFSampler',
    'FeatureFilter',
    'ASFFS',
    'GeneticAlgorithm',
    'densifier',  # from autosklearn
    'extra_trees_preproc_for_classification', 
    'extra_trees_preproc_for_regression', 
    'fast_ica', 
    'feature_agglomeration',
    'kernel_pca', 
    'kitchen_sinks', 
    'liblinear_svc_preprocessor', 
    'nystroem_sampler', 
    'pca', 
    'polynomial', 
    'random_trees_embedding',
#    'select_percentile', 
    'select_percentile_classification', 
    'select_percentile_regression', 
    'select_rates_classification',
    'select_rates_regression', 
    'truncatedSVD',
    'AutoClassifier'
]

base = {'load_data' : load_data}

encoders = {'DataEncoding' : DataEncoding}

imputers = {
    'SimpleImputer' : SimpleImputer,
#    'DummyImputer' : DummyImputer,
    'JointImputer' : JointImputer,
    'ExpectationMaximization' : ExpectationMaximization,
    'KNNImputer' : KNNImputer,
    'MissForestImputer' : MissForestImputer,
    'MICE' : MICE,
    'GAIN' : GAIN
}
# Markov Chain Monte Carlo (MCMC)

scalings = {
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
    'LDASelection' : LDASelection,
    'PCA_FeatureSelection' : PCA_FeatureSelection,
    'RBFSampler' : RBFSampler,
    'FeatureFilter' : FeatureFilter,
    'ASFFS' : ASFFS,
    'GeneticAlgorithm' : GeneticAlgorithm,
#    'densifier' : Densifier,  # from autosklearn
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
#    'select_percentile' : SelectPercentileBase, 
    'select_percentile_classification' : SelectPercentileClassification, 
    'select_percentile_regression' : SelectPercentileRegression, 
    'select_rates_classification' : SelectClassificationRates,
    'select_rates_regression' : SelectRegressionRates, 
    'truncatedSVD' : TruncatedSVD
}

model_selection = {
    'AutoClassifier' : AutoClassifier
}