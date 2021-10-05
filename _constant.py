from autosklearn.pipeline.components.feature_preprocessing.no_preprocessing import NoPreprocessing
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

_feature_mol : dict = {
    'no_preprocessing' : NoPreprocessing,
    'densifier' : Densifier, 
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