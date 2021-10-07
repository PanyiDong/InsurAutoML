from ._base import random_guess, random_index, random_list, train_test_split, minloc, \
    is_date, no_processing, load_data
from ._missing import SimpleImputer, DummyImputer, JointImputer, ExpectationMaximization, \
    KNNImputer, MissForestImputer, MICE, GAIN
from ._encoding import DataEncoding
from ._scaling import MinMaxScale, Standardize, Normalize, RobustScale, PowerTransformer, \
    QuantileTransformer, Winsorization
from ._imbalance import SimpleRandomOverSampling, SimpleRandomUnderSampling, TomekLink, \
    EditedNearestNeighbor, CondensedNearestNeighbor, OneSidedSelection, CNN_TomekLink, \
    Smote, Smote_TomekLink, Smote_ENN
from ._feature_selection import PCA_FeatureSelection, RBFSampler

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
    'RBFSampler' : RBFSampler
}