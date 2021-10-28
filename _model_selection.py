import os
import shutil
import warnings
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
import mlflow
import hyperopt
from hyperopt import (
    fmin,
    hp,
    rand,
    tpe,
    atpe,
    Trials,
    SparkTrials,
    space_eval,
    STATUS_OK,
    pyll,
)
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import My_AutoML
from My_AutoML._base import no_processing
from My_AutoML._hyperparameters import (
    encoder_hyperparameter,
    imputer_hyperparameter,
    scaling_hyperparameter,
    balancing_hyperparameter,
    feature_selection_hyperparameter,
    classifiers,
    classifier_hyperparameter,
    regressors,
    regressor_hyperparameters,
)

"""
Classifiers/Hyperparameters from autosklearn:
1. AdaBoost: n_estimators, learning_rate, algorithm, max_depth
2. Bernoulli naive Bayes: alpha, fit_prior
3. Decision Tree: criterion, max_features, max_depth_factor, min_samples_split, 
            min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease
4. Extra Trees: criterion, min_samples_leaf, min_samples_split,  max_features, 
            bootstrap, max_leaf_nodes, max_depth, min_weight_fraction_leaf, min_impurity_decrease
5. Gaussian naive Bayes
6. Gradient boosting: loss, learning_rate, min_samples_leaf, max_depth,
            max_leaf_nodes, max_bins, l2_regularization, early_stop, tol, scoring
7. KNN: n_neighbors, weights, p
8. LDA: shrinkage, tol
9. Linear SVC (LibLinear): penalty, loss, dual, tol, C, multi_class, 
            fit_intercept, intercept_scaling
10. kernel SVC (LibSVM): C, kernel, gamma, shrinking, tol, max_iter
11. MLP (Multilayer Perceptron): hidden_layer_depth, num_nodes_per_layer, activation, alpha,
            learning_rate_init, early_stopping, solver, batch_size, n_iter_no_change, tol,
            shuffle, beta_1, beta_2, epsilon
12. Multinomial naive Bayes: alpha, fit_prior
13. Passive aggressive: C, fit_intercept, tol, loss, average
14. QDA: reg_param
15. Random forest: criterion, max_features, max_depth, min_samples_split, 
            min_samples_leaf, min_weight_fraction_leaf, bootstrap, max_leaf_nodes
16. SGD (Stochastic Gradient Descent): loss, penalty, alpha, fit_intercept, tol,
            learning_rate
"""

# Auto binary classifier
class AutoClassifier:

    """
    Perform model selection and hyperparameter optimization for classification tasks
    using sklearn models, predefine hyperparameters

    Parameters
    ----------
    timeout: Total time limit for the job in seconds, default = 360
    
    max_evals: Maximum number of function evaluations allowd, default = 32

    encoder: Encoders selected for the job, default = 'auto'
    support ('DataEncoding')
    'auto' will select all default encoders, or use a list to select

    imputer: Imputers selected for the job, default = 'auto'
    support ('SimpleImputer', 'JointImputer', 'ExpectationMaximization', 'KNNImputer',
    'MissForestImputer', 'MICE', 'GAIN')
    'auto' will select all default imputers, or use a list to select

    scaling: Scalings selected for the job, default = 'auto'
    support ('no_processing', 'MinMaxScale', 'Standardize', 'Normalize', 'RobustScale',
    'PowerTransformer', 'QuantileTransformer', 'Winsorization')
    'auto' will select all default scalings, or use a list to select

    balancing: Balancings selected for the job, default = 'auto'
    support ('no_processing', 'SimpleRandomOverSampling', 'SimpleRandomUnderSampling',
    'TomekLink', 'EditedNearestNeighbor', 'CondensedNearestNeighbor', 'OneSidedSelection', 
    'CNN_TomekLink', 'Smote', 'Smote_TomekLink', 'Smote_ENN')
    'auto' will select all default balancings, or use a list to select

    feature_selection: Feature selections selected for the job, default = 'auto'
    support ('no_processing', 'LDASelection', 'PCA_FeatureSelection', 'RBFSampler', 
    'FeatureFilter', 'ASFFS', 'GeneticAlgorithm', 'extra_trees_preproc_for_classification',
    'fast_ica', 'feature_agglomeration', 'kernel_pca', 'kitchen_sinks', 
    'liblinear_svc_preprocessor', 'nystroem_sampler', 'pca', 'polynomial', 
    'random_trees_embedding', 'select_percentile_classification','select_rates_classification', 
    'truncatedSVD')
    'auto' will select all default feature selections, or use a list to select
    
    models: Models selected for the job, default = 'auto'
    support ('AdaboostClassifier', 'BernoulliNB', 'DecisionTree', 'ExtraTreesClassifier',
            'GaussianNB', 'GradientBoostingClassifier', 'KNearestNeighborsClassifier',
            'LDA', 'LibLinear_SVC', 'LibSVM_SVC', 'MLPClassifier', 'MultinomialNB',
            'PassiveAggressive', 'QDA', 'RandomForest',  'SGD')
    'auto' will select all default models, or use a list to select

    validation: Whether to use train_test_split to test performance on test set, default = True
    
    test_size: Test percentage used to evaluate the perforamance, default = 0.15
    only effective when validation = True

    objective: Objective function to test performance, default = 'accuracy'
    support ("accuracy", "precision", "auc", "hinge", "f1")
    
    method: Model selection/hyperparameter optimization methods, default = 'Bayeisan'
    
    algo: Search algorithm, default = 'tpe'
    support (rand, tpe, atpe)
    
    spark_trials: Whether to use SparkTrials, default = True

    progressbar: Whether to show progress bar, default = False
    
    seed: Random seed, default = 1
    """

    def __init__(
        self,
        timeout=360,
        max_evals=64,
        temp_directory = 'tmp',
        delete_temp_after_terminate = False,
        encoder="auto",
        imputer="auto",
        scaling="auto",
        balancing="auto",
        feature_selection="auto",
        models="auto",
        validation=True,
        test_size=0.15,
        objective="accuracy",
        method="Bayeisan",
        algo="tpe",
        spark_trials=False,
        progressbar=False,
        seed=1,
    ):
        self.timeout = timeout
        self.max_evals = max_evals
        self.temp_directory = temp_directory
        self.delete_temp_after_terminate = delete_temp_after_terminate
        self.encoder = encoder
        self.imputer = imputer
        self.scaling = scaling
        self.balancing = balancing
        self.feature_selection = feature_selection
        self.models = models
        self.validation = validation
        self.test_size = test_size
        self.objective = objective
        self.method = method
        self.algo = algo
        self.spark_trials = spark_trials
        self.progressbar = progressbar
        self.seed = seed

        self._iter = 0 # record iteration number

    # create hyperparameter space using Hyperopt.hp.choice
    # the pipeline of AutoClassifier is [encoder, imputer, scaling, balancing, feature_selection, model]
    # only chosen ones will be added to hyperparameter space
    def _get_hyperparameter_space(
        self,
        X,
        encoders_hyperparameters,
        encoder,
        imputers_hyperparameters,
        imputer,
        scalings_hyperparameters,
        scaling,
        balancings_hyperparameters,
        balancing,
        feature_selection_hyperparameters,
        feature_selection,
        models_hyperparameters,
        models,
    ):

        # encoding space
        _encoding_hyperparameter = []
        for _encoder in [*encoder]:
            for (
                item
            ) in encoders_hyperparameters:  # search the encoders' hyperparameters
                if item["encoder"] == _encoder:
                    _encoding_hyperparameter.append(item)

        _encoding_hyperparameter = hp.choice(
            "classification_encoders", _encoding_hyperparameter
        )

        # imputation space
        _imputer_hyperparameter = []
        if not X.isnull().values.any():  # if no missing, no need for imputation
            _imputer_hyperparameter = hp.choice(
                "classification_imputers", [{"imputer": "no_processing"}]
            )
        else:
            for _imputer in [*imputer]:
                for (
                    item
                ) in imputers_hyperparameters:  # search the imputer' hyperparameters
                    if item["imputer"] == _imputer:
                        _imputer_hyperparameter.append(item)

            _imputer_hyperparameter = hp.choice(
                "classification_imputers", _imputer_hyperparameter
            )

        # scaling space
        _scaling_hyperparameter = []
        for _scaling in [*scaling]:
            for (
                item
            ) in scalings_hyperparameters:  # search the scalings' hyperparameters
                if item["scaling"] == _scaling:
                    _scaling_hyperparameter.append(item)

        _scaling_hyperparameter = hp.choice(
            "classification_scaling", _scaling_hyperparameter
        )

        # balancing space
        _balancing_hyperparameter = []
        for _balancing in [*balancing]:
            for (
                item
            ) in balancings_hyperparameters:  # search the balancings' hyperparameters
                if item["balancing"] == _balancing:
                    _balancing_hyperparameter.append(item)

        _balancing_hyperparameter = hp.choice(
            "classiciation_balancing", _balancing_hyperparameter
        )

        # feature selection space
        _feature_selection_hyperparameter = []
        for _feature_selection in [*feature_selection]:
            for (
                item
            ) in (
                feature_selection_hyperparameters
            ):  # search the feature selections' hyperparameters
                if item["feature_selection"] == _feature_selection:
                    _feature_selection_hyperparameter.append(item)

        _feature_selection_hyperparameter = hp.choice(
            "classification_feature_selection", _feature_selection_hyperparameter
        )

        # model selection and hyperparameter optimization space
        _model_hyperparameter = []
        for _model in [*models]:
            # checked before at models that all models are in default space
            for item in models_hyperparameters:  # search the models' hyperparameters
                if item["model"] == _model:
                    _model_hyperparameter.append(item)

        _model_hyperparameter = hp.choice(
            "classification_models", _model_hyperparameter
        )

        # return _model_hyperparameter

        # the pipeline search space
        return pyll.as_apply(
            {
                "encoder": _encoding_hyperparameter,
                "imputer": _imputer_hyperparameter,
                "scaling": _scaling_hyperparameter,
                "balancing": _balancing_hyperparameter,
                "feature_selection": _feature_selection_hyperparameter,
                "classification": _model_hyperparameter,
            }
        )

    # initialize and get hyperparameter search space
    def get_hyperparameter_space(self, X, y):
        
        # initialize default search options
        # all encoders avaiable
        self._all_encoders = My_AutoML.encoders

        # all hyperparameters for encoders
        self._all_encoders_hyperparameters = encoder_hyperparameter

        # all imputers available
        self._all_imputers = My_AutoML.imputers

        # all hyperparemeters for imputers
        self._all_imputers_hyperparameters = imputer_hyperparameter

        # all scalings avaiable
        self._all_scalings = My_AutoML.scalings

        # all hyperparameters for scalings
        self._all_scalings_hyperparameters = scaling_hyperparameter

        # all balancings available
        self._all_balancings = My_AutoML.balancing

        # all hyperparameters for balancing methods
        self._all_balancings_hyperparameters = balancing_hyperparameter

        # all feature selections available
        self._all_feature_selection = My_AutoML.feature_selection
        # special treatment, remove some feature selection for regression
        del self._all_feature_selection["extra_trees_preproc_for_regression"]
        del self._all_feature_selection["select_percentile_regression"]
        del self._all_feature_selection["select_rates_regression"]

        # all hyperparameters for feature selections
        self._all_feature_selection_hyperparameters = feature_selection_hyperparameter

        # all classfication models available
        self._all_models = classifiers

        # all hyperparameters for the classification models
        self._all_models_hyperparameters = classifier_hyperparameter

        self.hyperparameter_space = None

        # Encoding
        # convert string types to numerical type
        # get encoder space
        if self.encoder == "auto":
            encoder = self._all_encoders.copy()
        else:
            encoder = {}  # if specified, check if encoders in default encoders
            for _encoder in self.encoder:
                if _encoder not in [*self._all_encoders]:
                    raise ValueError(
                        "Only supported encoders are {}, get {}.".format(
                            [*self._all_encoders], _encoder
                        )
                    )
                encoder[_encoder] = self._all_encoders[_encoder]

        # Imputer
        # fill missing values
        # get imputer space
        if self.imputer == "auto":
            if not X.isnull().values.any():  # if no missing values
                imputer = {"no_processing": no_processing}
                self._all_imputers = imputer # limit default imputer space
            else:
                imputer = self._all_imputers.copy()
        else:
            if not X.isnull().values.any():  # if no missing values
                imputer = {"no_processing": no_processing}
                self._all_imputers = imputer
            else:
                imputer = {}  # if specified, check if imputers in default imputers
                for _imputer in self.imputer:
                    if _imputer not in [*self._all_imputers]:
                        raise ValueError(
                            "Only supported imputers are {}, get {}.".format(
                                [*self._all_imputers], _imputer
                            )
                        )
                    imputer[_imputer] = self._all_imputers[_imputer]

        # Scaling
        # get scaling space
        if self.scaling == "auto":
            scaling = self._all_scalings.copy()
        else:
            scaling = {}  # if specified, check if scalings in default scalings
            for _scaling in self.scaling:
                if _scaling not in [*self._all_scalings]:
                    raise ValueError(
                        "Only supported scalings are {}, get {}.".format(
                            [*self._all_scalings], _scaling
                        )
                    )
                scaling[_scaling] = self._all_scalings[_scaling]

        # Balancing
        # deal with imbalanced dataset, using over-/under-sampling methods
        # get balancing space
        if self.balancing == "auto":
            balancing = self._all_balancings.copy()
        else:
            balancing = {}  # if specified, check if balancings in default balancings
            for _balancing in self.balancing:
                if _balancing not in [*self._all_balancings]:
                    raise ValueError(
                        "Only supported balancings are {}, get {}.".format(
                            [*self._all_balancings], _balancing
                        )
                    )
                balancing[_balancing] = self._all_balancings[_balancing]

        # Feature selection
        # Remove redundant features, reduce dimensionality
        # get feature selection space
        if self.feature_selection == "auto":
            feature_selection = self._all_feature_selection.copy()
        else:
            feature_selection = (
                {}
            )  # if specified, check if balancings in default balancings
            for _feature_selection in self.feature_selection:
                if _feature_selection not in [*self._all_feature_selection]:
                    raise ValueError(
                        "Only supported feature selections are {}, get {}.".format(
                            [*self._all_feature_selection], _feature_selection
                        )
                    )
                feature_selection[_feature_selection] = self._all_feature_selection[
                    _feature_selection
                ]

        # Model selection/Hyperparameter optimization
        # using Bayesian Optimization
        # model space, only select chosen models to space
        if self.models == "auto":  # if auto, model pool will be all default models
            models = self._all_models.copy()
        else:
            models = {}  # if specified, check if models in default models
            for _model in self.models:
                if _model not in [*self._all_models]:
                    raise ValueError(
                        "Only supported models are {}, get {}.".format(
                            [*self._all_models], _model
                        )
                    )
                models[_model] = self._all_models[_model]

        # initialize the hyperparameter space
        _all_encoders_hyperparameters = self._all_encoders_hyperparameters.copy()
        _all_imputers_hyperparameters = self._all_imputers_hyperparameters.copy()
        _all_scalings_hyperparameters = self._all_scalings_hyperparameters.copy()
        _all_balancings_hyperparameters = self._all_balancings_hyperparameters.copy()
        _all_feature_selection_hyperparameters = (
            self._all_feature_selection_hyperparameters.copy()
        )
        _all_models_hyperparameters = self._all_models_hyperparameters.copy()

        # generate the hyperparameter space
        if self.hyperparameter_space is None:
            self.hyperparameter_space = self._get_hyperparameter_space(
                X,
                _all_encoders_hyperparameters,
                encoder,
                _all_imputers_hyperparameters,
                imputer,
                _all_scalings_hyperparameters,
                scaling,
                _all_balancings_hyperparameters,
                balancing,
                _all_feature_selection_hyperparameters,
                feature_selection,
                _all_models_hyperparameters,
                models,
            )  # _X to choose whether include imputer
            # others are the combinations of default hyperparamter space & methods selected

        return encoder, imputer, scaling, balancing, feature_selection, models

    # select optimal settings and fit on opitmal hyperparameters
    def _fit_optimal(self, best_results, _X, _y):

        # mapping the optimal model and hyperparameters selected
        # fit the optimal setting
        optimal_point = space_eval(self.hyperparameter_space, best_results)
        # optimal encoder
        self.optimal_encoder_hyperparameters = optimal_point["encoder"]
        self.optimal_encoder = self.optimal_encoder_hyperparameters["encoder"]
        del self.optimal_encoder_hyperparameters["encoder"]
        # optimal imputer
        self.optimal_imputer_hyperparameters = optimal_point["imputer"]
        self.optimal_imputer = self.optimal_imputer_hyperparameters["imputer"]
        del self.optimal_imputer_hyperparameters["imputer"]
        # optimal scaling
        self.optimal_scaling_hyperparameters = optimal_point["scaling"]
        self.optimal_scaling = self.optimal_scaling_hyperparameters["scaling"]
        del self.optimal_scaling_hyperparameters["scaling"]
        # optimal balancing
        self.optimal_balancing_hyperparameters = optimal_point["balancing"]
        self.optimal_balancing = self.optimal_balancing_hyperparameters["balancing"]
        del self.optimal_balancing_hyperparameters["balancing"]
        # optimal feature selection
        self.optimal_feature_selection_hyperparameters = optimal_point[
            "feature_selection"
        ]
        self.optimal_feature_selection = self.optimal_feature_selection_hyperparameters[
            "feature_selection"
        ]
        del self.optimal_feature_selection_hyperparameters["feature_selection"]
        # optimal classifier
        self.optimal_classifier_hyperparameters = optimal_point[
            "classification"
        ]  # optimal model selected
        self.optimal_classifier = self.optimal_classifier_hyperparameters[
            "model"
        ]  # optimal hyperparameter settings selected
        del self.optimal_classifier_hyperparameters["model"]

        # encoding
        self._fit_encoder = self._all_encoders[self.optimal_encoder](
            **self.optimal_encoder_hyperparameters
        )
        _X = self._fit_encoder.fit(_X)
        # imputer
        self._fit_imputer = self._all_imputers[self.optimal_imputer](
            **self.optimal_imputer_hyperparameters
        )
        _X = self._fit_imputer.fill(_X)
        # scaling
        self._fit_scaling = self._all_scalings[self.optimal_scaling](
            **self.optimal_scaling_hyperparameters
        )
        self._fit_scaling.fit(_X)
        _X = self._fit_scaling.transform(_X)
        # balancing
        self._fit_balancing = self._all_balancings[self.optimal_balancing](
            **self.optimal_balancing_hyperparameters
        )
        _X = self._fit_balancing.fit_transform(_X)
        # feature selection
        self._fit_feature_selection = self._all_feature_selection[
            self.optimal_feature_selection
        ](**self.optimal_feature_selection_hyperparameters)
        self._fit_feature_selection.fit(_X, _y)
        _X = self._fit_feature_selection.transform(_X)
        # classification
        self._fit_classifier = self._all_models[self.optimal_classifier](
            **self.optimal_classifier_hyperparameters
        )
        self._fit_classifier.fit(_X.values, _y.values.ravel())

        return self

    def fit(self, X, y):

        # initialize temp directory
        # check if temp directory exists, if exists, empty it
        if os.path.isdir(self.temp_directory) :
            shutil.rmtree(self.temp_directory)
        os.makedirs(self.temp_directory)

        # write basic information to init.txt
        with open(self.temp_directory + '/init.txt', 'w') as f:
            f.write('Features of the dataset: {}'.format(list(X.columns)))
            f.write('Shape of the design matrix: {} * {}'.format(X.shape[0], X.shape[1]))
            f.write('Resposne of the dataset: {}'.format(list(y.columns)))
            f.write('Shape of the response vector: {} * {}'.format(y.shape[0], y.shape[1]))
            f.write('Type of the task: Classification.')

        _X = X.copy()
        _y = y.copy()

        (
            encoder,
            imputer,
            scaling,
            balancing,
            feature_selection,
            models,
        ) = self.get_hyperparameter_space(_X, _y)

        if self.validation:  # only perform train_test_split when validation
            # train test split so the performance of model selection and
            # hyperparameter optimization can be evaluated
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                _X, _y, test_size=self.test_size, random_state=self.seed
            )

        # the objective function of Bayesian Optimization tries to minimize
        # use accuracy score
        @ignore_warnings(category=ConvergenceWarning)
        def _objective(params):
            # evaluation for predictions
            if self.objective == "accuracy":
                from sklearn.metrics import accuracy_score

                _obj = accuracy_score
            elif self.objective == "precision":
                from sklearn.metrics import precision_score

                _obj = precision_score
            elif self.objective == "auc":
                from sklearn.metrics import roc_auc_score

                _obj = roc_auc_score
            elif self.objective == "hinge":
                from sklearn.metrics import hinge_loss

                _obj = hinge_loss
            elif self.objective == "f1":
                from sklearn.metrics import f1_score

                _obj = f1_score
            else:
                raise ValueError(
                    'Only support ["accuracy", "precision", "auc", "hinge", "f1"], get{}'.format(
                        self.objective
                    )
                )

            # pipeline of objective, [encoder, imputer, scaling, balancing, feature_selection, model]
            # select encoder and set hyperparameters
            # must have encoder
            _encoder_hyper = params["encoder"]
            _encoder = _encoder_hyper["encoder"]
            del _encoder_hyper["encoder"]
            enc = encoder[_encoder](**_encoder_hyper)

            # select imputer and set hyperparameters
            _imputer_hyper = params["imputer"]
            _imputer = _imputer_hyper["imputer"]
            del _imputer_hyper["imputer"]
            imp = imputer[_imputer](**_imputer_hyper)

            # select scaling and set hyperparameters
            # must have scaling, since no_preprocessing is included
            _scaling_hyper = params["scaling"]
            _scaling = _scaling_hyper["scaling"]
            del _scaling_hyper["scaling"]
            scl = scaling[_scaling](**_scaling_hyper)

            # select balancing and set hyperparameters
            # must have balancing, since no_preprocessing is included
            _balancing_hyper = params["balancing"]
            _balancing = _balancing_hyper["balancing"]
            del _balancing_hyper["balancing"]
            blc = balancing[_balancing](**_balancing_hyper)

            # select feature selection and set hyperparameters
            # must have feature selection, since no_preprocessing is included
            _feature_selection_hyper = params["feature_selection"]
            _feature_selection = _feature_selection_hyper["feature_selection"]
            del _feature_selection_hyper["feature_selection"]
            fts = feature_selection[_feature_selection](**_feature_selection_hyper)

            # select classifier model and set hyperparameters
            # must have a classifier
            _classifier_hyper = params["classification"]
            _classifier = _classifier_hyper["model"]
            del _classifier_hyper["model"]
            clf = models[_classifier](
                **_classifier_hyper
            )  # call the model using passed parameters

            obj_tmp_dicretory = self.temp_directory + '/iter_' + str(self._iter + 1)
            if not os.path.isdir(obj_tmp_dicretory) :
                os.makedirs(obj_tmp_dicretory)
            
            with open(obj_tmp_dicretory + '/hyperparamter_settings.txt', 'w') as f:
                f.write('Encoding method: {}\n'.format(_encoder))
                f.write('Encoding Hyperparameters:')
                print(_encoder_hyper, file = f, end = '\n\n')
                f.write('Imputation method: {}\n'.format(_imputer))
                f.write('Imputation Hyperparameters:')
                print(_imputer_hyper, file = f, end = '\n\n')
                f.write('Scaling method: {}\n'.format(_scaling))
                f.write('Scaling Hyperparameters:')
                print(_scaling_hyper, file = f, end = '\n\n')
                f.write('Balancing method: {}\n'.format(_balancing))
                f.write('Balancing Hyperparameters:')
                print(_balancing_hyper, file = f, end = '\n\n')
                f.write('Feature Selection method: {}\n'.format(_feature_selection))
                f.write('Feature Selection Hyperparameters:')
                print(_feature_selection_hyper, file = f, end = '\n\n')
                f.write('Classification model: {}\n'.format(_classifier))
                f.write('Classifier Hyperparameters:')
                print(_classifier_hyper, file = f, end = '\n\n')

            if self.validation:
                _X_train_obj, _X_test_obj = X_train.copy(), X_test.copy()
                _y_train_obj, _y_test_obj = y_train.copy(), y_test.copy()

                # encoding
                _X_train_obj = enc.fit(_X_train_obj)
                _X_test_obj = enc.refit(_X_test_obj)
                # imputer
                _X_train_obj = imp.fill(_X_train_obj)
                _X_test_obj = imp.fill(_X_test_obj)
                # scaling
                scl.fit(_X_train_obj, _y_train_obj)
                _X_train_obj = scl.transform(_X_train_obj)
                _X_test_obj = scl.transform(_X_test_obj)
                # balancing
                # _X_train_obj = blc.fit_transform(_X_train_obj)
                # feature selection
                fts.fit(_X_train_obj, _y_train_obj)
                _X_train_obj = fts.transform(_X_train_obj)
                _X_test_obj = fts.transform(_X_test_obj)
                # classification
                clf.fit(_X_train_obj, _y_train_obj.values.ravel())
                
                y_pred = clf.predict(_X_test_obj)
                _loss = -_obj(y_pred, _y_test_obj.values)
                
                with open(obj_tmp_dicretory + '/testing_objective.txt', 'w') as f:
                    f.write('Loss from objective function is: {:.6f}\n'.format(_loss))
                    f.write('Loss is calculate using {}.'.format(self.objective))
                self._iter += 1

                # since fmin of Hyperopt tries to minimize the objective function, take negative accuracy here
                return {"loss": _loss, "status": STATUS_OK}
            else:
                _X_obj = _X.copy()
                _y_obj = _y.copy()

                # encoding
                _X_obj = enc.fit(_X_obj)
                # imputer
                _X_obj = imp.fill(_X_obj)
                # scaling
                scl.fit(_X_obj, _y_obj)
                _X_obj = scl.transform(_X_obj)
                # balancing
                _X_obj = blc.fit_transform(_X_obj)
                # feature selection
                fts.fit(_X_obj, _y_obj)
                _X_obj = fts.transform(_X_obj)
                # classification
                clf.fit(_X_obj.values, _y_obj.values.ravel())

                y_pred = clf.predict(_X_obj.values)
                _loss = -_obj(y_pred, _y_obj.values)

                with open(obj_tmp_dicretory + '/testing_objective.txt', 'w') as f:
                    f.write('Loss from objective function is: {.6f}\n'.format(_loss))
                    f.write('Loss is calculate using {}.'.format(self.objective))
                self._iter += 1

                return {"loss": _loss, "status": STATUS_OK}

        # call hyperopt to use Bayesian Optimization for Model Selection and Hyperparameter Selection

        # search algorithm
        if self.algo == "rand":
            algo = rand.suggest
        elif self.algo == "tpe":
            algo = tpe.suggest
        elif self.algo == "atpe":
            algo = atpe.suggest

        # Storage for evaluation points
        if self.spark_trials:
            trials = SparkTrials()
        else:
            trials = Trials()

        # run fmin to search for optimal hyperparameter settings
        with mlflow.start_run():
            best_results = fmin(
                fn=_objective,
                space=self.hyperparameter_space,
                algo=algo,
                max_evals=self.max_evals,
                timeout=self.timeout,
                trials=trials,
                show_progressbar=self.progressbar,
                rstate=np.random.RandomState(seed=self.seed),
            )

        # select optimal settings and fit optimal pipeline
        self._fit_optimal(best_results, _X, _y)

        # whether to retain temp files
        if not self.delete_temp_after_terminate :
            shutil.rmtree(self.temp_directory)

        return self

    def predict(self, X):

        _X = X.copy()

        # may need preprocessing for test data, the preprocessing shoul be the same as in fit part
        # Encoding
        # convert string types to numerical type
        _X = self._fit_encoder.refit(_X)

        # Imputer
        # fill missing values
        _X = self._fit_imputer.fill(_X)

        # Scaling
        _X = self._fit_scaling.transform(_X)

        # Balancing
        # deal with imbalanced dataset, using over-/under-sampling methods
        # No need to balance on test data

        # Feature selection
        # Remove redundant features, reduce dimensionality
        _X = self._fit_feature_selection.transform(_X)

        return self._fit_classifier.predict(_X.values)


"""
Regressors/Hyperparameters from sklearn:
1. AdaBoost: n_estimators, learning_rate, loss, max_depth
2. Ard regression: n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2,
            threshold_lambda, fit_intercept
3. Decision tree: criterion, max_features, max_depth_factor,
            min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_leaf_nodes, min_impurity_decrease
4. extra trees: criterion, min_samples_leaf, min_samples_split, 
            max_features, bootstrap, max_leaf_nodes, max_depth, 
            min_weight_fraction_leaf, min_impurity_decrease
5. Gaussian Process: alpha, thetaL, thetaU
6. Gradient boosting: loss, learning_rate, min_samples_leaf, max_depth,
            max_leaf_nodes, max_bins, l2_regularization, early_stop, tol, scoring
7. KNN: n_neighbors, weights, p
8. Linear SVR (LibLinear): loss, epsilon, dual, tol, C, fit_intercept,
            intercept_scaling
9. Kernel SVR (LibSVM): kernel, C, epsilon, tol, shrinking
10. Random forest: criterion, max_features, max_depth, min_samples_split, 
            min_samples_leaf, min_weight_fraction_leaf, bootstrap, 
            max_leaf_nodes, min_impurity_decrease
11. SGD (Stochastic Gradient Descent): loss, penalty, alpha, fit_intercept, tol,
            learning_rate
12. MLP (Multilayer Perceptron): hidden_layer_depth, num_nodes_per_layer, 
            activation, alpha, learning_rate_init, early_stopping, solver, 
            batch_size, n_iter_no_change, tol, shuffle, beta_1, beta_2, epsilon
"""


class AutoRegressor:
    def __init__(
        self,
        timeout=360,
        max_evals=32,
        models="auto",
        test_size=0.15,
        method="Bayeisan",
        algo="tpe",
        spark_trials=True,
        seed=1,
    ):
        self.timeout = timeout
        self.max_evals = max_evals
        self.models = models
        self.test_size = test_size
        self.method = method
        self.algo = algo
        self.spark_trials = spark_trials
        self.seed = seed

        # all regression models available
        self._all_models = regressors

        # all hyperparameters for the regression models
        self._all_models_hyperparameters = regressor_hyperparameters

        self.hyperparameter_space = None

    # create hyperparameter space using Hyperopt.hp.choice
    # only models in models will be added to hyperparameter space
    def get_hyperparameter_space(self, models):

        _hyperparameter = []
        for _model in [*models]:
            # checked before at models that all models are in default space
            for (
                item
            ) in self._all_models_hyperparameters:  # search the models' hyperparameters
                if item["model"] == _model:
                    _hyperparameter.append(item)

        return hp.choice("regression_models", _hyperparameter)

    def fit(self, X, y):

        # Encoding
        # convert string types to numerical type
        from My_AutoML import DataEncoding

        x_encoder = DataEncoding()
        _X = x_encoder.fit(X)

        y_encoder = DataEncoding()
        _y = y_encoder.fit(y)

        # Imputer
        # fill missing values
        from My_AutoML import SimpleImputer

        imputer = SimpleImputer(method="mean")
        _X = imputer.fill(_X)

        # Scaling

        # Balancing
        # deal with imbalanced dataset, using over-/under-sampling methods

        # Feature selection
        # Remove redundant features, reduce dimensionality

        # train test split so the performance of model selection and
        # hyperparameter optimization can be evaluated
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            _X, _y, test_size=self.test_size, random_state=self.seed
        )

        if self.models == "auto":  # if auto, model pool will be all default models
            models = self._all_models.copy()
        else:
            models = {}  # if specified, check if models in default models
            for _model in self.models:
                if _model not in [*self._all_models]:
                    raise ValueError(
                        "Only supported models are {}, get {}.".format(
                            [*self._all_models], _model
                        )
                    )
                models[_model] = self._all_models[_model]

        # initialize the hyperparameter space
        if self.hyperparameter_space is None:
            self.hyperparameter_space = self.get_hyperparameter_space(models)

        # call hyperopt to use Bayesian Optimization for Model Selection and Hyperparameter Selection

        # the objective function of Bayesian Optimization tries to minimize
        # use mean_squared_error
        def _objective(params):

            _model = params["model"]
            del params["model"]
            clf = models[_model](**params)  # call the model using passed parameters

            from sklearn.metrics import mean_squared_error

            clf.fit(X_train.values, y_train.values.ravel())
            y_pred = clf.predict(X_test.values)

            # since fmin of Hyperopt tries to minimize the objective function, take negative
            return {
                "loss": -mean_squared_error(y_pred, y_test.values),
                "status": STATUS_OK,
            }

        # search algorithm
        if self.algo == "rand":
            algo = rand.suggest
        elif self.algo == "tpe":
            algo = tpe.suggest
        elif self.algo == "atpe":
            algo = atpe.suggest

        # Storage for evaluation points
        if self.spark_trials:
            trials = SparkTrials()
        else:
            trials = Trials()

        with mlflow.start_run():
            best_results = fmin(
                fn=_objective,
                space=self.hyperparameter_space,
                algo=algo,
                max_evals=self.max_evals,
                timeout=self.timeout,
                trials=trials,
                show_progressbar=False,
            )
