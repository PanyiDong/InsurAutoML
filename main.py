'''
File: main.py
Author: Panyi Dong
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /main.py
File Created: Friday, 25th February 2022 6:13:42 pm
Author: Panyi Dong (panyid2@illinois.edu)
-----
Last Modified: Tuesday, 1st March 2022 12:10:04 am
Modified By: Panyi Dong (panyid2@illinois.edu>)
-----
Copyright (c) 2022 - 2022, Panyi Dong
All rights reserved.
'''

import argparse
from sqlite3 import DatabaseError

import sklearn
import My_AutoML
from My_AutoML import load_data, train_test_split, type_of_task
from My_AutoML import AutoML, AutoClassifier, AutoRegressor

model_dict = {
    'auto': AutoML,
    'classification': AutoClassifier,
    'regression': AutoRegressor,
}

# read arguments
parser = argparse.ArgumentParser(description = 'set arguments for AutoML job')
# data-related arguments
parser.add_argument('--data_folder', default = '', type = str, 
                    help = 'read-in data folder')
parser.add_argument('--train_data', type = str, 
                    help = 'train data name (no file extension needed)')
parser.add_argument('--test_data', default = '', type = str, 
                    help = 'test data name (if not provided, will train_test_split on train data)')
parser.add_argument('--response', type = str, 
                    help = 'response column name')
parser.add_argument('--test_eval', default = 'auto', type = str, 
                    help = 'evaluation metric on test data')
# model-related arguments
parser.add_argument('--task_type', default = 'auto', type = str, 
                    help = 'task_type: auto, regression, classification')
parser.add_argument('--timeout', default = 360, type = int, 
                    help = 'total time allowed for the task')
parser.add_argument('--max_evals', default = 64, type = int, 
                    help = 'number of evaluation processes for the task')
parser.add_argument('--temp_directory', default = "tmp", type = str, 
                    help = 'path to store temporary model/evaluation information')
parser.add_argument('--delete_temp_after_terminate', default = False, type = bool, 
                    help = 'whether to delete temporary information after training')
parser.add_argument('--save', default = True, type = bool, 
                    help = 'whether to save optimal training model')
parser.add_argument('--ignore_warning', default = True, type = bool, 
                    help = 'whether to ignore all warnings')
parser.add_argument('--encoder', default = "auto", type = str, 
                    help = 'encoders for the tasks, auto or a list of encoders')
parser.add_argument('--imputer', default = "auto", type = str, 
                    help = 'imputers for the tasks, auto or a list of imputers')
parser.add_argument('--balancing', default = "auto", type = str, 
                    help = 'balancings for the tasks, auto or a list of balancings')
parser.add_argument('--scaling', default = "auto", type = str, 
                    help = 'scalings for the tasks, auto or a list of scalings')
parser.add_argument('--feature_selection', default = "auto", type = str, 
                    help = 'feature_selections for the tasks, auto or a list of feature_selections')
parser.add_argument('--models', default = "auto", type = str, 
                    help = 'models for the tasks, auto or a list of models')
parser.add_argument('--validation', default = True, type = bool, 
                    help = 'whether to split a validation set')
parser.add_argument('--valid_size', default = 0.15, type = float, 
                    help = 'validation set percentage to split')
parser.add_argument('--objective', default = None, type = str, 
                    help = 'evaluation metrics for tasks performance')
parser.add_argument('--method', default = "Bayesian", type = str, 
                    help = 'model selection/hyperparameter optimization methods')
parser.add_argument('--algo', default = "tpe", type = str, 
                    help = 'search algorithm, select from ["rand", "tpe", "atpe"]')
parser.add_argument('--spark_trials', default = False, type = bool, 
                    help = 'whether to use SparkTrials')
parser.add_argument('--progressbar', default = True, type = bool, 
                    help = 'whether to display training progressbar')
parser.add_argument('--seed', default = 1, type = int, 
                    help = 'random seed')
args = parser.parse_args()

# convert arguments to parameters
MODEL = model_dict[args.task_type]
TIMEOUT = args.timeout
MAX_EVALS = args.max_evals
TEMP_DIRECTORY = args.temp_directory
DELETE_TEMP_AFTER_TERMINATE = args.delete_temp_after_terminate
SAVE = args.save
MODEL_NAME = args.train_data + '_model'
IGNORE_WARNING = args.ignore_warning
ENCODER  = args.encoder
IMPUTER = args.imputer
BALANCING = args.balancing
SCALING = args.scaling
FEATURE_SELECTION = args.feature_selection
MODELS = args.models
VALIDATION = args.validation
VALID_SIZE = args.valid_size
OBJECTIVE = args.objective
METHOD = args.method
ALGO = args.algo
SPARK_TRIALS = args.spark_trials
PROGRESSBAR = args.progressbar
SEED = args.seed

if __name__ == "__main__" :
    
    print('Preprocessing:')
    train = args.train_data
    test = args.test_data
    response = args.response
    
    print('Train/Test identification.')
    # if test_data provided, use train/test data seperately
    if test != '' :
        database = load_data().load(args.data_folder, [train, test])
        
        # check whether train/test set share same columns
        if set(database[train].columns) != set(database[test].columns) :
            raise DatabaseError(
                'Train/Test datasets have different columns!')
            
        features = list(database[train].columns)
        features.remove(response)
        
        train_X, train_y = database[train][features], \
            database[train][[response]]
        test_X, test_y = database[test][features], \
            database[test][[response]]
    # if no test_data provided, read only train_dat
    # and use train_test_split to get train/test sets
    else :
        database = load_data().load(args.data_folder, [train])
        features = list(database[train].columns)
        features.remove(response)
        
        train_X, test_X, train_y, test_y = train_test_split(
            database[train][features], database[train][[response]]
        )
    
    print('Training:')
    # construct the model by parameters
    model = MODEL(
        timeout = TIMEOUT,
        max_evals = MAX_EVALS,
        temp_directory = TEMP_DIRECTORY,
        delete_temp_after_terminate = DELETE_TEMP_AFTER_TERMINATE,
        save = SAVE,
        model_name = MODEL_NAME,
        ignore_warning = IGNORE_WARNING,
        encoder = ENCODER,
        imputer = IMPUTER,
        balancing = BALANCING,
        scaling = SCALING,
        feature_selection = FEATURE_SELECTION,
        models = MODELS,
        validation = VALIDATION,
        valid_size = VALID_SIZE,
        objective = OBJECTIVE,
        method = METHOD,
        algo = ALGO,
        spark_trials = SPARK_TRIALS,
        progressbar = PROGRESSBAR,
        seed = SEED,
    )
    
    # training process
    model.fit(train_X, train_y)
    
    print('Evaluation:')
    # test and evaluation
    y_pred = model.predict(test_X)
    
    # select from evaluation metrics
    eval_metrics = {
        "accuracy": sklearn.metrics.accuracy_score,
        "precision": sklearn.metrics.precision_score,
        "auc": sklearn.metrics.roc_auc_score,
        "hinge": sklearn.metrics.hinge_loss,
        "f1": sklearn.metrics.f1_score,
        "MSE": sklearn.metrics.mean_squared_error,
        "MAE": sklearn.metrics.mean_absolute_error,
        "MSLE": sklearn.metrics.mean_squared_log_error,
        "R2": sklearn.metrics.r2_score,
        "MAX": sklearn.metrics.max_error,
    }
    
    if args.test_eval == 'auto' :
        _type = type_of_task(train_y)
        if _type in ['binary', 'multiclass'] :
            args.test_eval = "accuracy"
        elif _type in ['integer', 'continuous'] :
            args.test_eval = "MSE"
            
    print("The {} of test data is: {:.4f}".format(
        args.test_eval, eval_metrics[args.test_eval](y_pred, test_y)
    ))
            