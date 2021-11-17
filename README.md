# This is project for Auto Machine Learning

IRisk Lab Project, UIUC, Fall 2021

The project aims to create a AutoML package with special focus on insurance data (with some inbalance in nature).

```
Required Packages: numpy, pandas, scipy, os, shutil, time, itertools, functools, importlib, cmath, tqdm, warnings, rpy2*, tensorflow**, mlflow, hyperopt, sklearn, autosklearn
* rpy2 is only used for reading .rda/.rdata datasets. If rpy2 is not installed, it will not cause import problems (using importlib to check), but you will not be able to read R datasets
** tesnsorflow is now only used for imputation with GAIN network. If tensorflow not installed, it will not caused import problems, but the GAIN imputation method will be disabled in default hyperparameter space.
```

Current Progress:
```
1. base: load data with some common file format (.csv, .data, .rda, etc.) given path into a database dict, and some common utilities
2. encoding: encoding for categorical/string type to numerical, and can perform dummy coding; category shift
3. impute for missing values: some of the available methods: Simple Imputer, Miss Forest Imputer, MICE, GAIN, etc.
4. common scaling methods: Standardize, MinMax Scaling, Winsorization, etc.
5. deal with imbalance data: use over-/under-sampling methods to balance the dataset, some of the available methods: Tome kLink, One Sided Selection (OSS), Smote, etc.
6. feature selection: PCA, AFFS, etc. And some models from autosklearn will be applied.
7. apply hyperopt (Bayesian Optimization) to create a pipeline of AutoML workflow. Consider the methods as a hyperparamter, and create a hyperparamter space, where we can find the optimal ML workflow. Only supervised classfication/regression models supported.
```

The pipeline of AutoML: 
```
1. Data Encoding: encode some string type and datetime types to numerical type so it can be more easily handled.
2. Imputation: impute the missing values in the dataset using supervised/unsupervised learning in case the following processes can not handle NaN values. Will be no_processing if no missing values
3. Balancing: balance the datasets using under-/over-sampling methods to emphasize the importance of extreme minority class. Will not change the datasets if the datasets are already balanced.
4. Scaling: scale the datasets to increase the efficiency of convergence, or allows better performance. Can be selected as no_processing.
5. Feature Selection: remove reducatant or insignificant features to improve the performance of the model. Can be selected as no_prorcessing.
6. Regression/Classification: performan regression/classficaiton models to fit the datasets.
```
