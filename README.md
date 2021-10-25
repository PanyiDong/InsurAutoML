# This is project for Auto Machine Learning

IRisk Lab Project, UIUC, Fall 2021

The project aims to create a AutoML package with special focus on insurance data (with some inbalance in nature).


Current Progress:
1. base: load data with some common file format (.csv, .data, .rda, etc.) given path into a database dict, and some common utilities
2. encoding: encoding for categorical/string type to numerical, and can perform dummy coding; category shift
3. impute for missing values: some of the available methods: Simple Imputer, Miss Forest Imputer, MICE, GAIN, etc.
4. common scaling methods: Standardize, MinMax Scaling, Winsorization, etc.
5. deal with imbalance data: use over-/under-sampling methods to balance the dataset, some of the available methods: Tome kLink, One Sided Selection (OSS), Smote, etc.
6. feature selection: PCA, AFFS, etc. And some models from autosklearn will be applied.
7. apply hyperopt (Bayesian Optimization) to create a pipeline of AutoML workflow. Consider the methods as a hyperparamter, and create a hyperparamter space, where we can find the optimal ML workflow.
