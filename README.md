# This is project for Auto Machine Learning

IRisk Lab Project, UIUC, Fall 2021

The project aims to create a AutoML package with special focus on insurance data (with some imbalance in nature).

## Usage

1. Clone the repository

```console
git clone https://github.com/PanyiDong/My_AutoML.git
```

2. Install dependencies

```console
cd My_AutoML
pip install -r requirements.txt
```

3. Put data in the folder and run for training/evaluation

Example below runs a classification task on heart.csv file in example/example_data folder

`train_data` and `response` are two must-have arguments corresponds to training data name (no file extension needed), and response variable name.

Your can specify the data folder (or by default at current folder), test_data name (or evaluation set will be split from train_data), evaluation metrics and all model parameters by arguments. 

```console
python main.py --data_folder example/example_data --train_data heart --response HeartDisease
```

Or, you can treat it like a package and follows the fit/predict workflow like jupyter notebooks in `examples`.

```python
from My_AutoML import AutoTabular
model = AutoTabular()
model.fit(train_X, train_y)

model.predict(test_X)
```

## Summary

> Required Packages: numpy, pandas, scipy, os, shutil, time, itertools, functools, importlib, cmath, tqdm, warnings, rpy2$^{*}$, tensorflow$^{**}$, mlflow, hyperopt, sklearn, autosklearn
>
> $*$ rpy2 is only used for reading .rda/.rdata datasets. If rpy2 is not installed, it will not cause import problems (using importlib to check), but you will not be able to read R datasets
>
> $**$ tensorflow is now only used for imputation with GAIN network. If tensorflow not installed, it will not caused import problems, but the GAIN imputation method will be disabled in default hyperparameter space.

Current Progress:

> 1. base: load data with some common file format (.csv, .data, .rda, etc.) given path into a database dict, and some common utilities
>
> 2. encoding: encoding for categorical/string type to numerical, and can perform dummy coding; category shift
>
> 3. impute for missing values: some of the available methods: Simple Imputer, Miss Forest Imputer, MICE, GAIN, etc.
>
> 4. common scaling methods: Standardize, MinMax Scaling, Winsorization, etc.
>
> 5. deal with imbalance data: use over-/under-sampling methods to balance the dataset, some of the available methods: Tome kLink, One Sided Selection (OSS), Smote, etc.
>
> 6. feature selection: PCA, AFFS, etc. And some models from autosklearn will be applied.
>
> 7. apply hyperopt (Bayesian Optimization) to create a pipeline of AutoML workflow. Consider the methods as a hyperparameter, and create a hyperparameter space, where we can find the optimal ML workflow. Only supervised classification/regression models supported.

The pipeline of AutoML:

> 1. Data Encoding: encode some string type and datetime types to numerical type so it can be more easily handled.
>
> 2. Imputation: impute the missing values in the dataset using supervised/unsupervised learning in case the following processes can not handle NaN values. Will be no_processing if no missing values
>
> 3. Balancing: balance the datasets using under-/over-sampling methods to emphasize the importance of extreme minority class. Will not change the datasets if the datasets are already balanced.
>
> 4. Scaling: scale the datasets to increase the efficiency of convergence, or allows better performance. Can be selected as no_processing.
>
> 5. Feature Selection: remove redundant or insignificant features to improve the performance of the model. Can be selected as no_processing.
>
> 6. Regression/Classification: perform regression/classification models to fit the datasets.

Save and load the models: To save reproduction time, when the optimal model/hyperparameter settings are configured, all settings will be stored as a `model` file. Next time when AutoML pipeline starts training, it will detect whether the `model` file exists and only fit the optimal pipeline, which can save the training time (for optimization). On test dataset Employee Future prediction, the over 3 minutes training time can be reduced to 2.1 seconds reproduction time.

Other files in the repository:

1. `report.pdf` provides an introduction to the AutoML pipeline and demonstrates test performance on some real-life datasets, and `Appendix` provides test datasets in the report.

2. `Dockerfiles` provides a Docker environment preparation files, you can easily build a virtual environment and test your datasets on the AutoML pipeline. The dockerfiles will install necessary packages and clone this repository to workspace.

## Future

I'm still interested to expand current AutoML package, and have plans to add MLP/RNN structure AutoML for tabular tasks/NLP tasks. But no schedules for such progress can be made.
