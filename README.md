# Project for Auto Machine Learning (AutoML)

[![Linux](https://github.com/PanyiDong/InsurAutoML/actions/workflows/build.yml/badge.svg)](https://github.com/PanyiDong/InsurAutoML/actions/workflows/build.yml) [![Windows](https://github.com/PanyiDong/InsurAutoML/actions/workflows/build-windows.yml/badge.svg)](https://github.com/PanyiDong/InsurAutoML/actions/workflows/build-windows.yml) [![release](https://img.shields.io/github/v/release/PanyiDong/InsurAutoML)](https://img.shields.io/github/v/release/PanyiDong/InsurAutoML) [![codecov](https://codecov.io/gh/PanyiDong/InsurAutoML/branch/master/graph/badge.svg?token=S12Q35HH2Y)](https://codecov.io/gh/PanyiDong/InsurAutoML) 

IRisk Lab Project, UIUC, Fall 2021

Now a personally-maintained project

The project aims to create a AutoML package with special focus on insurance data (with some imbalance in nature). The pipeline is now workable with encoding, imputation, balancing, scaling, feature selection, models (regression, classification models) as pipeline components and model selection/hyperparameter optimization (HPO) process as it's core of connection among all components for tuning.

## Prerequisites

System Requirements:

- Linux (write and test on Ubuntu 20.04) and Windows (Windows is now compatible)

- As all trials are running parallelized, more threads (correspondingly, more memory) will increase the training efficiency

- `pip` is required for installation (`git` for building from source)
  
- Python version: should support version >= 3.7 (write and test on `3.8` and `3.9`)
  
- If neural network is required, GPU supported device will increase the training efficiency

## Installation

### 1. Install from `pip`

Open a command terminal.

```console
pip install InsurAutoML
```

### 2. Build from source

#### 1. Clone the repository

Open a command terminal.

```console
git clone https://github.com/PanyiDong/InsurAutoML.git
cd InsurAutoML
```

#### 2. Install dependencies

#### install by `setup.py`

```console
pip install -e .[normal]
```

This method will use `setup.py` to install the dependencies, by default, if no GPU support, should install `normal` version. If GPU is supported, and you wish to test neural network related architectures, use `pip install -e .[nn]` for neural network installation; or, if you wish to use a lightweight, essential-only installation, use `pip install -e .[lightweight]`.

At this moment, `normal` contains few more ML packages that allows testing on a larger model/hyperparameter space. The differences may becomes larger for later versions.

#### install by `requirements.txt`

> It's recommended to use above `setup.py` to install dependencies as it provides more flexibility to recognize your operating system, compilation issues and some optional packages.

```console
pip install -r requirements.txt
```

For neural network related support (recommended for `CUDA` supported devices), please use `pip install -r requirements_nn.txt`. The pipeline works without any neural network support with the loss of neural network support. If no CUDA device available, please use a non-`torch` environment as those neural network methods can take forever to finish.

> NOTE: 1. For Windows, the installation of `rpy2` is not straightforward, so it may not actually useable, but it's in-built here in `requirements.txt`. If Windows user install through `setup.py`, it will not be included.

## Usage

### 1. Put data in the folder and run for training/evaluation

Example below runs a classification task on `heart.csv` file in `example/example_data` folder

`train_data` and `response` are two must-have arguments corresponds to training data name (no file extension needed), and response variable name.

Your can specify the data folder (or by default at current folder), test_data name (or evaluation set will be split from train_data), evaluation metrics and all model parameters by arguments.

### 2. Run experiments

In the command terminal:

```console
python main.py --data_folder example/example_data --train_data heart --response HeartDisease
```

And an example run regression task on `insurance.csv` file in `example/example_data` folder

```console
python main.py --data_folder example/example_data --train_data insurance --response expenses
```

Or, you can treat it like a package and follows the fit/predict workflow like jupyter notebooks in `examples`.

```python
from InsurAutoML import AutoTabular
model = AutoTabular()
model.fit(train_X, train_y)

model.predict(test_X)
```

By default, progress reporter `CLIReporter` is prepared for terminal/command-like report, when using jupyter notebook, call by `AutoTabular(progress_reporter = "JupyterNotebookReporter")` for overwriting previous outputs. (Now, the pipeline can identify whether console terminal/Jupyter Notebook environment is used, you don't need to worry about it.)

Moreover, in current version, model selection and hyperparameter optimization is achieved by `ray.tune`. However, if you still need previous `HyperOpt` version, you can call as:

```python
from InsurAutoML._legacy import AutoTabular
model = AutoTabular()
model.fit(train_X, train_y)

model.predict(test_X)
```

One important issue I find now is that, `ray.tune` does not force to stop running trials but only stop generating new trials. So, if setting `timeout` too low, it's common the fitting does not end in time (long running trials are expected not to finish in short time). However, if the pipeline is used as normal cases, which the `timeout` and `max_evals` both should be set to a rather large number, this case should not be as significant as very short time limit. I think that force trial runtime to stop can be a solution, but with few tryouts, I haven't found anything useful yet.

## Summary

> Required Packages: numpy, pandas, scipy, matplotlib, ray, ray[tune], ray[rllib], tqdm, mlflow, tensorboardX, hyperopt, auto-sklearn $^{1}$, scikit-learn, lightgbm, xgboost, pygam, rpy2 $^{2}$, tensorflow $^{3}$, pytorch $^{4}$
>
> <sub><sup>1.</sup></sub> auto-sklearn is a AutoML package, which provides some models for the pipeline. But it's only compatible with Linux system. So if you are using windows, this package should not be installed.
>
> <sub><sup>2.</sup></sub> rpy2 is only used for reading .rda/.rdata datasets. If rpy2 is not installed, it will not cause import problems (using importlib to check), but you will not be able to read R datasets
>
> <sub><sup>3.</sup></sub> tensorflow is now only used for imputation with GAIN network. If tensorflow not installed, it will not caused import problems, but the GAIN imputation method will be disabled in default hyperparameter space.
>
> <sub><sup>4.</sup></sub> pytorch is required for neural network support.

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
> 7. apply `ray.tune` (with plentiful search algorithms and search schedulers) to create a pipeline of AutoML workflow. Consider the methods as a hyperparameter, and create a hyperparameter space, where we can find the optimal ML workflow. Only supervised classification/regression models supported.

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

With the new update, cutomized/personalized preprocessing methods and models are supported by using the `additional.py` file as additional input. In the file, there's also a template of how to write the methods and correpsonding hyperparameter space.

Save and load the models: To save reproduction time, when the optimal model/hyperparameter settings are configured, all settings will be stored as a `model` file (`pickle` file). Next time when AutoML pipeline starts training, it will detect whether the `model` file exists and only fit the optimal pipeline, which can save the training time (for optimization). On test dataset Employee Future prediction, the over 3 minutes training time can be reduced to 2.1 seconds reproduction time. Fitted models, preprocessed train/test datasets, hyperparameter settings for each trials will also by stored in `tmp` folders for inspection. (Both `model` and `tmp` are changeable arguments if you prefer other names.)

### Configuration

Configuration allowed for `AutoTabular` (`AutoTabularClassifier`, `AutoTabularRegressor`) (only some common arguments here):

> 1. timeout: maximum allowed time for the tuning job.
>
> 2. max_evals: maximum allowed trials for the tuning job. (each trial is trained multiple times, whose training iterations are controlled by limitation time, `timeout` and performance improvements.)
>
> 3. allow_error_prop: maximum allowed failure errors proportion (number of allowed error = proportion * max_evals)
>
> 4. model_name: saved model names, will be used to recognized for later runs
>
> 5. encoder, imputer, balancing, scaling, feature_selection, models: `auto` (use default methods) or limit as a list of allowed methods.
>
> 6. objective: metrics use to evaluate trials' performance
>
> 7. search_algo: search algorithm, `GridSearch`, `RandomSearch` and `HyperOpt` are now supported, may seek compatibility for more search algorithms
>
> 8. cpu_threads, use_gpu: computational resources used for the job, will use all available by default

Other files in the repository:

1. `report.pdf` and presentation provides an introduction to the basic idea of AutoML pipeline and demonstrates test performance on some real-life datasets, and `Appendix` provides test datasets in the report.

2. `Dockerfiles` provides a Docker environment preparation files, you can easily build a virtual environment and test your datasets on the AutoML pipeline. The dockerfiles will install necessary packages and clone this repository to workspace.

## Future

I'm still interested to expand current AutoML package, and have plans to add MLP/RNN structure AutoML for tabular tasks/NLP tasks (it's available now for tabular tasks). Furthermore, explore the topics in explainability, fairness of ML models. But no schedules for such progress can be made.
