"""
File Name: _eda.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_eda.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 9:24:20 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2022, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Union
import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

from InsurAutoML._utils._data import feature_type


def EDA(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, np.ndarray] = None,
    plot: bool = True,
    save: bool = True,
    path: str = "tmp/EDA",
    skip: bool = True,
) -> None:

    # Check if the path exists
    if not os.path.exists(path):
        os.makedirs(path)
    # if the path exists, clean it
    else:
        files = glob.glob(path + "/*")
        for f in files:
            os.remove(f)

    # get column types
    data_type = {}

    for column in X.columns:
        data_type[column] = feature_type(X[column])

    # if y is not None, add it to the data_type
    if y is not None:
        if isinstance(y, pd.DataFrame):
            response = y.columns[0]
            y = y.values
        elif isinstance(y, pd.Series):
            response = y.name
        elif isinstance(y, list) or isinstance(y, np.ndarray):
            response = "response"
        data_type[response] = feature_type(y)

    # print for examination
    print("Data Column Types:")
    for column, type in zip(data_type.keys(), data_type.values()):
        print("{:40s}: {:>25s}".format(column, type))

    # save the data types
    if save:
        pd.DataFrame.from_dict(data_type, orient="index").to_csv(
            os.path.join(path, "data_type.csv"), index=False
        )

    # check whether to skip plotting process
    if len(X.columns) > 100 and skip:
        warnings.warn(
            "Too many columns to plot. Skip plotting. If insist, please set skip=False."
        )
        return None
    if len(X.columns) > 100 and not skip:
        warnings.warn("Too many columns to plot. But will continue.")

    # plot the data types
    # if y is None, plot histogram for each column
    if y is None:
        for column in X.columns:
            # plt.figure(figsize = (10, 5))
            if data_type[column] == "continuous":
                # summary statistics
                print("Summary Statistics for Column: {}".format(column))
                summary = X[column].describe()
                print(summary)
                # write mode for summary
                write_mode = "w" if not os.path.exists(path + "/summary.txt") else "a"
                if save:
                    with open(path + "/summary.txt", write_mode) as f:
                        f.write("Summary Statistics for Column: {}\n".format(column))
                        f.write(str(summary) + "\n\n")
                # plot the histogram
                plt.tight_layout()
                sns.histplot(X[column], kde=True, color="blue")
                plt.xlabel("{}".format(column))
                plt.ylabel("Frequency")
                plt.title("Histogram of Column {}".format(column))
                if save:
                    plt.savefig(os.path.join(path, "plot_{}.png".format(column)))
                if plot:
                    plt.show()
            if data_type[column] in ["string_categorical", "numerical_categorical"]:
                # summary statistics
                print("Summary Statistics for Column: {}".format(column))
                summary = X[column].value_counts()
                print(summary)
                # write mode for summary
                write_mode = "w" if not os.path.exists(path + "/summary.txt") else "a"
                if save:
                    with open(path + "/summary.txt", write_mode) as f:
                        f.write("Summary Statistics for Column: {}\n".format(column))
                        f.write(str(summary) + "\n\n")
                # plot the histogram
                plt.tight_layout()
                sns.histplot(X[column], kde=False, color="blue")
                plt.xlabel("{}".format(column))
                plt.ylabel("Frequency")
                plt.title("Histogram of Column {}".format(column))
                if save:
                    plt.savefig(os.path.join(path, "plot_{}.png".format(column)))
                if plot:
                    plt.show()
            elif data_type[column] == "text":
                print("Column {} is text. Skip plotting.".format(column))

            plt.close()
    # if y is not None, plot boxplot for each column with respect to y
    else:
        for column in X.columns:
            if data_type[column] == "continuous":
                # summary statistics
                print("Summary Statistics for Column: {}".format(column))
                summary = X[column].describe()
                print(summary)
                # write mode for summary
                write_mode = "w" if not os.path.exists(path + "/summary.txt") else "a"
                if save:
                    with open(path + "/summary.txt", write_mode) as f:
                        f.write("Summary Statistics for Column: {}\n".format(column))
                        f.write(str(summary) + "\n\n")
                # plot the histogram
                fig, (ax1, ax2) = plt.subplots(1, 2)
                plt.tight_layout()
                sns.histplot(X[column], ax=ax1, kde=True, color="blue")
                ax1.set_xlabel("{}".format(column))
                ax1.set_ylabel("Frequency")
                ax1.set_title("Histogram of Column {}".format(column))
                ax2.scatter(X[column], y, color="blue")
                linear_model = LinearRegression()
                linear_model.fit(X[column].values.reshape(-1, 1), y)
                y_pred = linear_model.predict(X[column].values.reshape(-1, 1))
                ax2.plot(X[column], y_pred, color="red", label="linear interpolation")
                ax2.set_xlabel("{}".format(column))
                ax2.set_ylabel("{}".format(response))
                ax2.set_title("{} ~ Column {}".format(response, column))
                plt.legend()
                if save:
                    plt.savefig(os.path.join(path, "plot_{}.png".format(column)))
                if plot:
                    plt.show()
            elif data_type[column] in ["string_categorical", "numerical_categorical"]:
                # summary statistics
                print("Summary Statistics for Column: {}".format(column))
                summary = X[column].value_counts()
                print(summary)
                # write mode for summary
                write_mode = "w" if not os.path.exists(path + "/summary.txt") else "a"
                if save:
                    with open(path + "/summary.txt", write_mode) as f:
                        f.write("Summary Statistics for Column: {}\n".format(column))
                        f.write(str(summary) + "\n\n")
                # plot the boxplot
                plt.tight_layout()
                sns.stripplot(x=X[column], y=y)
                sns.boxplot(x=X[column], y=y)
                plt.xlabel("{}".format(column))
                plt.ylabel("{}".format(response))
                plt.title("BoxPlot of {} ~ Column {}".format(response, column))
                if save:
                    plt.savefig(os.path.join(path, "plot_{}.png".format(column)))
                if plot:
                    plt.show()
            elif data_type[column] == "text":
                print("Column {} is text. Skip plotting.".format(column))

            plt.close()
