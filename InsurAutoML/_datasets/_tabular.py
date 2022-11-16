"""
File Name: _tabular.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_datasets/_tabular.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 6:58:57 pm
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

from __future__ import annotations

import os
import requests
import pandas as pd
from typing import Union, Tuple, List, Dict

URL_PREFIX = "https://github.com/EpistasisLab/pmlb/raw/master/datasets"
SUFFIX = ".tsv.gz"


class BaseTabularDataset:
    def __init__(
        self,
        data_name: str,
        data_path: str = "tmp",
        download: bool = True,
    ) -> None:
        self.data_name = data_name
        self.data_path = data_path
        self.download = download

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if os.path.exists(
            os.path.join(
                self.data_path, self.data_name, self.data_name + ".csv"
            ).replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(self.data_path, self.data_name, self.data_name + ".csv")
            )
        else:
            # get the file url
            dataset_url = os.path.join(
                URL_PREFIX, self.data_name, self.data_name + SUFFIX
            ).replace("\\", "/")

            # check if the dataset exists
            re = requests.get(dataset_url)
            if re.status_code != 200:
                raise ValueError("Dataset not found.")

            # retrieve the dataset
            dataset = pd.read_csv(dataset_url, sep="\t", compression="gzip")
            if self.download:

                # make sure folder exists
                if not os.path.exists(
                    os.path.join(self.data_path, self.data_name).replace("\\", "/")
                ):
                    os.makedirs(
                        os.path.join(self.data_path, self.data_name).replace("\\", "/")
                    )

                dataset.to_csv(
                    os.path.join(
                        self.data_path, self.data_name, self.data_name + ".csv"
                    ).replace("\\", "/"),
                    index=False,
                )

        return dataset.iloc[:, :-1], dataset.iloc[:, -1]


##################################################################################################################### Tabular Classification Dataset


def ADULT(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("adult", data_path, download)

    return data_loader.get_data()


def AUTO(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("auto", data_path, download)

    return data_loader.get_data()


def BREAST(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("breast", data_path, download)

    return data_loader.get_data()


def CAR_EVALUATION(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("car_evaluation", data_path, download)

    return data_loader.get_data()


def NURSERY(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("nursery", data_path, download)

    return data_loader.get_data()


def PAGE_BLOCKS(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("page_blocks", data_path, download)

    return data_loader.get_data()


def HEART() -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_csv("Appendix/heart.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


def EMPLOYEE() -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv("Appendix/Employee.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


def STROKE() -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_csv("Appendix/healthcare-dataset-stroke-data.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


def HEART2020() -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_csv("Appendix/heart_2020_cleaned.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


def TRAVEL_INSURANCE() -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_csv("Appendix/TravelInsurancePrediction.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


def IMBALANCED_INSURANCE() -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_csv("Appendix/ImbalancedInsurance.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


def HEALTH_INSURANCE() -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_csv("Appendix/HealthInsurance.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


##################################################################################################################### Tabular Regression Dataset


def CPU_ACT(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("197_cpu_act", data_path, download)

    return data_loader.get_data()


def WIND(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("503_wind", data_path, download)

    return data_loader.get_data()


def HOUSES(
    data_path: str = "tmp", download: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_loader = BaseTabularDataset("537_houses", data_path, download)

    return data_loader.get_data()


def INSURANCE() -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_csv("Appendix/insurance.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


def MEDICAL_PREMIUM() -> Tuple[pd.DataFrame, pd.DataFrame]:

    dataset = pd.read_csv("Appendix/Medicalpremium.csv")

    return dataset.iloc[:, :-1], dataset.iloc[:, -1]


##################################################################################################################### Tabular Classification Dataset with text features


# Machine Hack Product Sentiment Dataset
def PROD(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/machine_hack_product_sentiment"
    DATANAME = "machine_hack_product_sentiment"

    # need to convert test to dev for reading
    # if string, only one split needed
    if isinstance(split, str):
        # split = "dev" if split == "test" else split
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(
                os.path.join(
                    URL, "{}.csv".format("dev" if split == "test" else split)
                ).replace("\\", "/")
            )
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )
        return dataset.iloc[:, :-1], dataset.iloc[:, -1]
    elif isinstance(split, list):
        # if "test" in split:
        #     split = list(map(lambda x: "dev" if x == "test" else x, split))

        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(
                        URL, "{}.csv".format("dev" if _split == "test" else _split)
                    ).replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (dataset[key].iloc[:, :-1], dataset[key].iloc[:, -1])
            for key in dataset.keys()
        }


# Jigsaw unintended bias in toxicity classification dataset
def JIGSAW(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/jigsaw_unintended_bias"
    DATANAME = "jigsaw_unintended_bias"

    # need to convert test to dev for reading
    # if string, only one split needed
    if isinstance(split, str):
        # split = "dev" if split == "test" else split
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
        ):
            dataset = pd.read_parquet(
                os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_parquet(
                os.path.join(
                    URL, "{}.pq".format("dev" if split == "test" else split)
                ).replace("\\", "/")
            )
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_parquet(
                    os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/"),
                    index=False,
                )
        return dataset.loc[:, dataset.columns != "target"], dataset["target"]
    elif isinstance(split, list):
        # if "test" in split:
        #     split = list(map(lambda x: "dev" if x == "test" else x, split))

        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
            ):
                dataset[_split] = pd.read_parquet(
                    os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
                )
            else:
                dataset[_split] = pd.read_parquet(
                    os.path.join(
                        URL, "{}.pq".format("dev" if _split == "test" else _split)
                    ).replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_parquet(
                        os.path.join(data_path, DATANAME, f"{_split}.pq").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "target"],
                dataset[key]["target"],
            )
            for key in dataset.keys()
        }


# Jigsaw unintended bias in toxicity classification 100K dataset
def JIGSAW100K(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/jigsaw_unintended_bias100K"
    DATANAME = "jigsaw_unintended_bias100K"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
        ):
            dataset = pd.read_parquet(
                os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_parquet(
                os.path.join(URL, f"{split}.pq").replace("\\", "/")
            )
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_parquet(
                    os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/"),
                    index=False,
                )
        return dataset.loc[:, dataset.columns != "target"], dataset["target"]
    elif isinstance(split, list):

        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
            ):
                dataset[_split] = pd.read_parquet(
                    os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
                )
            else:
                dataset[_split] = pd.read_parquet(
                    os.path.join(URL, f"{_split}.pq").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_parquet(
                        os.path.join(data_path, DATANAME, f"{_split}.pq").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "target"],
                dataset[key]["target"],
            )
            for key in dataset.keys()
        }


# Melbourne Airbnb dataset
def AIRBNB(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/airbnb_melbourne"
    DATANAME = "airbnb_melbourne"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(os.path.join(data_path, DATANAME, f"{split}.pq")):
            dataset = pd.read_parquet(
                os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_parquet(
                os.path.join(URL, f"{split}.pq").replace("\\", "/")
            )
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_parquet(
                    os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/"),
                    index=False,
                )

        return dataset.loc[:, dataset.columns != "price_label"], dataset["price_label"]
    elif isinstance(split, list):

        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
            ):
                dataset[_split] = pd.read_parquet(
                    os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
                )
            else:
                dataset[_split] = pd.read_parquet(
                    os.path.join(URL, f"{_split}.pq").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_parquet(
                        os.path.join(data_path, DATANAME, f"{_split}.pq").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "price_label"],
                dataset[key]["price_label"],
            )
            for key in dataset.keys()
        }


# IMDB Genre Prediction dataset
def IMDBGenre(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/imdb_genre_prediction"
    DATANAME = "imdb_genre_prediction"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "Genre_is_Drama"],
            dataset["Genre_is_Drama"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "Genre_is_Drama"],
                dataset[key]["Genre_is_Drama"],
            )
            for key in dataset.keys()
        }


# Fake Job Posting Prediction dataset
def FakeJob(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/fake_job_postings"
    DATANAME = "fake_job_postings"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "fraudulent"],
            dataset["fraudulent"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "fraudulent"],
                dataset[key]["fraudulent"],
            )
            for key in dataset.keys()
        }


# Fake Job Posting Prediction dataset 2
def FakeJob2(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/fake_job_postings2"
    DATANAME = "fake_job_postings2"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "fraudulent"],
            dataset["fraudulent"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "fraudulent"],
                dataset[key]["fraudulent"],
            )
            for key in dataset.keys()
        }


# Kick Starter Success Prediction dataset
def KickStarter(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/kick_starter_funding"
    DATANAME = "kick_starter_funding"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "final_status"],
            dataset["final_status"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "final_status"],
                dataset[key]["final_status"],
            )
            for key in dataset.keys()
        }


# WINE Variety Classification dataset
def WINEReview(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/wine_reviews"
    DATANAME = "wine_reviews"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "variety"],
            dataset["variety"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "variety"],
                dataset[key]["variety"],
            )
            for key in dataset.keys()
        }


# News Channel Classification dataset
def NewsChannel(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/news_channel"
    DATANAME = "news_channel"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "channel"],
            dataset["channel"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "channel"],
                dataset[key]["channel"],
            )
            for key in dataset.keys()
        }


##################################################################################################################### Tabular Regression Dataset with text features

# women clothing reviews dataset
def WomenCloth(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/women_clothing_review"
    DATANAME = "women_clothing_review"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
        ):
            dataset = pd.read_parquet(
                os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_parquet(
                os.path.join(URL, f"{split}.pq").replace("\\", "/")
            )
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_parquet(
                    os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/"),
                    index=False,
                )
        return dataset.loc[:, dataset.columns != "Rating"], dataset["Rating"]
    elif isinstance(split, list):

        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
            ):
                dataset[_split] = pd.read_parquet(
                    os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
                )
            else:
                dataset[_split] = pd.read_parquet(
                    os.path.join(URL, f"{_split}.pq").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_parquet(
                        os.path.join(data_path, DATANAME, f"{_split}.pq").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "Rating"],
                dataset[key]["Rating"],
            )
            for key in dataset.keys()
        }


# Mercari Price Suggestion dataset
def MERCARI(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/mercari_price_suggestion"
    DATANAME = "mercari_price_suggestion"

    # need to convert test to dev for reading
    # if string, only one split needed
    if isinstance(split, str):
        # split = "dev" if split == "test" else split
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
        ):
            dataset = pd.read_parquet(
                os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_parquet(
                os.path.join(
                    URL, "{}.pq".format("dev" if split == "test" else split)
                ).replace("\\", "/")
            )
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_parquet(
                    os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/"),
                    index=False,
                )
        return dataset.loc[:, dataset.columns != "price"], dataset["price"]
    elif isinstance(split, list):
        # if "test" in split:
        #     split = list(map(lambda x: "dev" if x == "test" else x, split))

        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
            ):
                dataset[_split] = pd.read_parquet(
                    os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
                )
            else:
                dataset[_split] = pd.read_parquet(
                    os.path.join(
                        URL, "{}.pq".format("dev" if _split == "test" else _split)
                    ).replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_parquet(
                        os.path.join(data_path, DATANAME, f"{_split}.pq").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "price"],
                dataset[key]["price"],
            )
            for key in dataset.keys()
        }


# Mercari Price Suggestion 100K dataset
def MERCARI100K(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/mercari_price_suggestion100K"
    DATANAME = "mercari_price_suggestion100K"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
        ):
            dataset = pd.read_parquet(
                os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_parquet(
                os.path.join(URL, f"{split}.pq").replace("\\", "/")
            )
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_parquet(
                    os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/"),
                    index=False,
                )
        return dataset.loc[:, dataset.columns != "price"], dataset["price"]
    elif isinstance(split, list):

        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
            ):
                dataset[_split] = pd.read_parquet(
                    os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
                )
            else:
                dataset[_split] = pd.read_parquet(
                    os.path.join(URL, f"{_split}.pq").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_parquet(
                        os.path.join(data_path, DATANAME, f"{_split}.pq").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "price"],
                dataset[key]["price"],
            )
            for key in dataset.keys()
        }


# Innerwear (sold by American Eagle) Price Prediction
def AE(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/ae_price_prediction"
    DATANAME = "ae_price_prediction"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
        ):
            dataset = pd.read_parquet(
                os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_parquet(
                os.path.join(URL, f"{split}.pq").replace("\\", "/")
            )
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_parquet(
                    os.path.join(data_path, DATANAME, f"{split}.pq").replace("\\", "/"),
                    index=False,
                )
        return dataset.loc[:, dataset.columns != "price"], dataset["price"]
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
            ):
                dataset[_split] = pd.read_parquet(
                    os.path.join(data_path, DATANAME, f"{_split}.pq").replace("\\", "/")
                )
            else:
                dataset[_split] = pd.read_parquet(
                    os.path.join(URL, f"{_split}.pq").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_parquet(
                        os.path.join(data_path, DATANAME, f"{_split}.pq"), index=False
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "price"],
                dataset[key]["price"],
            )
            for key in dataset.keys()
        }


# JC Penney Price Prediction dataset
def JCPenney(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/jc_penney_products"
    DATANAME = "jc_penney_products"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "sale_price"],
            dataset["sale_price"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "sale_price"],
                dataset[key]["sale_price"],
            )
            for key in dataset.keys()
        }


# News Popularity Regression dataset
def NewsPopularity(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/news_popularity"
    DATANAME = "news_popularity"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "log_shares"],
            dataset["log_shares"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "log_shares"],
                dataset[key]["log_shares"],
            )
            for key in dataset.keys()
        }


# News Popularity Regression dataset 2
def NewsPopularity2(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/news_popularity2"
    DATANAME = "news_popularity2"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "log_shares"],
            dataset["log_shares"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "log_shares"],
                dataset[key]["log_shares"],
            )
            for key in dataset.keys()
        }


# Book Price Regression dataset
def BookPrice(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books"
    DATANAME = "predict_the_price_of_books"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "Price"],
            dataset["Price"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "Price"],
                dataset[key]["Price"],
            )
            for key in dataset.keys()
        }


# Data Science Salary Classification dataset
def DSSalary(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_data_scientists_salary_in_india_hackathon"
    DATANAME = "predict_the_data_scientists_salary"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "salary"],
            dataset["salary"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "salary"],
                dataset[key]["salary"],
            )
            for key in dataset.keys()
        }


# California House Price dataset
def CAHousePrice(
    split: Union[str, List[str]] = "train", save: bool = True, data_path: str = "tmp"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:

    URL = "https://automl-mm-bench.s3.amazonaws.com/kaggle-california-house-prices"
    DATANAME = "california_house_prices"

    # if string, only one split needed
    if isinstance(split, str):
        # if file exists, read it
        if os.path.exists(
            os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
        ):
            dataset = pd.read_csv(
                os.path.join(data_path, DATANAME, f"{split}.csv").replace("\\", "/")
            )
        # else, download it
        else:
            dataset = pd.read_csv(os.path.join(URL, f"{split}.csv").replace("\\", "/"))
            # whether need to save
            if save:
                # if folder not exists, create it
                if not os.path.exists(
                    os.path.join(data_path, DATANAME).replace("\\", "/")
                ):
                    os.makedirs(os.path.join(data_path, DATANAME).replace("\\", "/"))
                dataset.to_csv(
                    os.path.join(data_path, DATANAME, f"{split}.csv").replace(
                        "\\", "/"
                    ),
                    index=False,
                )

        return (
            dataset.loc[:, dataset.columns != "Sold Price"],
            dataset["Sold Price"],
        )
    elif isinstance(split, list):
        dataset = {}
        # if file exists, read it
        for _split in split:
            if os.path.exists(
                os.path.join(data_path, DATANAME, f"{_split}.csv").replace("\\", "/")
            ):
                dataset[_split] = pd.read_csv(
                    os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                        "\\", "/"
                    )
                )
            else:
                dataset[_split] = pd.read_csv(
                    os.path.join(URL, f"{_split}.csv").replace("\\", "/")
                )
                # whether need to save
                if save:
                    # if folder not exists, create it
                    if not os.path.exists(
                        os.path.join(data_path, DATANAME).replace("\\", "/")
                    ):
                        os.makedirs(
                            os.path.join(data_path, DATANAME).replace("\\", "/")
                        )
                    dataset[_split].to_csv(
                        os.path.join(data_path, DATANAME, f"{_split}.csv").replace(
                            "\\", "/"
                        ),
                        index=False,
                    )
        return {
            key: (
                dataset[key].loc[:, dataset[key].columns != "Sold Price"],
                dataset[key]["Sold Price"],
            )
            for key in dataset.keys()
        }
