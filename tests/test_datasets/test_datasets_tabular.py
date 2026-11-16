"""
File Name: test_datasets_tabular.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_datasets/test_datasets_tabular.py
File Created: Monday, 16th November 2026 12:00:00 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 12:00:00 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2026, Panyi Dong

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

# All remote reads (requests.get / pd.read_csv / pd.read_parquet on URLs) are
# mocked; no network access is needed.

import gzip
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from InsurAutoML.datasets import tabular

REPO_ROOT = Path(__file__).resolve().parents[2]

_ORIG_READ_CSV = pd.read_csv
_ORIG_READ_PARQUET = pd.read_parquet


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def _fake_frame(target, n=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num": rng.integers(0, 100, size=n),
            "txt": ["text number {}".format(i) for i in range(n)],
            target: rng.integers(0, 2, size=n),
        }
    )


def _gzip_tsv(df):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(df.to_csv(sep="\t", index=False).encode())
    return buf.getvalue()


@pytest.fixture
def remote(monkeypatch):
    """
    Mock remote access. Return a dict recording the requested urls and allowing
    the test to define the served frame / status code.
    """
    state = {"urls": [], "frame": None, "status": 200, "gz": None}

    def fake_get(url, *args, **kwargs):
        state["urls"].append(("get", url))
        return _FakeResponse(state["status"])

    def fake_read_csv(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("https://"):
            state["urls"].append(("csv", path))
            if kwargs.get("compression") == "gzip":
                # decode real gzip-compressed TSV bytes
                return _ORIG_READ_CSV(io.BytesIO(state["gz"]), *args, **kwargs)
            return state["frame"].copy()
        return _ORIG_READ_CSV(path, *args, **kwargs)

    def fake_read_parquet(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("https://"):
            state["urls"].append(("pq", path))
            return state["frame"].copy()
        return _ORIG_READ_PARQUET(path, *args, **kwargs)

    monkeypatch.setattr(tabular.requests, "get", fake_get)
    monkeypatch.setattr(tabular.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(tabular.pd, "read_parquet", fake_read_parquet)

    return state


##########################################################################
# PMLB datasets (BaseTabularDataset)

PMLB_LOADERS = [
    (tabular.ADULT, "adult"),
    (tabular.AUTO, "auto"),
    (tabular.BREAST, "breast"),
    (tabular.CAR_EVALUATION, "car_evaluation"),
    (tabular.NURSERY, "nursery"),
    (tabular.PAGE_BLOCKS, "page_blocks"),
    (tabular.CPU_ACT, "197_cpu_act"),
    (tabular.WIND, "503_wind"),
    (tabular.HOUSES, "537_houses"),
]


@pytest.mark.parametrize("loader, name", PMLB_LOADERS)
def test_pmlb_download_and_cache(loader, name, remote, tmp_path):
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [5, 6, 7, 8], "target": [0, 1, 0, 1]}
    )
    remote["gz"] = _gzip_tsv(frame)

    X, y = loader(data_path=str(tmp_path), download=True)

    # url requested follows PMLB layout
    expected_url = "{}/{}/{}.tsv.gz".format(tabular.URL_PREFIX, name, name)
    assert ("get", expected_url) in remote["urls"]
    assert ("csv", expected_url) in remote["urls"]

    # split into features and last column response
    assert list(X.columns) == ["a", "b"]
    assert X.shape == (4, 2)
    assert y.name == "target"
    assert y.tolist() == [0, 1, 0, 1]

    # file cached on disk
    cached = tmp_path / name / (name + ".csv")
    assert cached.exists()
    pd.testing.assert_frame_equal(_ORIG_READ_CSV(cached), frame)

    # second call reads from the cache, no further remote access
    n_calls = len(remote["urls"])
    X2, y2 = loader(data_path=str(tmp_path), download=True)
    assert len(remote["urls"]) == n_calls
    pd.testing.assert_frame_equal(X2, X)
    pd.testing.assert_series_equal(y2, y)


def test_pmlb_no_download(remote, tmp_path):
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
    remote["gz"] = _gzip_tsv(frame)

    X, y = tabular.BaseTabularDataset("adult", str(tmp_path), False).get_data()

    assert X.shape == (3, 1)
    assert y.tolist() == [3, 2, 1]
    # nothing written to disk
    assert not (tmp_path / "adult").exists()


def test_pmlb_not_found(remote, tmp_path):
    remote["status"] = 404

    with pytest.raises(ValueError, match="Dataset not found"):
        tabular.ADULT(data_path=str(tmp_path))

    assert not (tmp_path / "adult").exists()


##########################################################################
# Local csv datasets


LOCAL_LOADERS = [
    (tabular.HEART, "heart.csv"),
    (tabular.EMPLOYEE, "Employee.csv"),
    (tabular.STROKE, "healthcare-dataset-stroke-data.csv"),
    (tabular.HEART2020, "heart_2020_cleaned.csv"),
    (tabular.TRAVEL_INSURANCE, "TravelInsurancePrediction.csv"),
    (tabular.IMBALANCED_INSURANCE, "ImbalancedInsurance.csv"),
    (tabular.HEALTH_INSURANCE, "HealthInsurance.csv"),
    (tabular.INSURANCE, "insurance.csv"),
    (tabular.MEDICAL_PREMIUM, "Medicalpremium.csv"),
]


@pytest.mark.parametrize("loader, filename", LOCAL_LOADERS)
def test_local_loaders(loader, filename, monkeypatch, tmp_path):
    src = REPO_ROOT / "data" / filename
    if not src.exists():
        pytest.skip("{} not available".format(filename))

    # loaders read relative path data/<file>; create it in tmp cwd with a
    # small slice of the real file
    full = _ORIG_READ_CSV(src, nrows=50)
    (tmp_path / "data").mkdir()
    full.to_csv(tmp_path / "data" / filename, index=False)
    monkeypatch.chdir(tmp_path)

    X, y = loader()

    assert X.shape == (len(full), full.shape[1] - 1)
    assert list(X.columns) == list(full.columns[:-1])
    assert y.name == full.columns[-1]
    np.testing.assert_array_equal(y.values, full.iloc[:, -1].values)


##########################################################################
# Multimodal (text) datasets

# (loader, target column (None -> last column), file format, test -> dev)
MM_LOADERS = [
    (tabular.PROD, None, "csv", "machine_hack_product_sentiment", True),
    (tabular.JIGSAW, "target", "pq", "jigsaw_unintended_bias", True),
    (tabular.JIGSAW100K, "target", "pq", "jigsaw_unintended_bias100K", False),
    (tabular.AIRBNB, "price_label", "pq", "airbnb_melbourne", False),
    (tabular.IMDBGenre, "Genre_is_Drama", "csv", "imdb_genre_prediction", False),
    (tabular.FakeJob, "fraudulent", "csv", "fake_job_postings", False),
    (tabular.FakeJob2, "fraudulent", "csv", "fake_job_postings2", False),
    (tabular.KickStarter, "final_status", "csv", "kick_starter_funding", False),
    (tabular.WINEReview, "variety", "csv", "wine_reviews", False),
    (tabular.NewsChannel, "channel", "csv", "news_channel", False),
    (tabular.WomenCloth, "Rating", "pq", "women_clothing_review", False),
    (tabular.MERCARI, "price", "pq", "mercari_price_suggestion", True),
    (tabular.MERCARI100K, "price", "pq", "mercari_price_suggestion100K", False),
    (tabular.AE, "price", "pq", "ae_price_prediction", False),
    (tabular.JCPenney, "sale_price", "csv", "jc_penney_products", False),
    (tabular.NewsPopularity, "log_shares", "csv", "news_popularity", False),
    (tabular.NewsPopularity2, "log_shares", "csv", "news_popularity2", False),
    (tabular.BookPrice, "Price", "csv", "predict_the_price_of_books", False),
    (tabular.DSSalary, "salary", "csv", "predict_the_data_scientists_salary", False),
    (tabular.CAHousePrice, "Sold Price", "csv", "california_house_prices", False),
]

MM_IDS = [item[0].__name__ for item in MM_LOADERS]


def _check_xy(X, y, frame, target):
    if target is None:
        target = frame.columns[-1]
    assert target not in X.columns
    assert X.shape == (len(frame), frame.shape[1] - 1)
    assert y.name == target
    np.testing.assert_array_equal(y.values, frame[target].values)


@pytest.mark.parametrize(
    "loader, target, fmt, dataname, to_dev", MM_LOADERS, ids=MM_IDS
)
def test_multimodal_single_split(
    loader, target, fmt, dataname, to_dev, remote, tmp_path
):
    # put the response column in the middle so name-based selection matters
    frame = _fake_frame(target or "label")
    frame = frame[[frame.columns[0], frame.columns[2], frame.columns[1]]]
    if target is None:
        frame = _fake_frame("label")
    remote["frame"] = frame

    X, y = loader(split="test", save=True, data_path=str(tmp_path))
    _check_xy(X, y, frame, target)

    # url requested (test is mapped to dev for some datasets)
    kind, url = remote["urls"][-1]
    assert kind == fmt
    assert url.endswith("/{}.{}".format("dev" if to_dev else "test", fmt))

    # file cached on disk as test.<fmt>
    cached = tmp_path / dataname / "test.{}".format(fmt)
    assert cached.exists()

    # second call uses the cache
    n_calls = len(remote["urls"])
    X2, y2 = loader(split="test", save=True, data_path=str(tmp_path))
    assert len(remote["urls"]) == n_calls
    _check_xy(X2, y2, frame, target)


@pytest.mark.parametrize(
    "loader, target, fmt, dataname, to_dev", MM_LOADERS, ids=MM_IDS
)
def test_multimodal_list_split(loader, target, fmt, dataname, to_dev, remote, tmp_path):
    frame = _fake_frame(target or "label", seed=1)
    remote["frame"] = frame

    result = loader(split=["train", "test"], save=True, data_path=str(tmp_path))

    assert isinstance(result, dict)
    assert set(result.keys()) == {"train", "test"}
    for key in ["train", "test"]:
        X, y = result[key]
        _check_xy(X, y, frame, target)
        assert (tmp_path / dataname / "{}.{}".format(key, fmt)).exists()

    requested = [url for kind, url in remote["urls"]]
    assert any(url.endswith("/train.{}".format(fmt)) for url in requested)
    assert any(
        url.endswith("/{}.{}".format("dev" if to_dev else "test", fmt))
        for url in requested
    )

    # cached on second call
    n_calls = len(remote["urls"])
    result2 = loader(split=["train", "test"], save=True, data_path=str(tmp_path))
    assert len(remote["urls"]) == n_calls
    for key in ["train", "test"]:
        _check_xy(*result2[key], frame, target)


@pytest.mark.parametrize(
    "loader, target, fmt, dataname, to_dev", MM_LOADERS, ids=MM_IDS
)
def test_multimodal_no_save(loader, target, fmt, dataname, to_dev, remote, tmp_path):
    frame = _fake_frame(target or "label", seed=2)
    remote["frame"] = frame

    X, y = loader(split="train", save=False, data_path=str(tmp_path))
    _check_xy(X, y, frame, target)

    result = loader(split=["train"], save=False, data_path=str(tmp_path))
    _check_xy(*result["train"], frame, target)

    # nothing was cached
    assert not (tmp_path / dataname).exists()
    assert not os.listdir(tmp_path)
