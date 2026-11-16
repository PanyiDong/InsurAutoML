"""
File Name: conftest.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.8
Relative Path: /tests/test_informed/conftest.py
File Created: Monday, 16th November 2026 9:51:19 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 9:56:57 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2026, Panyi Dong

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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from InsurAutoML._optional import informed_available

# the informed pipelines need the optional informed installation
# (INFORMEDAUTOML=1 pip install "InsurAutoML[informed]"), skip them otherwise
if not informed_available():
    collect_ignore_glob = ["test_*.py"]

EXAMPLE_DATA = Path(__file__).resolve().parents[2] / "example" / "example_data"


def inject_missing(
    X: pd.DataFrame, columns: list, frac: float = 0.15, seed: int = 0
) -> pd.DataFrame:
    """Set a random fraction of the given columns to NaN (deterministic)."""
    X = X.copy()
    rng = np.random.default_rng(seed)
    for column in columns:
        X.loc[rng.random(len(X)) < frac, column] = np.nan
    return X


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """Run the test inside a temporary directory, AutoML writes tmp/ and model files."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def heart_missing():
    """Classification data (HeartDisease) with missing values, 400 rows."""
    data = pd.read_csv(EXAMPLE_DATA / "heart.csv").sample(n=400, random_state=0)
    data = data.reset_index(drop=True)
    X = inject_missing(
        data.drop(columns=["HeartDisease"]), ["Cholesterol", "MaxHR", "Oldpeak"]
    )
    return X, data[["HeartDisease"]]


@pytest.fixture(scope="session")
def insurance_missing():
    """Regression data (expenses) with missing values, 400 rows."""
    data = pd.read_csv(EXAMPLE_DATA / "insurance.csv").sample(n=400, random_state=0)
    data = data.reset_index(drop=True)
    X = inject_missing(data.drop(columns=["expenses"]), ["bmi", "age"])
    return X, data[["expenses"]]


@pytest.fixture
def numeric_missing():
    """Small numerical dataset with missing values for the preparation pipelines."""
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "x4": rng.normal(size=n),
        }
    )
    y = pd.Series(2 * X["x1"] - X["x2"] + rng.normal(scale=0.1, size=n), name="y")
    X = inject_missing(X, ["x3", "x4"], frac=0.2, seed=2)
    return X, y
