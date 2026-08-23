"""
File Name: fixed.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/hpo/informed/fixed.py
File Created: Wednesday, 3rd December 2025 1:38:19 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 28th December 2025 10:12:15 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025, Panyi Dong

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

from typing import Optional
import pandas as pd
import numpy as np

from ...encoding import DataEncoding
from ...prep import CompletePrepPipeline, MissingPrepPipeline
from ..ML import AutoTabular
from ...utils.base import type_of_task


class InformedAutoTabular:
    """Simple informed AutoTabular wrapper.

    Workflow:
    - Use `DataEncoding(dummy_coding=False)` to encode features.
    - Use `CompletePrepPipeline` and `MissingPrepPipeline` to derive complete/missing
      train/validation splits.
    - Run two separate `AutoTabular` HPO runs (one on the complete-set, one on the
      missing-set) with half of the `max_evals` budget each.
    - Store the two fitted AutoTabular models and use them jointly for prediction
      (average for regression, average predicted probabilities then argmax for
      classification).

    This class is intentionally light-weight and meant for quick experimentation.
    """

    def __init__(
        self, model_name="model", max_evals: int = 32, timeout: int = 360, **auto_kwargs
    ):
        self.model_name = model_name
        self.max_evals = int(max_evals)
        self.timeout = timeout
        # kwargs forwarded to AutoTabular constructor (encoder, models, objective etc.)
        self.auto_kwargs = auto_kwargs
        # keep fitted models
        self.complete_model: Optional[AutoTabular] = None
        self.missing_model: Optional[AutoTabular] = None
        self.encoder = DataEncoding(dummy_coding=False)
        self.task_mode = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # basic checks
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # determine task type
        self.task_mode = type_of_task(y)

        # encode features (use dummy_coding=False as requested)
        X_enc = self.encoder.fit(X)

        # prepare complete & missing datasets
        self.complete_prep = CompletePrepPipeline()

        X_train_complete, X_valid_complete, y_train_complete, y_valid_complete = (
            self.complete_prep.fit(X_enc, y)
        )
        # skip if no missing data
        if X_enc.isnull().sum().sum() != 0:
            self.missing_prep = MissingPrepPipeline()

            X_train_missing, X_valid_missing, y_train_missing, y_valid_missing = (
                self.missing_prep.fit(X_enc, y)
            )

        # initialize and fit AutoTabular on complete set
        if len(X_train_complete) > 0:
            self.complete_model = AutoTabular(
                model_name=self.model_name + "_complete",
                max_evals=self.max_evals,
                timeout=self.timeout,
                validation="STAT",
                # encoder=["NoEncoding"],
                imputer=["no_processing"],
                balancing=["no_processing"],
                scaling=["no_processing"],
                feature_selection=["no_processing"],
                **self.auto_kwargs
            )
            self.complete_model.fit(
                X_train_complete, y_train_complete, X_valid_complete, y_valid_complete
            )

        # initialize and fit AutoTabular on missing set
        if X_enc.isnull().sum().sum() != 0:
            self.missing_model = AutoTabular(
                model_name=self.model_name + "_missing",
                max_evals=self.max_evals,
                timeout=self.timeout,
                validation="STAT",
                # encoder=["NoEncoding"],
                imputer=["no_processing"],
                balancing=["no_processing"],
                scaling=["no_processing"],
                feature_selection=["no_processing"],
                **self.auto_kwargs
            )
            self.missing_model.fit(
                X_train_missing, y_train_missing, X_valid_missing, y_valid_missing
            )

        return self

    def predict(self, X: pd.DataFrame):
        # ensure dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # If encoder exists, refit/transform using same encoder
        X_enc = self.encoder.refit(X)

        X_complete, _ = self.complete_prep.transform(X_enc, y=None)
        complete_idx = X_complete.index
        if X_enc.isnull().sum().sum() != 0:
            X_missing, _ = self.missing_prep.transform(X_enc, y=None)
            missing_idx = X_missing.index

        if len(X_complete) > 0 and self.complete_model is not None:
            pred_complete = self.complete_model.predict(X_complete)
            pred_complete.index = complete_idx

        if X_enc.isnull().sum().sum() != 0:
            pred_missing = self.missing_model.predict(X_missing)
            pred_missing.index = missing_idx

        # use averaging strategy to combine predictions
        if X_enc.isnull().sum().sum() == 0:
            return pred_complete
        else:
            pred = pred_missing.copy()
            # if complete predictions exist, average them in
            if len(X_complete) > 0 and self.complete_model is not None:
                pred.loc[complete_idx, :] += pred_complete
                pred.loc[complete_idx, :] /= 2
            return pred
