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

import os
import warnings
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from ...encoding import DataEncoding
from ...prep import CompletePrepPipeline, MissingPrepPipeline
from ..ML import AutoTabular
from ..base import AutoTabularBase
from ...utils.base import type_of_task
from ...utils.file import load_methods, save_methods

# fitted state of the wrapper, saved together (the models with the preprocessing
# they were trained on)
FITTED_STATE = (
    "task_mode",
    "encoder",
    "complete_prep",
    "missing_prep",
    "complete_model",
    "missing_model",
)


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

    Parameters
    ----------
    seed: seed of the preparation pipelines (feature selection tie breaking,
    MissForest, SPlit) and of the two AutoTabular runs, default = None (random)
    the same seed gives the same preparation (splits) and search

    save: whether to save the fitted wrapper (preparation pipelines and models
    together) as `model_name`, default = True
    when the file exists, fit loads it instead of fitting again (as AutoTabular),
    so the models are always used with the preparation they were trained on

    other keyword arguments are passed to both AutoTabular
    """

    def __init__(
        self,
        model_name="model",
        max_evals: int = 32,
        timeout: int = 360,
        seed: Optional[int] = None,
        save: bool = True,
        **auto_kwargs,
    ):
        self.model_name = model_name
        self.max_evals = int(max_evals)
        self.timeout = timeout
        self.seed = seed
        self.save = save
        # kwargs forwarded to AutoTabular constructor (encoder, models, objective etc.)
        self.auto_kwargs = auto_kwargs
        # keep fitted models
        self.complete_model: Optional[AutoTabular] = None
        self.missing_model: Optional[AutoTabular] = None
        self.complete_prep: Optional[CompletePrepPipeline] = None
        self.missing_prep: Optional[MissingPrepPipeline] = None
        self.encoder = DataEncoding(dummy_coding=False)
        self.task_mode = None
        self._fitted = False

    def _auto_tabular(self, model_name: str) -> AutoTabular:
        """AutoTabular of a subset (complete/missing cases)."""
        kwargs: Dict[str, Any] = dict(
            # a new fit searches again: the experiments of a previous fit (other
            # preparation) are not restored
            resume=False,
        )
        kwargs.update(self.auto_kwargs)
        kwargs.update(
            model_name=model_name,
            max_evals=self.max_evals,
            timeout=self.timeout,
            validation="STAT",
            # encoder=["NoEncoding"],
            imputer=["no_processing"],
            balancing=["no_processing"],
            scaling=["no_processing"],
            feature_selection=["no_processing"],
            seed=self.seed,
            # the wrapper saves the models with their preparation: a model saved
            # alone would be reloaded with a new preparation in the next fit
            save=False,
        )
        return AutoTabular(**kwargs)

    def _fingerprint(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Data and settings of a fit (stored with the fitted wrapper)."""
        settings = {"seed": repr(self.seed)}
        settings.update({key: repr(value) for key, value in self.auto_kwargs.items()})
        return {
            "data": [AutoTabularBase._data_hash(X), AutoTabularBase._data_hash(y)],
            "settings": settings,
        }

    def _state(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in FITTED_STATE}

    def _load(self, fingerprint: Dict[str, Any]) -> bool:
        """
        Load the saved fitted wrapper if it was fitted on the same data with the
        same settings, False otherwise.
        """
        if not (self.save and os.path.isfile(self.model_name)):
            return False
        state = load_methods(self.model_name)[0]
        if not isinstance(state, dict) or not set(FITTED_STATE) <= set(state):
            raise ValueError(
                "{} is not a saved InformedAutoTabular, remove it or use another "
                "model_name.".format(self.model_name)
            )
        # wrappers saved by previous versions have no fingerprint: loaded as before
        if state.get("fingerprint", fingerprint) != fingerprint:
            warnings.warn(
                "Stored model {} was fitted on other data or settings, fitting "
                "again (it will be replaced).".format(self.model_name)
            )
            return False
        for name in FITTED_STATE:
            setattr(self, name, state[name])
        self._fitted = True
        return True

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # saved fitted wrapper: reuse it (models with their preparation)
        fingerprint = self._fingerprint(X, y)
        if self._load(fingerprint):
            return self

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
        self.complete_prep = CompletePrepPipeline(seed=self.seed)

        (
            X_train_complete,
            X_valid_complete,
            y_train_complete,
            y_valid_complete,
        ) = self.complete_prep.fit(X_enc, y)

        # initialize and fit AutoTabular on complete set
        self.complete_model = None
        if len(X_train_complete) > 0:
            self.complete_model = self._auto_tabular(self.model_name + "_complete")
            self.complete_model.fit(
                X_train_complete, y_train_complete, X_valid_complete, y_valid_complete
            )

        # initialize and fit AutoTabular on missing set (skip if no missing data)
        self.missing_prep = None
        self.missing_model = None
        if X_enc.isnull().values.any():
            self.missing_prep = MissingPrepPipeline(seed=self.seed)
            (
                X_train_missing,
                X_valid_missing,
                y_train_missing,
                y_valid_missing,
            ) = self.missing_prep.fit(X_enc, y)

            self.missing_model = self._auto_tabular(self.model_name + "_missing")
            self.missing_model.fit(
                X_train_missing, y_train_missing, X_valid_missing, y_valid_missing
            )

        self._fitted = True
        if self.save:
            save_methods(
                self.model_name, [{**self._state(), "fingerprint": fingerprint}]
            )

        return self

    def predict(self, X: pd.DataFrame):
        # ensure dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # If encoder exists, refit/transform using same encoder
        X_enc = self.encoder.refit(X)

        X_complete, _ = self.complete_prep.transform(X_enc, y=None)
        pred_complete = None
        if len(X_complete) > 0 and self.complete_model is not None:
            pred_complete = pd.DataFrame(self.complete_model.predict(X_complete))
            pred_complete.index = X_complete.index

        # no missing data: predictions of the complete-case model
        if not X_enc.isnull().values.any():
            return pred_complete

        if self.missing_model is None:
            raise ValueError(
                "No model fitted on missing data, but X has missing values."
            )
        X_missing, _ = self.missing_prep.transform(X_enc, y=None)
        pred = pd.DataFrame(self.missing_model.predict(X_missing))
        pred.index = X_missing.index

        # use averaging strategy to combine predictions
        # if complete predictions exist, average them in
        if pred_complete is not None:
            pred.loc[X_complete.index, :] += pred_complete
            pred.loc[X_complete.index, :] /= 2
        return pred
