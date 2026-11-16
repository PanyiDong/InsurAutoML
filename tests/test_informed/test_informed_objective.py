"""
File: test_informed_objective.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Relative Path: /tests/test_informed/test_informed_objective.py

InformedTabularObjective (the Ray Tune trainable of the informed pipelines) run
in-process: one trial = encoding, complete/missing data preparation, two models and
the averaged validation loss.
"""

import func_timeout
import numpy as np
import pytest

from InsurAutoML.encoding import DataEncoding
from InsurAutoML.hpo.informed import utils as informed_utils
from InsurAutoML.hpo.informed.utils import InformedTabularObjective
from InsurAutoML.model import LinearRegression, LogisticRegression
from InsurAutoML.prep import CompletePrepPipeline, MissingPrepPipeline

PREP = {
    "encoder": {"encoder": "DataEncoding", "dummy_coding": True},
    "complete_prep": {
        "complete_prep": "CompletePrepPipeline",
        "missing_threshold": 0.5,
        "cc_threshold": 0.9,
    },
    "missing_prep": {
        "missing_prep": "MissingPrepPipeline",
        "missing_threshold": 0.8,
        "imputation_max_iter": 2,
        "cc_threshold": 0.9,
    },
}

COMPONENTS = dict(
    encoder={"DataEncoding": DataEncoding},
    complete_prep={"CompletePrepPipeline": CompletePrepPipeline},
    missing_prep={"MissingPrepPipeline": MissingPrepPipeline},
)

STATUS_KEYS = {
    "encoder",
    "complete_prep",
    "missing_prep",
    "model",
    "model_hyperparameter",
}


def _objective(
    config, data, models, task_mode, objective, full_status=False, timeout=120
):
    trainable = InformedTabularObjective.__new__(InformedTabularObjective)
    trainable.setup(
        dict(config),
        data_split=data,
        models=models,
        model_name="informed_obj",
        task_mode=task_mode,
        objective=objective,
        full_status=full_status,
        timeout=timeout,
        seed=1,
        **COMPONENTS,
    )
    return trainable


@pytest.fixture(scope="module")
def insurance(insurance_missing):
    X, y = insurance_missing
    return X.iloc[:200].reset_index(drop=True), y.iloc[:200].reset_index(drop=True)


@pytest.fixture(scope="module")
def heart(heart_missing):
    X, y = heart_missing
    return X.iloc[:200].reset_index(drop=True), y.iloc[:200].reset_index(drop=True)


def test_regression_objective(in_tmp_dir, insurance):
    config = {**PREP, "model": {"model": "LinearRegression"}}
    trainable = _objective(
        config, insurance, {"LinearRegression": LinearRegression}, "regression", "MAE"
    )

    result = trainable.step()

    assert result["training_status"] == "FITTED"
    assert result["fitted_model"] == "LinearRegression"
    assert np.isfinite(result["loss"]) and result["loss"] > 0
    # the fitted models are saved in the trial directory (cwd outside of Ray Tune)
    assert (in_tmp_dir / "informed_obj").exists()


@pytest.mark.parametrize("objective", ["accuracy", "auc"])
def test_classification_objective_full_status(in_tmp_dir, heart, objective):
    config = {
        **PREP,
        "model": {
            "model": "LogisticRegression",
            "penalty": "l2",
            "C": 1.0,
            "tol": 1e-4,
        },
    }
    trainable = _objective(
        config,
        heart,
        {"LogisticRegression": LogisticRegression},
        "classification",
        objective,
        full_status=True,
    )

    result = trainable.step()

    assert result["training_status"] == "FITTED"
    # accuracy/auc are maximized, the loss is their negative
    assert -1.0 <= result["loss"] < 0
    assert STATUS_KEYS <= set(result)
    assert result["model_hyperparameter"]["C"] == 1.0


def test_r2_objective_is_negated(in_tmp_dir, insurance):
    config = {**PREP, "model": {"model": "LinearRegression"}}
    trainable = _objective(
        config, insurance, {"LinearRegression": LinearRegression}, "regression", "R2"
    )

    # R2 is maximized, the objective returns its negative
    objective = trainable._get_objective()
    assert objective(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == -1.0


def test_timeout(in_tmp_dir, insurance, monkeypatch):
    config = {**PREP, "model": {"model": "LinearRegression"}}
    trainable = _objective(
        config,
        insurance,
        {"LinearRegression": LinearRegression},
        "regression",
        "MAE",
        full_status=True,
    )

    def _timed_out(timeout, func, *args, **kwargs):
        raise func_timeout.FunctionTimedOut()

    monkeypatch.setattr(informed_utils.func_timeout, "func_timeout", _timed_out)
    result = trainable.step()

    assert result["training_status"] == "TIMEOUT"
    assert result["loss"] == np.inf
    assert STATUS_KEYS <= set(result)


def test_reset_config_and_checkpoint(in_tmp_dir, insurance, tmp_path):
    models = {
        "LinearRegression": LinearRegression,
        "LogisticRegression": LogisticRegression,
    }
    trainable = _objective(
        {**PREP, "model": {"model": "LinearRegression"}},
        insurance,
        models,
        "regression",
        "MAE",
    )
    result = trainable.step()

    # the checkpoint stores the last status
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    # None: Ray Tune uses the content of the directory as the checkpoint
    assert trainable.save_checkpoint(str(checkpoint_dir)) is None
    assert (checkpoint_dir / "status.json").exists()

    # reused actor: a new configuration replaces the pipeline components and
    # clears the status of the previous trial
    assert trainable.reset_config(
        {**PREP, "model": {"model": "LogisticRegression", "C": 0.5}}
    )
    assert trainable._model == "LogisticRegression"
    assert trainable._model_hyper == {"C": 0.5}
    assert trainable.status_dict is None

    trainable.load_checkpoint(str(checkpoint_dir))
    assert trainable.status_dict["loss"] == pytest.approx(result["loss"])


def test_timeout_after_fitted_step(in_tmp_dir, insurance, monkeypatch):
    config = {**PREP, "model": {"model": "LinearRegression"}}
    trainable = _objective(
        config, insurance, {"LinearRegression": LinearRegression}, "regression", "MAE"
    )
    fitted = trainable.step()

    def _timed_out(timeout, func, *args, **kwargs):
        raise func_timeout.FunctionTimedOut()

    monkeypatch.setattr(informed_utils.func_timeout, "func_timeout", _timed_out)

    # the fitted result of the earlier step is kept, the trial is stopped
    result = trainable.step()
    assert result["training_status"] == "FITTED"
    assert result["loss"] == fitted["loss"]
    assert result["done"] is True

    # reused actor: a new trial does not inherit the previous result
    trainable.reset_config(config)
    assert trainable.step()["training_status"] == "TIMEOUT"
