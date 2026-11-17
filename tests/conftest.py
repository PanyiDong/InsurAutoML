"""
Shared test configuration.

The torch neural network models (MLP/RNN) are slow to train, in particular on
CPU-only machines. They are tested (and covered) by tests/test_model; in all other
tests (AutoTabular/InformedAutoML searches) they are removed from the model pool,
so the hyperparameter searches only sample the faster models.

Set INSURAUTOML_TEST_NN=1 to keep them in the searches.
"""

import os

import pytest

TORCH_MODELS = ("MLP_Classifier", "MLP_Regressor", "RNN_Classifier", "RNN_Regressor")


@pytest.fixture(autouse=True)
def _exclude_torch_models_from_search(request, monkeypatch):
    if os.environ.get("INSURAUTOML_TEST_NN") == "1":
        return
    if "test_model" in request.path.parts:
        return

    from InsurAutoML import model

    # the searches read the model pools at fit time (in the driver, the models are
    # passed to the Ray Tune trials), restored after each test
    for pool in (model.classifiers, model.regressors):
        for name in TORCH_MODELS:
            if name in pool:
                monkeypatch.delitem(pool, name)
