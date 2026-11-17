"""
Shared test configuration.

Working directory: the fits write files in the working directory (saved models
named model_name, tmp/, catboost_info/, ...). The tests run in a temporary
directory of the session, with the data folders (example/, data/) linked into it,
so the repository stays clean.

The torch neural network models (MLP/RNN) are slow to train, in particular on
CPU-only machines. They are tested (and covered) by tests/test_model; in all other
tests (AutoTabular/InformedAutoML searches) they are removed from the model pool,
so the hyperparameter searches only sample the faster models.

Set INSURAUTOML_TEST_NN=1 to keep them in the searches.
"""

import os
import shutil
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
# folders of the repository the tests read with relative paths
DATA_FOLDERS = ("example", "data")

TORCH_MODELS = ("MLP_Classifier", "MLP_Regressor", "RNN_Classifier", "RNN_Regressor")


@pytest.fixture(scope="session", autouse=True)
def _session_working_directory(tmp_path_factory):
    working_directory = tmp_path_factory.mktemp("workdir")
    for name in DATA_FOLDERS:
        source = REPOSITORY / name
        if not source.is_dir():
            continue
        try:
            (working_directory / name).symlink_to(source, target_is_directory=True)
        except OSError:
            # no symlink privilege (Windows): copy once per session
            shutil.copytree(source, working_directory / name)

    previous = os.getcwd()
    os.chdir(working_directory)
    try:
        yield working_directory
    finally:
        os.chdir(previous)


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
