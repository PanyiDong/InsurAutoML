"""
File: test_hpo_spaces.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Relative Path: /tests/test_hpo/test_hpo_spaces.py

Hyperparameter spaces of every supported search algorithm, for the standard and the
informed pipelines: the space is built and handed to the search algorithm the same
way as in fit() (AutoTabularBase._get_search_alg), and every suggested configuration
must be accepted by the trainable, i.e. all components are constructed with the
suggested hyperparameters.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.variant_generator import generate_variants

from InsurAutoML import AutoTabularClassifier, AutoTabularRegressor
from InsurAutoML._optional import informed_available
from InsurAutoML.hpo.utils import TabularObjective
from InsurAutoML.utils.optimize import get_algo, set_algo_seed

EXAMPLE_DATA = Path(__file__).resolve().parents[2] / "example" / "example_data"
SEARCH_ALGOS = ["RandomSearch", "GridSearch", "HyperOpt", "Optuna", "Nevergrad", "CFO"]


@pytest.fixture(scope="module")
def heart():
    data = pd.read_csv(EXAMPLE_DATA / "heart.csv").sample(n=200, random_state=0)
    data = data.reset_index(drop=True)
    return data.drop(columns=["HeartDisease"]), data[["HeartDisease"]]


@pytest.fixture(scope="module")
def insurance():
    data = pd.read_csv(EXAMPLE_DATA / "insurance.csv").sample(n=200, random_state=0)
    data = data.reset_index(drop=True)
    return data.drop(columns=["expenses"]), data[["expenses"]]


def _suggest(mol, space, n=3):
    """Configurations suggested by the search algorithm, as Ray Tune would."""
    algo = get_algo(mol.search_algo)
    mol.search_algo_settings.update(set_algo_seed(mol.search_algo, mol.seed))
    search_alg, param_space = mol._get_search_alg(algo, space)

    if isinstance(search_alg, BasicVariantGenerator):
        # random/grid search: Ray Tune samples param_space itself
        return [
            config
            for seed in range(n)
            for _, config in generate_variants(
                param_space, random_state=np.random.RandomState(seed)
            )
        ][:n]

    search_alg.set_search_properties("loss", "min", param_space or {})
    configs = [search_alg.suggest("trial_{}".format(i)) for i in range(n)]
    # constants of param_space are merged into the suggested configurations
    return [{**(param_space or {}), **config} for config in configs]


@pytest.mark.parametrize("search_algo", SEARCH_ALGOS)
@pytest.mark.parametrize(
    "estimator, data", [(AutoTabularClassifier, "heart"), (AutoTabularRegressor, "insurance")]
)
def test_search_space(search_algo, estimator, data, request, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = request.getfixturevalue(data)

    mol = estimator(search_algo=search_algo, model_name="space", seed=1)
    mol._init_fit()
    *components, space = mol.get_hyperparameter_space(X, y)

    configs = _suggest(mol, space)
    assert len(configs) == 3

    for config in configs:
        objective = TabularObjective.__new__(TabularObjective)
        (
            objective.encoder,
            objective.imputer,
            objective.balancing,
            objective.scaling,
            objective.feature_selection,
            objective.models,
        ) = components
        objective.dict2config(dict(config))

        # every component of the pipeline is constructed from the configuration
        for name, component in zip(
            ["encoder", "imputer", "balancing", "scaling", "feature_selection", "models"],
            [objective.enc, objective.imp, objective.blc, objective.scl, objective.fts, objective.mol],
        ):
            assert component is not None
        assert objective._model in components[-1]


@pytest.mark.skipif(not informed_available(), reason="informed installation missing")
@pytest.mark.parametrize("search_algo", SEARCH_ALGOS)
def test_informed_search_space(search_algo, heart, tmp_path, monkeypatch):
    from InsurAutoML import InformedAutoTabularClassifier
    from InsurAutoML.hpo.informed.utils import InformedTabularObjective

    monkeypatch.chdir(tmp_path)
    X, y = heart

    mol = InformedAutoTabularClassifier(search_algo=search_algo, model_name="space", seed=1)
    mol._init_fit()
    encoder, complete_prep, missing_prep, models, space = mol.get_hyperparameter_space(X, y)

    for config in _suggest(mol, space):
        objective = InformedTabularObjective.__new__(InformedTabularObjective)
        objective.encoder, objective.complete_prep = encoder, complete_prep
        objective.missing_prep, objective.models = missing_prep, models
        objective.dict2config(dict(config))

        assert objective._encoder in encoder
        assert objective._complete_prep in complete_prep
        assert objective._missing_prep in missing_prep
        assert objective._model in models
