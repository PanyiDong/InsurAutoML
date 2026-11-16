"""
File Name: test_datasets_hf.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_datasets/test_datasets_hf.py
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

# The text/image loaders are thin wrappers around huggingface
# `datasets.load_dataset`. The `datasets` package is replaced by a fake module
# recording the calls, so no network access (or the package) is needed.

import importlib
import importlib.machinery
import sys
import types

import pytest

_HF_NAMES = [
    "IMDB",
    "GLUE",
    "AG_NEWS",
    "WIKITEXT",
    "SQUAD",
    "ADVERSARIAL_QA",
    "CIFAR",
    "MNIST",
    "IMAGENET",
]


@pytest.fixture
def hf(monkeypatch):
    calls = []

    def load_dataset(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return {"name": name, "split": kwargs.get("split"), "kwargs": kwargs}

    fake = types.ModuleType("datasets")
    fake.__spec__ = importlib.machinery.ModuleSpec("datasets", None)
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    import InsurAutoML.datasets as ds_pkg

    pkg_attrs = set(vars(ds_pkg))
    # make sure the wrappers are (re)imported against the fake package
    for mod in ["InsurAutoML.datasets.text", "InsurAutoML.datasets.image"]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    text = importlib.import_module("InsurAutoML.datasets.text")
    image = importlib.import_module("InsurAutoML.datasets.image")

    yield types.SimpleNamespace(text=text, image=image, calls=calls)

    # drop the modules bound to the fake package
    for mod in ["InsurAutoML.datasets.text", "InsurAutoML.datasets.image"]:
        sys.modules.pop(mod, None)
    for name in set(vars(ds_pkg)) - pkg_attrs:
        delattr(ds_pkg, name)


TEXT_LOADERS = [
    ("IMDB", "imdb", {}),
    ("GLUE", "glue", {}),
    ("AG_NEWS", "ag_news", {}),
    ("WIKITEXT", "wikitext", {}),
    ("SQUAD", "squad", {}),
    ("SQUAD", "squad_v2", {"version": 2}),
    ("ADVERSARIAL_QA", "adversarial_qa", {}),
]

IMAGE_LOADERS = [
    ("CIFAR", "cifar10", {}),
    ("CIFAR", "cifar100", {"version": 100}),
    ("MNIST", "mnist", {}),
    ("IMAGENET", "imagenet-1k", {}),
]


def _check_splits(func, dataname, extra, calls, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}

    # train only
    out = func(train=True, test=False, **extra)
    assert out["name"] == dataname and out["split"] == "train"
    # test only
    out = func(train=False, test=True, **extra)
    assert out["name"] == dataname and out["split"] == "test"
    # both -> full DatasetDict, no split argument
    out = func(train=True, test=True, **extra)
    assert out["name"] == dataname and out["split"] is None
    # neither -> error
    with pytest.raises(ValueError, match="cannot be both False"):
        func(train=False, test=False, **extra)

    assert [c[0] for c in calls] == [dataname] * 3
    for _, _, kwargs in calls:
        for key, value in extra_kwargs.items():
            assert kwargs[key] == value


@pytest.mark.parametrize("func_name, dataname, extra", TEXT_LOADERS)
def test_text_loaders(func_name, dataname, extra, hf):
    _check_splits(getattr(hf.text, func_name), dataname, extra, hf.calls)


@pytest.mark.parametrize("func_name, dataname, extra", IMAGE_LOADERS)
def test_image_loaders(func_name, dataname, extra, hf):
    extra_kwargs = {"use_auth_token": True} if func_name == "IMAGENET" else {}
    _check_splits(getattr(hf.image, func_name), dataname, extra, hf.calls, extra_kwargs)


def test_default_is_train(hf):
    assert hf.text.IMDB()["split"] == "train"
    assert hf.image.MNIST()["split"] == "train"


def test_package_exports_hf_loaders_when_available(hf):
    import InsurAutoML.datasets as ds_pkg

    before = set(vars(ds_pkg))
    try:
        importlib.reload(ds_pkg)
        for name in _HF_NAMES:
            assert callable(getattr(ds_pkg, name))
        assert ds_pkg.IMDB(train=False, test=True)["split"] == "test"
    finally:
        # restore the package namespace to its original state
        for name in set(vars(ds_pkg)) - before:
            delattr(ds_pkg, name)
