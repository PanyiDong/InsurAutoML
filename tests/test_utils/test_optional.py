"""
File Name: test_optional.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /tests/test_utils/test_optional.py
File Created: Monday, 16th November 2026 7:12:56 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:36:43 pm
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

import os
import subprocess
import sys
import textwrap
from importlib.machinery import EXTENSION_SUFFIXES

import pytest

from InsurAutoML import _optional
from InsurAutoML._optional import (
    INFORMED_INSTALL_CMD,
    InformedNotInstalledError,
    informed_not_installed_error,
    missing_informed_components,
    require_informed,
)

EXTENSIONS = [
    "InsurAutoML.ext.suppsplit.suppsplit_cpp",
]


def test_missing_package(monkeypatch):
    real_find_spec = _optional.importlib.util.find_spec
    monkeypatch.setattr(
        _optional.importlib.util,
        "find_spec",
        lambda name, *args: (
            None if name == "missforest" else real_find_spec(name, *args)
        ),
    )

    assert "missforest" in missing_informed_components()
    assert not _optional.informed_available()


def test_missing_extensions(tmp_path, monkeypatch):
    monkeypatch.setattr(_optional, "_EXT_DIR", str(tmp_path))
    for subpackage, module in _optional.INFORMED_EXTENSIONS:
        # the C++ source folder has the module name, it is not the compiled extension
        (tmp_path / subpackage / module).mkdir(parents=True)

    assert set(EXTENSIONS) <= set(missing_informed_components())

    # compiled extension files make them available
    for subpackage, module in _optional.INFORMED_EXTENSIONS:
        (tmp_path / subpackage / (module + EXTENSION_SUFFIXES[0])).touch()

    assert not set(EXTENSIONS) & set(missing_informed_components())


def test_require_informed(monkeypatch):
    monkeypatch.setattr(
        _optional, "missing_informed_components", lambda: ["missforest"]
    )

    with pytest.raises(InformedNotInstalledError) as error:
        require_informed("InformedAutoTabular")
    assert INFORMED_INSTALL_CMD in str(error.value)
    assert "missforest" in str(error.value)
    # InformedNotInstalledError is an ImportError
    assert isinstance(informed_not_installed_error(), ImportError)

    monkeypatch.setattr(_optional, "missing_informed_components", lambda: [])
    require_informed("InformedAutoTabular")  # does not raise


# import blockers run in a separate interpreter, so the test session is not affected
_BLOCKER = """
import importlib.abc, importlib.util, sys
BLOCKED = {blocked!r}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in BLOCKED or name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("No module named %r" % name)
sys.meta_path.insert(0, Blocker())
_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a: None if name in BLOCKED else _find_spec(name, *a)
"""


def _run(blocked, code):
    script = _BLOCKER.format(blocked=blocked) + textwrap.dedent(code)
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def test_informed_not_installed():
    out = _run(
        ["missforest"],
        """
        import InsurAutoML
        from InsurAutoML._optional import InformedNotInstalledError
        from InsurAutoML.hpo.informed import INFORMED_AVAILABLE
        assert not INFORMED_AVAILABLE
        # the default pipelines are still available
        assert InsurAutoML.AutoTabular.__module__ == "InsurAutoML.hpo.ML"
        for name in ["InformedAutoTabular", "InformedAutoTabularClassifier",
                     "InformedAutoTabularRegressor"]:
            try:
                getattr(InsurAutoML, name)()
            except InformedNotInstalledError as error:
                assert "missforest" in str(error) and "[informed]" in str(error)
            else:
                raise AssertionError(name + " did not raise")
        try:
            import InsurAutoML.prep
        except InformedNotInstalledError:
            pass
        else:
            raise AssertionError("InsurAutoML.prep did not raise")
        print("OK")
        """,
    )
    assert "OK" in out


def test_extension_not_built():
    out = _run(
        ["InsurAutoML.ext.suppsplit.suppsplit_cpp"],
        """
        from InsurAutoML._optional import InformedNotInstalledError
        try:
            import InsurAutoML.ext
        except InformedNotInstalledError as error:
            assert "INFORMEDAUTOML=1" in str(error)
            print("OK")
        """,
    )
    assert "OK" in out
