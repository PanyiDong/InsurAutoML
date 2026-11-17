"""
File: _optional.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Relative Path: /InsurAutoML/_optional.py

Availability of the optional "informed" installation.

The informed pipelines (InformedAutoTabular*, InsurAutoML.prep) need the compiled
C++ extension in InsurAutoML.ext (SPlit) and the extra Python
dependencies (missforest, ...). They are only built/installed with

    INFORMEDAUTOML=1 pip install "InsurAutoML[informed]"

The checks below do not import anything, so a default installation can import
InsurAutoML without the optional components.
"""

from __future__ import annotations

import importlib.util
import os
from importlib.machinery import EXTENSION_SUFFIXES
from typing import List

INFORMED_INSTALL_CMD = 'INFORMEDAUTOML=1 pip install "InsurAutoML[informed]"'
# an existing (normal) installation is kept by pip as "already satisfied", reinstall
# InsurAutoML itself to build the C++ extension
INFORMED_REINSTALL_CMD = (
    "INFORMEDAUTOML=1 pip install --force-reinstall --no-deps --no-cache-dir "
    "InsurAutoML"
)

# Python packages only needed by the informed pipelines
INFORMED_PACKAGES = ["missforest"]

# compiled extensions: (sub-package of InsurAutoML.ext, extension module name)
INFORMED_EXTENSIONS = [
    ("suppsplit", "suppsplit_cpp"),
]

_EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext")


class InformedNotInstalledError(ImportError):
    """The optional informed installation is missing."""


def missing_informed_components() -> List[str]:
    """Names of the missing informed components (empty if all are available)."""
    missing = [
        package
        for package in INFORMED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    for subpackage, module in INFORMED_EXTENSIONS:
        # look for the compiled extension file (e.g. suppsplit_cpp.cpython-312-*.so)
        # without importing; the C++ source folder has the same name as the module
        folder = os.path.join(_EXT_DIR, subpackage)
        if not any(
            os.path.isfile(os.path.join(folder, module + suffix))
            for suffix in EXTENSION_SUFFIXES
        ):
            missing.append("InsurAutoML.ext.{}.{}".format(subpackage, module))

    return missing


def informed_available() -> bool:
    """Whether the informed pipelines can be used."""
    return len(missing_informed_components()) == 0


def informed_not_installed_error(feature: str = "InformedAutoTabular") -> ImportError:
    """Error asking the user to install the informed version."""
    return InformedNotInstalledError(
        "{} is not available, the InformedAutoML version is not installed "
        "(missing: {}).\nInstall it with:\n    {}\nIf InsurAutoML is already installed, "
        "also rebuild it with the C++ extension:\n    {}".format(
            feature,
            ", ".join(missing_informed_components()),
            INFORMED_INSTALL_CMD,
            INFORMED_REINSTALL_CMD,
        )
    )


def require_informed(feature: str = "InformedAutoTabular") -> None:
    """Raise InformedNotInstalledError if the informed installation is missing."""
    if not informed_available():
        raise informed_not_installed_error(feature)
