"""
File: setup.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /setup.py
File: setup.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Wednesday, 16th November 2022 8:31:31 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2022, Panyi Dong

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
import sys
import glob
import json
import logging
import importlib
import subprocess
from typing import Optional

from setuptools import setup, find_packages, Extension
from setuptools.command import build_ext

logging.basicConfig()
log = logging.getLogger(__file__)

torch_spec = importlib.util.find_spec("torch")

if sys.version_info < (3, 7):
    log.info("Python >= 3.7 is required for the pipeline!")

try:
    from Cython.Build import cythonize
except ImportError:
    # create closure for deferred import
    def cythonize(*args, **kwargs):
        from Cython.Build import cythonize

        return cythonize(*args, **kwargs)


# Automatically get release version
InsurAutoML_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()[1:]  # remove the first character 'v'
)
# get only the version release number in case push is attached
InsurAutoML_version = (
    InsurAutoML_version.split("-")[0]
    if "-" in InsurAutoML_version
    else InsurAutoML_version
)

# assert version file
assert os.path.isfile("InsurAutoML/version.py")
# write version to VERSION file
with open("InsurAutoML/VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % InsurAutoML_version)

# Constant variables
INSTALL_LIST = [
    "setuptools==59.5.0",
    "cython",
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "fastparquet",
    "matplotlib",
    "seaborn>=0.11.0",
    "ray<2.0.0",
    "gensim",
    # "ray[tune]",
    # "ray[rllib]",
    "redis;platform_system=='Windows'",
    "tqdm==4.62.3",
    "mlflow==1.21.0",
    "tensorboardX",
    "hyperopt==0.2.5",
    # "auto-sklearn==0.14.6;platform_system=='Linux'",
    # "scikit-learn==0.24.2;platform_system=='Linux'",
    # "scikit-learn>1.0.0;platform_system=='Windows'",
    # "scikit-learn>1.0.0;platform_system=='MacOS'",
    "scikit-learn>=1.1.0",
]

EXTRA_DICT = {
    "normal": [],
    "extended": [
        # "rpy2;platform_system=='Linux'",
        "lightgbm",
        "xgboost",
        "pygam",
        "flaml",
        "nevergrad",
        "optuna",
    ],
    "nn": [
        # "rpy2;platform_system=='Linux'",
        "gensim",
        "torch",
        "nni",
        # "transformers",
        # "datasets",
    ],
    "dev": [
        "lightgbm",
        "xgboost",
        "pygam",
        "flaml",
        "nevergrad",
        "optuna",
        "gensim",
        "torch",
        "nni",
    ],
}

# check R installation
def r_home_from_subprocess() -> Optional[str]:
    """Return the R home directory from calling 'R RHOME'."""
    cmd = ("R", "RHOME")
    log.debug("Looking for R home with: {}".format(" ".join(cmd)))
    try:
        tmp = subprocess.check_output(cmd, universal_newlines=True)
    except Exception as e:  # FileNotFoundError, WindowsError, etc
        log.error(f"Unable to determine R home: {e}")
        return None
    r_home = tmp.split(os.linesep)
    if r_home[0].startswith("WARNING"):
        res = r_home[1]
    else:
        res = r_home[0].strip()
    return res


# TODO: move all Windows all code into an os-specific module ?
def r_home_from_registry() -> Optional[str]:
    """Return the R home directory from the Windows Registry."""
    from packaging.version import Version

    try:
        import winreg  # type: ignore
    except ImportError:
        import _winreg as winreg  # type: ignore
    # There are two possible locations for RHOME in the registry
    # We prefer the user installation (which the user has more control
    # over). Thus, HKEY_CURRENT_USER is the first item in the list and
    # the for-loop breaks at the first hit.
    for w_hkey in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
        try:
            with winreg.OpenKeyEx(w_hkey, "Software\\R-core\\R") as hkey:

                # >v4.x.x: grab the highest version installed
                def get_version(i):
                    try:
                        return Version(winreg.EnumKey(hkey, i))
                    except Exception:
                        return None

                latest = max(
                    (
                        v
                        for v in (
                            get_version(i) for i in range(winreg.QueryInfoKey(hkey)[0])
                        )
                        if v is not None
                    )
                )

                with winreg.OpenKeyEx(hkey, f"{latest}") as subkey:
                    r_home = winreg.QueryValueEx(subkey, "InstallPath")[0]

                # check for an earlier version
                if not r_home:
                    r_home = winreg.QueryValueEx(hkey, "InstallPath")[0]
        except Exception:  # FileNotFoundError, WindowsError, OSError, etc.
            pass
        else:
            # We have a path RHOME
            if sys.version_info[0] == 2:
                # Python 2 path compatibility
                r_home = r_home.encode(sys.getfilesystemencoding())
            # Break the loop, because we have a hit.
            break
    else:
        # for-loop did not break - RHOME is unknown.
        log.error("Unable to determine R home.")
        r_home = None
    return


def get_r_home() -> Optional[str]:
    """Get R's home directory (aka R_HOME).
    If an environment variable R_HOME is found it is returned,
    and if none is found it is trying to get it from an R executable
    in the PATH. On Windows, a third last attempt is made by trying
    to obtain R_HOME from the registry. If all attempt are unfruitful,
    None is returned.
    """

    r_home = os.environ.get("R_HOME")

    if not r_home:
        r_home = r_home_from_subprocess()
    if not r_home and os.name == "nt":
        r_home = r_home_from_registry()
    log.info(f"R home found: {r_home}")
    return r_home


# get R Home environment variable
# if found, install rpy2
# otherwise, do not install rpy2
R_HOME = get_r_home()
if not R_HOME:
    raise RuntimeError("""The R home directory could not be determined.""")

# only install for Linux
if not os.environ.get("R_HOME") and sys.platform == "linux":
    os.environ["R_HOME"] = R_HOME
    EXTRA_DICT["extended"].append("rpy2")
    EXTRA_DICT["nn"].append("rpy2")

DATA_LIST = ["Appendix/*", "example/*"]

EXCLUDE_LIST = [
    "tests",
    "example",
    "archive",
    "Appendix",
    "docs",
    ".github",
    "build",
    "dist",
]

SETUP_REQUIRES = [
    "setuptools==59.5.0",
    "cython",
    "numpy",
]

SETUP_ARGS = {
    "name": "InsurAutoML",
    "version": InsurAutoML_version,
    "author": "Panyi Dong",
    "url": "https://github.com/PanyiDong/InsurAutoML",
    "author_email": "panyid2@illinois.edu",
    "description": "Automated Machine Learning/AutoML pipeline.",
    "license": "MIT",
    # "cmdclass": {"build_ext": build_ext},
    "ext_modules": [],
}
EXT_MODULES = []

# get long description
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def setup_package():

    # global dep_list

    setup(
        packages=find_packages(
            exclude=EXCLUDE_LIST,
        ),
        package_dir={"InsurAutoML": "InsurAutoML"},
        include_package_data=True,
        package_data={"InsurAutoML": DATA_LIST},
        platforms=["Linux", "Windows", "MacOS"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=">=3.7",
        install_requires=INSTALL_LIST,
        extras_require=EXTRA_DICT,
        zip_safe=False,
        setup_requires=SETUP_REQUIRES,
        **SETUP_ARGS,
    )

    # dep_list = setup.install_requires + setup.extras_require


# torch extensions build (not used at this moment)
def build_torch_extensions():

    torch_extensions = []

    if torch_spec is not None:
        from torch.utils.cpp_extension import include_paths, BuildExtension

        # Extensions
        torch_extensions.append(
            Extension(
                name="pytorch_ext",
                sources=glob.glob("**/*.cpp", recursive=True)
                + glob.glob("**/*.cu", recursive=True),
                include_dirs=include_paths(),
                language="c++",
            )
        )

    SETUP_ARGS["cmdclass"] = {"build_ext": BuildExtension}

    return torch_extensions


# factory function
def postp_cython_build_ext(pars):
    # import delayed:

    # include_dirs adjusted:
    class _build_ext(build_ext):
        def finalize_options(self):
            build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy

            self.include_dirs.append(numpy.get_include())

    # object returned:
    return _build_ext(pars)


class CythonExt(Extension):
    def init(self, *args, **kwargs):
        self._include = []
        super().init(*args, **kwargs)

    @property
    def include_dirs(self):
        # defer import of numpy
        import numpy

        return self._include + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self._include = dirs


# cython extension build
def build_cython_extensions():

    # import numpy

    c_ext = "pyx"  # if USING_CYTHON else "c"
    sources = glob.glob("**/*.{}".format(c_ext), recursive=True)

    cython_extensions = [
        CythonExt(
            name=source.split(".")[0].replace(os.path.sep, "."),
            sources=[source],
            # include_dirs=[numpy.get_include()],
            language="c",
        )
        for source in sources
    ]

    # SETUP_ARGS["directives"] = {"linetrace": False, "language_level": 3}
    # SETUP_ARGS["cmdclass"] = {"build_ext": postp_cython_build_ext}

    return cython_extensions


# prepare for package.json file
def get_package():

    result = {}
    result["name"] = SETUP_ARGS["name"]
    result["author"] = {
        "name": SETUP_ARGS["author"],
        "email": SETUP_ARGS["author_email"],
    }
    result["repository"] = {
        "type": "git",
        "url": SETUP_ARGS["url"],
    }
    result["main"] = "main.py"
    result["private"] = False
    result["version"] = SETUP_ARGS["version"]
    result["description"] = SETUP_ARGS["description"]
    result["install_requires"] = INSTALL_LIST
    result["extras_require"] = EXTRA_DICT

    if os.path.exists("package.json"):
        os.remove("package.json")

    with open("package.json", "w") as f:
        json.dump(result, f, indent=4)


def get_requirements():

    if os.path.exists("requirements.txt"):
        os.remove("requirements.txt")

    with open(r"requirements.txt", "w") as fp:
        dep_list = [
            item.split(";")[0] for item in INSTALL_LIST + EXTRA_DICT["extended"]
        ]
        dep_list = list(set(dep_list))
        for item in dep_list:
            # write each item on a new line
            fp.write("%s\n" % item)

    if os.path.exists("requirements_nn.txt"):
        os.remove("requirements_nn.txt")

    with open(r"requirements_nn.txt", "w") as fp:
        dep_list = [item.split(";")[0] for item in INSTALL_LIST + EXTRA_DICT["nn"]]
        dep_list = list(set(dep_list))
        for item in dep_list:
            # write each item on a new line
            fp.write("%s\n" % item)


def main():

    # # check whether need to build pytorch extensions
    # SETUP_ARGS["ext_modules"] += build_torch_extensions()

    # add cython extensions
    SETUP_ARGS["ext_modules"] += build_cython_extensions()

    # setup package
    setup_package()
    log.info(
        "{}-{} installation succeeded.".format(
            SETUP_ARGS["name"], SETUP_ARGS["version"]
        )
    )


if __name__ == "__main__":

    get_requirements()
    main()
    get_package()
