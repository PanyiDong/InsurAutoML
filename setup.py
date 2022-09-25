"""
File: setup.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.0.2
Relative Path: /setup.py
File Created: Friday, 4th March 2022 11:33:55 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 25th September 2022 11:41:31 am
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
import logging
import importlib

from setuptools import setup, find_packages, Extension

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

    USING_CYTHON = False
else:
    USING_CYTHON = True


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
    "ray",
    "gensim",
    # "ray[tune]",
    # "ray[rllib]",
    "redis;platform_system=='Windows'",
    "tqdm==4.62.3",
    "mlflow==1.21.0",
    "tensorboardX",
    "hyperopt==0.2.5",
    "auto-sklearn==0.14.6;platform_system=='Linux'",
    "scikit-learn==0.24.2;platform_system=='Linux'",
    "scikit-learn>1.0.0;platform_system=='Windows'",
    "scikit-learn>1.0.0;platform_system=='MacOS'",
]

EXTRA_DICT = {
    "lightweight": [],
    "normal": [
        "rpy2;platform_system=='Linux'",
        "lightgbm",
        "xgboost",
        "pygam",
    ],
    "nn": [
        "rpy2;platform_system=='Linux'",
        "gensim",
        "lightgbm",
        "xgboost",
        "pygam",
        "torch",
        "nni",
        # "transformers",
        # "datasets",
    ],
}

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

SETUP_ARGS = {
    "name": "My_AutoML",
    "version": "0.2.1",
    "author": "Panyi Dong",
    "url": "https://github.com/PanyiDong/My_AutoML",
    "author_email": "panyid2@illinois.edu",
    "description": "Automated Machine Learning/AutoML pipeline.",
    "license": "MIT",
    # "cmdclass": {"build_ext": build_ext},
    "ext_modules": [],
}
EXT_MODULES = []


def setup_package():

    setup(
        packages=find_packages(
            exclude=EXCLUDE_LIST,
        ),
        package_dir={"My_AutoML": "My_AutoML"},
        include_package_data=True,
        package_data={"My_AutoML": DATA_LIST},
        platforms=["Linux", "Windows", "MacOS"],
        python_requires=">=3.7",
        install_requires=INSTALL_LIST,
        extras_require=EXTRA_DICT,
        zip_safe=False,
        setup_requires=["cython", "numpy"],
        **SETUP_ARGS,
    )


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


def build_cython_extensions():

    import numpy

    c_ext = "pyx" if USING_CYTHON else "c"
    sources = glob.glob("**/*.{}".format(c_ext), recursive=True)

    cython_extensions = cythonize(
        [
            Extension(
                name=source.split(".")[0].replace(os.path.sep, "."),
                sources=[source],
                include_dirs=[numpy.get_include()],
                language="c++",
            )
            for source in sources
        ]
    )

    # SETUP_ARGS["directives"] = {"linetrace": False, "language_level": 3}

    return cython_extensions


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
    main()
