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
Last Modified: Tuesday, 10th May 2022 3:28:49 pm
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

import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 7):
    sys.exit("Python >= 3.7 is required for the pipeline!")


def setup_package():

    setup(
        name="My_AutoML",
        version="0.2.1",
        author="Panyi Dong",
        url="https://github.com/PanyiDong/My_AutoML",
        author_email="panyid2@illinois.edu",
        description="Automated Machine Learning/AutoML pipeline.",
        license="MIT",
        packages=find_packages(
            exclude=[
                "tests",
                "example",
                "archive",
                "Appendix",
                "docs",
                ".github",
                "build",
                "dist",
            ]
        ),
        package_dir={"My_AutoML": "My_AutoML"},
        include_package_data=True,
        package_data={"My_AutoML": ["Appendix/*", "example/*"]},
        platforms=["Linux"],
        python_requires=">=3.7",
        install_requires=[
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "ray",
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
        ],
        extras_require={
            "lightweight": [],
            "normal": [
                "rpy2;platform_system=='Linux'",
                "lightgbm",
                "xgboost",
                "pygam",
            ],
            "nn": [
                "rpy2;platform_system=='Linux'",
                "lightgbm",
                "xgboost",
                "pygam",
                "torch",
                # "transformers",
                # "datasets",
            ],
        },
    )


if __name__ == "__main__":
    setup_package()
