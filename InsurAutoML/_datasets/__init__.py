"""
File Name: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_datasets/__init__.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 6:58:40 pm
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

import importlib

datasets_spec = importlib.util.find_spec("datasets")

from ._tabular import (
    ADULT,  # Classification
    AUTO,
    BREAST,
    CAR_EVALUATION,
    NURSERY,
    PAGE_BLOCKS,
    HEART,
    EMPLOYEE,
    STROKE,
    HEART2020,
    TRAVEL_INSURANCE,
    IMBALANCED_INSURANCE,
    HEALTH_INSURANCE,
    CPU_ACT,  # Regression
    WIND,
    HOUSES,
    INSURANCE,
    MEDICAL_PREMIUM,
    PROD,  # Classification with text
    JIGSAW,
    JIGSAW100K,
    AIRBNB,
    IMDBGenre,
    FakeJob,
    FakeJob2,
    KickStarter,
    WINEReview,
    NewsChannel,
    WomenCloth,  # Regression with text
    MERCARI,
    MERCARI100K,
    AE,
    JCPenney,
    NewsPopularity,
    NewsPopularity2,
    BookPrice,
    DSSalary,
    CAHousePrice,
)


# datasets for Neural Network
if datasets_spec is not None:
    from ._text import (
        IMDB,  # Text Classification
        GLUE,
        AG_NEWS,
        WIKITEXT,  # Language Modeling
        SQUAD,  # Question Answering
        ADVERSARIAL_QA,
    )

    from ._image import (
        CIFAR,
        MNIST,
        IMAGENET,
    )
