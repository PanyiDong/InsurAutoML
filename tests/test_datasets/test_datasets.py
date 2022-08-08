"""
File: test_dataset.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /tests/test_datasets/test_dataset.py
File Created: Sunday, 7th August 2022 9:38:04 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 7th August 2022 10:29:52 pm
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

import pandas as pd
import shutil

def test_datasets() :

    from My_AutoML._datasets import (
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

    datasets = [
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
    ]
    
    for dataset in datasets :
    
        data = dataset()

        assert isinstance(data, pd.DataFrame), "Datasets not loaded correctly."

    datasets = [
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
    ]
    
    for dataset in datasets :
        
        data = dataset(split = "test")

        assert isinstance(data, pd.DataFrame), "Datasets not loaded correctly."
        
        data = dataset(split = ["test"])

        assert isinstance(data, pd.DataFrame), "Datasets not loaded correctly."
        
    # clear data storage
    shutil.rmtree("tmp")
        
    