"""
File Name: _image.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_datasets/_image.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 6:58:45 pm
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

from datasets import load_dataset


####################################################################################################################
# Image Classification


def CIFAR(train=True, test=False, version=10):

    if version == 10:
        dataname = "cifar10"
    elif version == 100:
        dataname = "cifar100"

    if train and not test:
        dataset = load_dataset(dataname, split="train")
    elif test and not train:
        dataset = load_dataset(dataname, split="test")
    elif train and test:
        dataset = load_dataset(dataname)
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset


def MNIST(train=True, test=False):
    if train and not test:
        dataset = load_dataset("mnist", split="train")
    elif test and not train:
        dataset = load_dataset("mnist", split="test")
    elif train and test:
        dataset = load_dataset("mnist")
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset


def IMAGENET(train=True, test=False):
    if train and not test:
        dataset = load_dataset("imagenet-1k", use_auth_token=True, split="train")
    elif test and not train:
        dataset = load_dataset("imagenet-1k", use_auth_token=True, split="test")
    elif train and test:
        dataset = load_dataset("imagenet-1k", use_auth_token=True)
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset
