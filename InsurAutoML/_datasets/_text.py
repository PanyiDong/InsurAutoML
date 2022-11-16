"""
File Name: _text.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_datasets/_text.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 6:59:03 pm
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
# Text Classification


def IMDB(train=True, test=False):
    if train and not test:
        dataset = load_dataset("imdb", split="train")
    elif test and not train:
        dataset = load_dataset("imdb", split="test")
    elif train and test:
        dataset = load_dataset("imdb")
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset


def GLUE(train=True, test=False):
    if train and not test:
        dataset = load_dataset("glue", split="train")
    elif test and not train:
        dataset = load_dataset("glue", split="test")
    elif train and test:
        dataset = load_dataset("glue")
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset


def AG_NEWS(train=True, test=False):
    if train and not test:
        dataset = load_dataset("ag_news", split="train")
    elif test and not train:
        dataset = load_dataset("ag_news", split="test")
    elif train and test:
        dataset = load_dataset("ag_news")
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset


####################################################################################################################
# Language Modeling


def WIKITEXT(train=True, test=False):
    if train and not test:
        dataset = load_dataset("wikitext", split="train")
    elif test and not train:
        dataset = load_dataset("wikitext", split="test")
    elif train and test:
        dataset = load_dataset("wikitext")
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset


####################################################################################################################
# Question Answering


def SQUAD(train=True, test=False, version=1):

    if version == 1:
        dataname = "squad"
    elif version == 2:
        dataname = "squad_v2"

    if train and not test:
        dataset = load_dataset(dataname, split="train")
    elif test and not train:
        dataset = load_dataset(dataname, split="test")
    elif train and test:
        dataset = load_dataset(dataname)
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset


def ADVERSARIAL_QA(train=True, test=False):
    if train and not test:
        dataset = load_dataset("adversarial_qa", split="train")
    elif test and not train:
        dataset = load_dataset("adversarial_qa", split="test")
    elif train and test:
        dataset = load_dataset("adversarial_qa")
    else:
        raise ValueError("train and test cannot be both False.")

    return dataset
