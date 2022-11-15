"""
File Name: _pretrained.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_model/_pretrained.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:15:26 pm
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
import numpy as np
import pandas as pd
from tqdm import tqdm

pytorch_spec = importlib.util.find_spec("torch")
if pytorch_spec is not None:
    import torch
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformers_spec = importlib.util.find_spec("transformers")
if transformers_spec is not None:
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForMultipleChoice,
        # AutoModelForImageClassification,
    )

from InsurAutoML._utils._preprocessing import text_preprocessing_transformers


def load_model(
    pretrained_model="bert-base-uncased",
    task_type="TextClassification",
):

    """
    Parameters
    ----------
    pretrained_model: pre-trained model name, default = "bert-base-uncased"
    common pre-trained models
    1. ALBERT:   albert-base-v2
    2. BERT:     bert-base-uncased
    3. DeBERTa:  microsoft/deberta-v2-xlarge
    4. GPT2:     gpt2
    5. RoBERTa:  roberta-base
    6. T5:       t5-small
    7. XLNet:    xlnet-base-cased

    task_type: task type, default = "TextClassification"
    supported task types: "TextClassification", "TokenClassification",
    "QuestionAnswering", "LanguageModeling", "Translation", "Summarization",
    "MultipleChoice"
    """

    # map task type to model class
    task_mapping = {
        "TextClassification": AutoModelForSequenceClassification,
        "TokenClassification": AutoModelForTokenClassification,
        "QuestionAnswering": AutoModelForQuestionAnswering,
        "LanguageModeling": AutoModelForCausalLM,
        "Translation": AutoModelForSeq2SeqLM,
        "Summarization": AutoModelForSeq2SeqLM,
        "MultipleChoice": AutoModelForMultipleChoice,
    }

    model = task_mapping[task_type].from_pretrained(pretrained_model)

    return model


class PretrainedModel:
    def __init__(
        self,
        pretrained_model="bert-base-uncased",
        task_type="TextClassification",
        batch_size=32,
        input_len=512,
        optimizer="Adam",
        lr=None,
        metric="CrossEntropyLoss",
        num_epochs=10,
        progressbar=True,
        save=False,
    ):
        """
        Parameters
        ----------
        pretrained_model: pre-trained model name, default = "bert-base-uncased"
        common pre-trained models
        1. ALBERT:   albert-base-v2
        2. BERT:     bert-base-uncased
        3. DeBERTa:  microsoft/deberta-v2-xlarge
        4. GPT2:     gpt2
        5. RoBERTa:  roberta-base
        6. T5:       t5-small
        7. XLNet:    xlnet-base-cased

        task_type: task type, default = "TextClassification"
        supported task types: "TextClassification", "TokenClassification",
        "QuestionAnswering", "LanguageModeling", "Translation", "Summarization",
        "MultipleChoice"
        """

        self.pretrained_model = pretrained_model
        self.task_type = task_type
        self.batch_size = batch_size
        self.input_len = input_len
        self.optimizer = optimizer
        self.lr = lr
        self.metric = metric
        self.num_epochs = num_epochs
        self.progressbar = progressbar
        self.save = save

    def fit(self, X, y=None):

        # if dataframe, check if has "text" and "label" columns
        if isinstance(X, pd.DataFrame):
            data = {"text": X.values.ravel(), "label": y.values.ravel()}
        elif isinstance(X, pd.Series):
            data = {"text": X.values, "label": y.values}
        else:
            data = {"text": X, "label": y}

        # preprocess text
        preprocessed_loader = text_preprocessing_transformers(
            data,
            batch_size=self.batch_size,
            max_len=self.input_len,
        )

        # load model
        self.model = load_model(
            pretrained_model=self.pretrained_model, task_type=self.task_type
        )
        self.model.to(device)  # load model to device
        self.model.train()  # set model to training mode

        # set training settings
        # optimizer
        # specify optimizer
        if self.optimizer == "Adam":
            lr = 0.001 if self.lr is None else self.lr
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif self.optimizer == "SGD":
            lr = 0.1 if self.lr is None else self.lr
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError("optimizer not recognized")

        # metric
        if self.metric == "CrossEntropyLoss":
            from torch.nn import CrossEntropyLoss

            self.metric = CrossEntropyLoss()
        elif self.metric == "NegativeLogLikelihoodLoss":
            from torch.nn import NLLLoss

            self.metric = NLLLoss()
        else:
            raise ValueError("metric not recognized")

        # set progress bar
        tepoch = (
            tqdm(preprocessed_loader, position=0, unit="batch")
            if self.progressbar
            else preprocessed_loader
        )

        # train model
        for _ in range(self.num_epochs):
            # iter through batches
            for idx, (input, label) in enumerate(tepoch):
                # load batch data to device
                input = input.to(device)
                label = label.to(device)

                output = self.model(input)  # forward phase
                loss = self.metric(output, label)  # calculate loss
                loss.backward()

                self.optimizer.step()  # update weights
                self.optimizer.zero_grad()

                if self.progressbar:
                    tepoch.set_postfix(loss=loss.item())

        # whether to save model
        if self.save:
            torch.save(self.model, "model.pt")

    def predict(self, X):

        # if dataframe, check if has "text" and "label" columns
        if isinstance(X, pd.DataFrame):
            data = {"text": X.values.ravel()}
        elif isinstance(X, pd.Series):
            data = {"text": X.values}
        else:
            data = {"text": X}

        # preprocess text
        preprocessed_loader = text_preprocessing_transformers(
            data,
            batch_size=len(data["text"]),
            max_len=self.input_len,
        )

        # only one epoch and one batch for prediction
        for idx, (input) in enumerate(preprocessed_loader):
            input = input.to(device)
            with torch.no_grad():
                output = self.model(input)

        return output.cpu().numpy()  # make prediction to numpy array
