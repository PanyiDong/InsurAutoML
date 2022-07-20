"""
File: test.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /test.py
File Created: Sunday, 17th July 2022 10:45:28 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 19th July 2022 7:51:36 pm
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

# import torch
# import torch.nn as nn
# import torch.optim as optim

# # from torch.utils.data import DataLoader, TensorDataset
# import nni
# from nni.retiarii import serialize
# import nni.retiarii.strategy as strategy
# import nni.retiarii.nn.pytorch as nninn

# # from nni.retiarii.evaluator.pytorch import DataLoader
# # from nni.retiarii.evaluator import FunctionalEvaluator
# from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from My_AutoML._utils._nas._nni import MLPBaseSpace
# from My_AutoML._utils._tensor import CustomTensorDataset

# from torchvision import transforms
# from torchvision.datasets import MNIST

# transf = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip()]
# normalize = [
#     transforms.ToTensor(),
#     transforms.Normalize(
#         [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]
#     ),
# ]
# train_dataset = serialize(
#     MNIST,
#     "data",
#     train=True,
#     download=True,
#     transform=transforms.Compose(transf + normalize),
# )
# test_dataset = serialize(
#     MNIST, "data", train=False, transform=transforms.Compose(normalize)
# )


# @nni.trace
# def epochEvaluation(
#     model,
#     dataloader,
#     optimizer,
#     criteria,
#     epoch,
#     mode="train",
# ):
#     # initialize loss and accuracy
#     report_loss = 0
#     accuracy = 0

#     # set model mode
#     if mode == "train":
#         model.train()
#         for idx, (input, label) in enumerate(dataloader):
#             input = input.to(device)
#             label = label.to(device)
#             optimizer.zero_grad()
#             output = model(input)
#             loss = criteria(output, label)
#             loss.backward()
#             optimizer.step()

#     elif mode == "eval":
#         model.eval()

#         with torch.no_grad():
#             for idx, (input, label) in enumerate(dataloader):
#                 input = input.to(device)
#                 label = label.to(device)
#                 output = model(input)
#                 loss = criteria(output, label)
#                 report_loss += loss.item()
#                 accuracy += (output.argmax(dim=1) == label).sum().item()

#         report_loss /= len(dataloader.dataset)
#         accuracy = 100.0 * accuracy / len(dataloader.dataset)

#         return accuracy


# @nni.trace
# def modelEvaluation(
#     model_cls,
#     trainloader,
#     testloader,
#     optimizer,
#     criteria,
#     num_epoch,
# ):
#     # train = TensorDataset(
#     #     torch.randn(2000, 32),
#     #     torch.randint(0, 2, (2000,)),
#     # )
#     # # inputSize = train.inputSize()
#     # # outputSize = train.outputSize()
#     # test = TensorDataset(
#     #     torch.randn(1000, 32),
#     #     torch.randint(0, 2, (1000,)),
#     # )
#     # trainloader = DataLoader(train, batch_size=64, shuffle=True, drop_last=True)
#     # testloader = DataLoader(test, batch_size=1)

#     model = model_cls()
#     model.to(device)

#     optimizer = optimizer(model.parameters())

#     for epoch in range(num_epoch):
#         epochEvaluation(
#             model,
#             trainloader,
#             optimizer,
#             criteria,
#             epoch,
#             mode="train",
#         )
#         accuracy = epochEvaluation(
#             model,
#             testloader,
#             optimizer,
#             criteria,
#             epoch,
#             mode="eval",
#         )
#         nni.report_intermediate_result(accuracy)

#     nni.report_final_result(accuracy)


# import nni.retiarii.evaluator.pytorch.lightning as pl


# @nni.trace
# class plEvaluator(pl.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, X):

#         return self.model(X)

#     def training_step(self, batch, batch_idx):
#         # training_step defined the train loop.
#         # It is independent of forward
#         print(batch)
#         input, label = batch
#         print(input, label)
#         output = self.model(input)  # model is the one that is searched for
#         loss = self.criterion(output, label)
#         # Logging to TensorBoard by default
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         input, label = batch
#         output = self.model(input)  # model is the one that is searched for
#         loss = self.criterion(output, label)
#         self.log("val_loss", loss)

#         # def configure_optimizers(self):
#         #     optimizer = RMSpropTF(self.parameters(), lr=self.hparams.learning_rate,
#         #                       weight_decay=self.hparams.weight_decay,
#         #                       momentum=0.9, alpha=0.9, eps=1.0)
#         # return {
#         #     'optimizer': optimizer,
#         #     'scheduler': CosineAnnealingLR(optimizer, self.hparams.max_epochs)
#         # }

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def on_validation_epoch_end(self):
#         nni.report_intermediate_result(self.trainer.callback_metrics["val_loss"].item())

#     def teardown(self, stage):
#         if stage == "fit":
#             nni.report_final_result(self.trainer.callback_metrics["val_loss"].item())

import torch
import nni

from My_AutoML._utils._tensor import CustomTensorDataset
from My_AutoML._utils._nas import Trainer, MLPBaseSpace


@nni.trace
def tensorGene(mode="train"):

    if mode == "train":
        return CustomTensorDataset(
            torch.randn(2000, 32), torch.randint(0, 2, (2000,)), format="tuple"
        )
    elif mode == "test":
        return CustomTensorDataset(
            torch.randn(1000, 32), torch.randint(0, 2, (1000,)), format="tuple"
        )


train = nni.trace(tensorGene)(mode="train")
inputSize = train.inputSize()
outputSize = train.outputSize()
test = nni.trace(tensorGene)(mode="test")

trainer = Trainer(
    search_space=MLPBaseSpace(inputSize, outputSize),
    evaluator="pl",
)
trainer.train(train, test)

# evaluator = FunctionalEvaluator(
#     modelEvaluation,
#     trainloader=nni.trace(DataLoader)(
#         train, batch_size=64, shuffle=True, drop_last=True
#     ),
#     testloader=nni.trace(DataLoader)(test, batch_size=1),
#     optimizer=nni.trace(optim.Adam),
#     criteria=nni.trace(nn.CrossEntropyLoss)(),
#     num_epoch=10,
# )


# evaluator = pl.Lightning(
#     lightning_module=plEvaluator(space),
#     trainer=pl.Trainer(max_epochs=10, gpus=1),
#     train_dataloader=pl.DataLoader(
#         train,
#         batch_size=64,
#         shuffle=True,
#         drop_last=True,
#     ),
#     val_dataloaders=pl.DataLoader(
#         test,
#         batch_size=1,
#     ),
# )

# exp = RetiariiExperiment(
#     space,
#     evaluator,
#     [],
#     searchStrategy,
# )
# exp_config = RetiariiExeConfig("local")
# exp_config.experiment_name = "nas_test"
# # exp_config.execution_engine = "oneshot"
# exp_config.max_trial_number = 100
# exp_config.max_trial_duration = 360
# exp_config.trial_concurrency = 1
# exp_config.trial_gpu_number = 1
# exp_config.training_service.use_active_gpu = True
# exp.run(exp_config)
# exp.stop()

# for model_dict in exp.export_top_models(formatter="dict"):
#     print(model_dict)

# # fuser -k -n tcp 8080
