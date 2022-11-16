"""
File Name: _nn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_imputation/_nn.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:13:58 pm
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

from __future__ import annotations

from typing import Union
from time import sleep
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings

# check if tensorflow exists
# if exists, import tensorflow
import importlib

tensorflow_spec = importlib.util.find_spec("tensorflow")
if tensorflow_spec is not None:
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.disable_v2_behavior() # use tf < 2.0 functions

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    import torch
    import torch.optim
    from torch import nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from InsurAutoML._utils import (
    random_index,
    feature_rounding,
    get_missing_matrix,
    formatting,
)
from InsurAutoML._scaling import MinMaxScale


class GAIN_tf(formatting, MinMaxScale):

    """
    Generative Adversarial Imputation Nets tensorflow version (GAIN)
    train Generator (G) and Discriminator (D) to impute missing values [1]

    [1] Yoon, J., Jordon, J. and Schaar, M., 2018, July. Gain: Missing data imputation using
    generative adversarial nets. In International Conference on Machine Learning (pp. 5689-5698). PMLR.
    github.com/jsyooon0823/GAIN

    Parameters
    ----------
    batch_size: mini-batch sample size, default = 128

    hint_rate: hint rate, default = 0.9

    alpha: penalty in optimizing Generator, default = 100

    optim: not supported for tensorflow version

    lr: not supported for tensorflow version

    max_iter: maximum number of iterations, default = 100

    delta: not supported for tensorflow version

    scaling: whether scale the dataset before imputation, default = True

    deep_copy: whether to deep copy dataframe, deafult = False

    seed: random seed, default = 1
    """

    def __init__(
        self,
        batch_size=128,
        hint_rate=0.9,
        alpha=100,
        optim=None,
        lr=None,
        max_iter=100,
        delta=1e-8,
        scaling=True,
        progressbar=False,
        deep_copy=False,
        seed=1,
    ):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.delta = delta
        self.scaling = scaling
        self.progressbar = progressbar
        self.deep_copy = deep_copy
        self.seed = seed

    def mask_matrix(self, X):

        """
        mask matrix, m_{ij} = 1 where x_{ij} exists; m_{ij} = 0 otherwise
        """
        return 1 - X.isnull().astype(int)

    # initialize normal tensor by size
    def normal_initial(self, size):

        _dim = size[0]
        return tf.random.normal(shape=size, stddev=1 / tf.sqrt(_dim / 2))

    # return random binary array by size
    def binary_sampler(self, p=0.5, size=(1, 1)):

        # allows only change row size with (n, )
        # cannot handle (, n)
        try:
            if size[0] == None:
                size[0] == 1
            elif size[1] == None:
                size[1] == 1
        except IndexError:
            size = (size[0], 1)

        _random_unit = np.random.uniform(low=0, high=1, size=size)
        return 1 * (_random_unit < p)

    # return random uniform array by size
    def uniform_sampler(self, low=0, high=1, size=(1, 1)):

        # allows only change row size with (n, )
        # cannot handle (, n)
        try:
            if size[0] == None:
                size[0] == 1
            elif size[1] == None:
                size[1] == 1
        except IndexError:
            size = (size[0], 1)

        return np.random.uniform(low=low, high=high, size=size)

    # Generator
    def Generator(self, data, mask):

        G_W1, G_W2, G_W3, G_b1, G_b2, G_b3 = self.theta_G
        _input = tf.concat(values=[data, mask], axis=1)  # concate data with mask
        G_h1 = tf.nn.relu(tf.matmul(_input, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_pro = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)  # MinMax normalization

        return G_pro

    # Discriminator
    def Discriminator(self, data, hint):

        D_W1, D_W2, D_W3, D_b1, D_b2, D_b3 = self.theta_D
        _input = tf.concat(values=[data, hint], axis=1)  # concate data with mask
        D_h1 = tf.nn.relu(tf.matmul(_input, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_pro = tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3)  # MinMax normalization

        return D_pro

    def fill(self, X):

        _X = X.copy(deep=True)

        if _X.isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No nan values found, no change.")

        return _X

    def _fill(self, data):

        _data = data.copy(deep=self.deep_copy)

        n, p = _data.shape
        _h_dim = int(p)  # Hidden state dimensions

        _mask = self.mask_matrix(_data).values

        # convert categorical to numerical
        formatter = formatting(inplace=True)
        formatter.fit(_X)

        # if scaling, use MinMaxScale to scale the features
        if self.scaling:
            scaling = MinMaxScale()
            _X = scaling.fit_transform(_X)

        _data_scaled = _data_scaled.fillna(0)

        # divide dataframe to np array for values and features names list
        _features = list(_data_scaled.columns)
        _data_scaled = _data_scaled.values

        # GAIN architecture
        _X = tf.compat.v1.placeholder(tf.float32, shape=[None, p])  # data
        _M = tf.compat.v1.placeholder(tf.float32, shape=[None, p])  # mask vector
        _H = tf.compat.v1.placeholder(tf.float32, shape=[None, p])  # hint vector

        # Generator Variables
        G_W1 = tf.Variable(self.normal_initial([p * 2, _h_dim]))
        G_b1 = tf.Variable(tf.zeros(shape=[_h_dim]))
        G_W2 = tf.Variable(self.normal_initial([_h_dim, _h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape=[_h_dim]))
        G_W3 = tf.Variable(self.normal_initial([_h_dim, p]))
        G_b3 = tf.Variable(tf.zeros(shape=[p]))

        self.theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        # Discriminator Varaibles
        D_W1 = tf.Variable(self.normal_initial([p * 2, _h_dim]))
        D_b1 = tf.Variable(tf.zeros(shape=[_h_dim]))
        D_W2 = tf.Variable(self.normal_initial([_h_dim, _h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape=[_h_dim]))
        D_W3 = tf.Variable(self.normal_initial([_h_dim, p]))
        D_b3 = tf.Variable(tf.zeros(shape=[p]))

        self.theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

        # GAIN structure
        _G = self.Generator(_X, _M)  # Generator
        _hat_X = _X * _M + _G * (1 - _M)  # combine mask with observed data
        _D = self.Discriminator(_hat_X, _H)  # Discriminator

        _D_loss_tmp = -tf.reduce_mean(
            _M * tf.compat.v1.log(_D + 1e-8)
            + (1 - _M) * tf.compat.v1.log(1.0 - _D + 1e-8)
        )  # Discriminator loss
        _G_loss_tmp = -tf.reduce_mean(
            (1 - _M) * tf.compat.v1.log(_D + 1e-8)
        )  # Generator loss
        _MSE_loss = tf.reduce_mean((_M * _X - _M * _G) ** 2) / tf.reduce_mean(_M)
        _D_loss = _D_loss_tmp
        _G_loss = _G_loss_tmp + self.alpha * _MSE_loss

        # GAIN solver
        _G_solver = tf.compat.v1.train.AdamOptimizer().minimize(
            _D_loss, var_list=self.theta_G
        )
        _D_solver = tf.compat.v1.train.AdamOptimizer().minimize(
            _G_loss, var_list=self.theta_D
        )

        # Iterations
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        _seed = self.seed  # initialize random seed

        # training step
        iterator = (
            tqdm(range(self.max_iter)) if self.progressbar else range(self.max_iter)
        )
        for _run in iterator:

            batch_index = random_index(
                self.batch_size, n, seed=_seed
            )  # random sample batch
            _X_mb = _data_scaled[batch_index, :]
            _M_mb = _mask[batch_index, :]
            _Z_mb = self.uniform_sampler(
                low=0, high=0.01, size=(self.batch_size, p)
            )  # random sample vector
            _H_mb_1 = self.binary_sampler(p=self.hint_rate, size=(self.batch_size, p))
            _H_mb = _M_mb * _H_mb_1  # sample hint vectors

            # combine random sample vector with observed data
            _X_mb = _M_mb * _X_mb + (1 - _M_mb) * _Z_mb
            _, _D_loss_now = sess.run(
                [_D_solver, _D_loss_tmp], feed_dict={_M: _M_mb, _X: _X_mb, _H: _H_mb}
            )
            _, _G_loss_now, _MSE_loss_now = sess.run(
                [_G_solver, _G_loss_tmp, _MSE_loss],
                feed_dict={_M: _M_mb, _X: _X_mb, _H: _H_mb},
            )

            _seed += 1

        # return imputed data
        _Z_mb = self.uniform_sampler(low=0, high=0.01, size=(n, p))
        _M_mb = _mask
        _X_mb = _data_scaled
        _X_mb = _M_mb * _X_mb + (1 - _M_mb) * _Z_mb

        _imputed_data = sess.run([_G], feed_dict={_X: _X_mb, _M: _M_mb})[0]
        _imputed_data = _mask * _data_scaled + (1 - _mask) * _imputed_data

        # combine data with column names to dataframe
        _imputed_data = pd.DataFrame(_imputed_data, columns=_features)

        # convert self._fitted and store self.train
        self._fitted = True

        # if scaling, scale back
        if self.scaling:
            _X = scaling.inverse_transform(_X)

        # convert numerical back to categorical
        formatter.refit(_X)

        return _imputed_data


class GAIN_torch(formatting, MinMaxScale):

    """
    Generative Adversarial Imputation Nets (GAIN) pytorch version [1]
    train Generator (G) and Discriminator (D) to impute missing values

    [1] Yoon, J., Jordon, J. and Schaar, M., 2018, July. Gain: Missing data imputation using
    generative adversarial nets. In International Conference on Machine Learning (pp. 5689-5698).
    PMLR. github.com/jsyooon0823/GAIN

    Parameters
    ----------
    batch_size: mini-batch sample size, default = 128

    hint_rate: hint rate, default = 0.9

    alpha: penalty in optimizing Generator, default = 100

    optim: optimization algorithms, default = 'Adam'
    support ["Adam", "SGD", "Adagrad", "LBFGS", "RMSprop"]

    lr: learning rate, default: None
    default lr will depend on optimizer
    for 'LBFGS', default lr = 1
    for 'Adam', default lr = 0.001
    else, default lr = 0.01

    max_iter: maximum number of iterations, default = 100

    delta: training early stopping criteria, default = 1e-8
    if changing percentage not significant, early stop training

    scaling: whether scale the dataset before imputation, default = True

    deep_copy: whether to deep copy dataframe, deafult = False

    seed: random seed, default = 1
    """

    def __init__(
        self,
        batch_size: int = 128,
        hint_rate: float = 0.9,
        alpha: int = 100,
        optim: str = "Adam",
        lr: float = None,
        max_iter: int = 100,
        delta: float = 1e-8,
        scaling: bool = True,
        progressbar: bool = False,
        deep_copy: bool = False,
        seed: int = 1,
    ) -> None:
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.optim = optim

        # default learning rate dependent on optimizer
        if self.optim == "LBFGS" and not lr:
            self.lr = 1
        elif self.optim == "Adam" and not lr:
            self.lr = 0.001
        elif not lr:
            self.lr = 0.01
        else:
            self.lr = lr

        self.max_iter = max_iter
        self.delta = delta
        self.scaling = scaling
        self.progressbar = progressbar
        self.deep_copy = deep_copy
        self.seed = seed

        np.random.seed(self.seed)

        self._fitted = False  # whether fitted on train set

    # get random m integer number in range [0, n - 1]
    def random_index(self, n: int, m: int) -> np.ndarray:

        return np.random.permutation(n)[:m]

    # initialize Generator/Discriminator variables
    def _initialization(self, p: int, h_dim: int) -> None:

        # W with random normal initialization and b with zero initialization
        # initialize Generator variables
        G_W1 = nn.init.normal_(
            torch.empty(2 * p, h_dim, dtype=torch.double, requires_grad=True)
        )
        G_b1 = torch.zeros(h_dim, dtype=torch.double, requires_grad=True)

        G_W2 = nn.init.normal_(
            torch.empty(h_dim, h_dim, dtype=torch.double, requires_grad=True)
        )
        G_b2 = torch.zeros(h_dim, dtype=torch.double, requires_grad=True)

        G_W3 = nn.init.normal_(
            torch.empty(h_dim, p, dtype=torch.double, requires_grad=True)
        )
        G_b3 = torch.zeros(p, dtype=torch.double, requires_grad=True)

        self.theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        # initialize Discriminator variables
        D_W1 = nn.init.normal_(
            torch.empty(2 * p, h_dim, dtype=torch.double, requires_grad=True)
        )
        D_b1 = torch.zeros(h_dim, dtype=torch.double, requires_grad=True)

        D_W2 = nn.init.normal_(
            torch.empty(h_dim, h_dim, dtype=torch.double, requires_grad=True)
        )
        D_b2 = torch.zeros(h_dim, dtype=torch.double, requires_grad=True)

        D_W3 = nn.init.normal_(
            torch.empty(h_dim, p, dtype=torch.double, requires_grad=True)
        )
        D_b3 = torch.zeros(p, dtype=torch.double, requires_grad=True)

        self.theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator network structure
    def Generator(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        G_W1, G_W2, G_W3, G_b1, G_b2, G_b3 = self.theta_G
        _input = torch.cat(tensors=[data, mask], dim=1)  # concate data with mask
        G_h1 = F.relu(torch.matmul(_input, G_W1) + G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)
        G_pro = torch.sigmoid(
            torch.matmul(G_h2, G_W3) + G_b3
        )  # normalize to probability

        return G_pro

    # Discriminator network structure
    def Discriminator(self, data: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:

        D_W1, D_W2, D_W3, D_b1, D_b2, D_b3 = self.theta_D
        _input = torch.cat(tensors=[data, hint], dim=1)  # concate data with hint matrix
        D_h1 = F.relu(torch.matmul(_input, D_W1) + D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
        D_pro = torch.sigmoid(
            torch.matmul(D_h2, D_W3) + D_b3
        )  # normalize to probability

        return D_pro

    # Generator loss
    def network_loss(
        self, X: torch.Tensor, M: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:

        _G = self.Generator(X, M)
        _hat_X = X * M + _G * (1 - M)
        _D = self.Discriminator(_hat_X, H)

        # Discriminator loss
        _D_loss = -torch.mean(
            M * torch.log(_D + 1e-8) + (1 - M) * torch.log(1.0 - _D + 1e-8)
        )
        # Generator loss
        _G_loss_1 = -torch.mean((1 - M) * torch.log(_D + 1e-8))
        _G_loss_2 = torch.mean((M * X - M * _G) ** 2) / torch.mean(M)
        _G_loss = _G_loss_1 + self.alpha * _G_loss_2

        return _G_loss, _D_loss

    def fill(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> pd.DataFrame:

        # make sure input is a dataframe
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except:
                raise TypeError("Expect a dataframe, get {}.".format(type(X)))

        _X = X.copy(deep=self.deep_copy)

        if _X.isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No missing values found, no change.")

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:

        _X = X.copy(deep=self.deep_copy)

        n, p = _X.shape  # get shape of dataset
        h_dim = int(p)  # get hidden state dimensions

        # make sure batch size is smaller than number of observations
        self.batch_size = min(self.batch_size, n)

        # convert categorical to numerical
        formatter = formatting(inplace=True)
        formatter.fit(_X)

        # if scaling, use MinMaxScale to scale the features
        if self.scaling:
            scaling = MinMaxScale()
            _X = scaling.fit_transform(_X)

        # GAIN imputation

        # initialization: fill missing with 0
        _X = _X.fillna(0)

        # divide dataframe to list of features and array of values
        _features = list(_X.columns)
        _X = _X.values
        # get mask matrix
        _M = get_missing_matrix(_X, missing=0)

        # if not fitted, train the networks
        if not self._fitted:
            # initialize Generator/Discriminator variables
            self._initialization(p, h_dim)

            # network optimizer
            if self.optim == "Adam":
                optimizer = torch.optim.Adam(
                    params=self.theta_D + self.theta_G, lr=self.lr
                )
            elif self.optim == "SGD":
                optimizer = torch.optim.SGD(
                    params=self.theta_D + self.theta_G, lr=self.lr
                )
            elif self.optim == "Adagrad":
                optimizer = torch.optim.Adagrad(
                    params=self.theta_D + self.theta_G, lr=self.lr
                )
            elif self.optim == "LBFGS":
                optimizer = torch.optim.LBFGS(
                    params=self.theta_D + self.theta_G, lr=self.lr
                )
            elif self.optim == "RMSprop":
                optimizer = torch.optim.RMSprop(
                    params=self.theta_D + self.theta_G, lr=self.lr
                )
            else:
                raise KeyError(
                    'Get unknown optimizer {}, should be one of ["Adam", "SGD", \
                    "Adagrad", "LBFGS", "RMSprop"].'.format(
                        self.optim
                    )
                )

            # initialize parameters to device
            self.theta_D = [item.to(device) for item in self.theta_D]
            self.theta_G = [item.to(device) for item in self.theta_G]

            # store the losses for early_stopping
            self.losses = []

            # training step
            iterator = (
                tqdm(range(self.max_iter)) if self.progressbar else range(self.max_iter)
            )
            for _ in iterator:

                # get mini-batch data
                batch_index = self.random_index(n, self.batch_size)
                _X_mb = _X[batch_index, :]
                _M_mb = _M[batch_index, :]  # mini-batch mask matrix
                # mini-batch random imputation
                _Z_mb = np.random.uniform(low=0, high=0.01, size=(self.batch_size, p))
                _H_mb_1 = 1 * (
                    np.random.uniform(0, 1, size=(self.batch_size, p)) < self.hint_rate
                )
                _H_mb = _M_mb * _H_mb_1  # mini-batch hint matrix

                # combine random imputation with data
                _X_mb = _M_mb * _X_mb + (1 - _M_mb) * _Z_mb

                # matrix to tensor
                _X_mb = torch.tensor(_X_mb, dtype=torch.double, device=device)
                _M_mb = torch.tensor(_M_mb, dtype=torch.double, device=device)
                _H_mb = torch.tensor(_H_mb, dtype=torch.double, device=device)

                # get the losses
                # combine two losses as one
                G_loss, D_loss = self.network_loss(_X_mb, _M_mb, _H_mb)
                loss = G_loss + D_loss

                # optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # early_stopping
                self.losses.append(loss.item())
                if len(self.losses) > 1:
                    # if losses changing not significant,
                    # early stop
                    if (
                        np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-2]
                        < self.delta
                    ):
                        # for tqdm, to break, need manual close
                        if self.progressbar:
                            iterator.close()
                        break

                # display loss
                if self.progressbar:
                    iterator.set_postfix({"loss": loss.item()})
                    iterator.refresh()  # to show immediately the update
                    sleep(0.01)

        # impute the missing data
        # _X or _M not tensor, convert to tensor
        if not torch.is_tensor(_X):
            _X = torch.tensor(_X, dtype=torch.double, device=device)
        if not torch.is_tensor(_M):
            _M = torch.tensor(_M, dtype=torch.double, device=device)

        # impute using trained Generator
        with torch.no_grad():
            _X = _M * _X + (1 - _M) * self.Generator(_X, _M)

        # if tensor, detach to numpy array
        if torch.is_tensor(_X):
            _X = _X.cpu().detach().numpy()

        # convert back to dataframe
        _X = pd.DataFrame(_X, columns=_features)

        # convert self._fitted and store self.train
        self._fitted = True

        # if scaling, scale back
        if self.scaling:
            _X = scaling.inverse_transform(_X)

        # convert numerical back to categorical
        formatter.refit(_X)

        return _X


class GAIN(GAIN_tf, GAIN_torch):

    """
    Generative Adversarial Imputation Nets (GAIN) [1]
    use pytorch/tensorflow when available
    train Generator (G) and Discriminator (D) to impute missing values

    [1] Yoon, J., Jordon, J. and Schaar, M., 2018, July. Gain: Missing data imputation using
    generative adversarial nets. In International Conference on Machine Learning (pp. 5689-5698).
    PMLR. github.com/jsyooon0823/GAIN

    Parameters
    ----------
    batch_size: mini-batch sample size, default = 128

    hint_rate: hint rate, default = 0.9

    alpha: penalty in optimizing Generator, default = 100

    optim: optimization algorithms, default = 'Adam'
    support ["Adam", "SGD", "Adagrad", "LBFGS", "RMSprop"]

    lr: learning rate, default: None
    default lr will depend on optimizer
    for 'LBFGS', default lr = 1
    for 'Adam', default lr = 0.001
    else, default lr = 0.01

    max_iter: maximum number of iterations, default = 100

    delta: training early stopping criteria, default = 1e-8
    if changing percentage not significant, early stop training

    scaling: whether scale the dataset before imputation, default = True

    deep_copy: whether to deep copy dataframe, deafult = False

    seed: random seed, default = 1
    """

    def __init__(
        self,
        batch_size: int = 128,
        hint_rate: float = 0.9,
        alpha: int = 100,
        optim: str = "Adam",
        lr: float = None,
        max_iter: int = 100,
        delta: float = 1e-8,
        scaling: bool = True,
        progressbar: bool = False,
        deep_copy: bool = False,
        seed: int = 1,
    ) -> None:

        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.optim = optim
        self.lr = lr
        self.max_iter = max_iter
        self.delta = delta
        self.scaling = scaling
        self.progressbar = progressbar
        self.deep_copy = deep_copy
        self.seed = seed

        np.random.seed(self.seed)

        self._fitted = False  # whether fitted on train set

        if torch_spec is not None:
            self.model = GAIN_torch(
                batch_size=self.batch_size,
                hint_rate=self.hint_rate,
                alpha=self.alpha,
                optim=self.optim,
                lr=self.lr,
                max_iter=self.max_iter,
                delta=self.delta,
                scaling=self.scaling,
                progressbar=self.progressbar,
                deep_copy=self.deep_copy,
                seed=self.seed,
            )
        elif tensorflow_spec is not None:
            self.model = GAIN_tf(
                batch_size=self.batch_size,
                hint_rate=self.hint_rate,
                alpha=self.alpha,
                optim=self.optim,
                lr=self.lr,
                max_iter=self.max_iter,
                delta=self.delta,
                scaling=self.scaling,
                progressbar=self.progressbar,
                deep_copy=self.deep_copy,
                seed=self.seed,
            )
        else:
            raise Exception(
                "No tensorflow or torch installed. This method is not supported."
            )

    def fill(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> pd.DataFrame:

        self._fitted = True

        return self.model.fill(X)
