"""
File: _nn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_imputation/_nn.py
File Created: Tuesday, 5th April 2022 11:50:10 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 8th April 2022 10:25:28 pm
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
    from tensorflow.python.types.core import Value

    tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.disable_v2_behavior() # use tf < 2.0 functions

from My_AutoML._utils import random_index, feature_rounding
from My_AutoML._scaling import MinMaxScale


class GAIN:

    """
    Generative Adversarial Imputation Nets (GAIN)
    train Generator (G) and Discriminator (D) to impute missing values [1]

    [1] Yoon, J., Jordon, J. and Schaar, M., 2018, July. Gain: Missing data imputation using
    generative adversarial nets. In International Conference on Machine Learning (pp. 5689-5698). PMLR.
    github.com/jsyooon0823/GAIN

    Parameters
    ----------
    batch_size: sampling size from data

    hint_rate: hint rate

    alpha: penalty in optimizing Generator

    iterations: number of iterations

    uni_class: unique classes in a column which will be considered as categorical class, default = 31
    round numerical to categorical in case after the imputation, the data type changes

    seed: random seed
    """

    def __init__(
        self,
        batch_size=128,
        hint_rate=0.9,
        alpha=100,
        iterations=10000,
        uni_class=31,
        seed=1,
    ):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.uni_class = uni_class
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

        _data = data.copy(deep=True)
        n, p = _data.shape

        _h_dim = int(p)  # Hidden state dimensions

        _mask = self.mask_matrix(_data).values
        # scaling data to [0, 1]
        scaler = MinMaxScale()
        scaler.fit(_data)
        _data_scaled = scaler.transform(_data)
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

        for _run in tqdm(range(self.iterations)):

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

        # Unscale the imputed data
        _imputed_data = scaler.inverse_transform(_imputed_data)
        _imputed_data = feature_rounding(_imputed_data)

        return _imputed_data
