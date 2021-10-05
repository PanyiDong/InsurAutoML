import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import time

class RBFSampler() :

    '''
    Implement of Weighted Sums of Random Kitchen Sinks

    Parameters
    ----------
    gamma: use to determine standard variance of random weight table, default = 1
    Parameter of RBF kernel: exp(-gamma * x^2).

    n_components: number of samples per original feature, default = 100

    seed: random generation seed, default = None
    '''

    def __init__(self, gamma = 1., n_components = 100, seed = None):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

    def fit(self, X) :
        
        if isinstance(X, list) :
            n_features = len(X[0])
        else :
            n_features = X.shape[1]

        if not self.seed :
            self.seed = np.random.seed(int(time.time()))
        elif not isinstance(self.seed, int) :
            raise ValueError('Seed must be integer, receive {}'.format(self.seed))

        self._random_weights = np.random.normal(0, np.sqrt(2 * self.gamma), size = (n_features, self.n_components))
        self._random_offset = np.random.uniform(0, 2 * np.pi, size = (1, self.n_components))

        return self

    def transform(self, X) :

        projection = np.dot(X, self._random_weights)
        projection += self._random_offset
        np.cos(projection, projection)
        projection *= np.sqrt(2.0 / self.n_components)

        return projection