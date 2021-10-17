import numpy as np
import pandas as pd
import itertools

import scipy.sparse.linalg
from scipy.sparse.linalg import svds

from sklearn.metrics import mean_squared_error
print(np.random.rand())

a = [0.1, 0.4, 0.3, 0.2]
b = [0.05, 0.35, 0.5, 0.1]

c = a[:2] + b[2:]
d = b[:2] + a[2:]
print(c, d)