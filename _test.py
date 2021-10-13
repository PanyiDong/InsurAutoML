import numpy as np
import pandas as pd
import itertools

import scipy.sparse.linalg
from scipy.sparse.linalg import svds

from sklearn.metrics import mean_squared_error
obj = - mean_squared_error
a = [1, 2, 3, 4]
b = [3, 4, 5, 2]
print(obj(a, b))