import numpy as np
import pandas as pd
import itertools

import scipy.sparse.linalg
from scipy.sparse.linalg import svds

a = [[1, 2, 3, 4], [5, 6, 7, 8]]
print((np.array(a) - np.mean(a, axis = 0)) ** 2)