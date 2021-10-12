import numpy as np
import pandas as pd
import itertools

import scipy.sparse.linalg
from scipy.sparse.linalg import svds

X = pd.DataFrame({
    'name' : [1, 2, 3, 4, 5],
    'type' : [5, 3, 5, 8, 9]
})
y = pd.DataFrame({
    'value' : [4, 4, 8, 9, 0]
})

_y_uni = np.unique(y)  # non-negative ints
print(_y_uni)
priors_ = []
for _value in _y_uni :
    priors_.append(y.loc[y.values == _value].count()[0] / len(y))

print(priors_)