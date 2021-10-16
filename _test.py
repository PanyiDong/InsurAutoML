import numpy as np
import pandas as pd
import itertools

import scipy.sparse.linalg
from scipy.sparse.linalg import svds

from sklearn.metrics import mean_squared_error

a = [0, 0, 0, 0, 0]
print(a.count(a[0]) == len(a))

_sel = [1, 0, 1]
X = pd.DataFrame({
    '1' : [1, 2, 3],
    '2' : [4, 5, 6],
    '3' : [7, 8, 9]
})

columns = []
for i in range(len(_sel)) :
    if _sel[i] == 1:
        columns.append(i)

print(X.iloc[:, columns])