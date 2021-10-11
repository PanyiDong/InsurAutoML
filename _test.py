import numpy as np
import pandas as pd
import itertools

a = ['column1', 'column2']
b = ['column1', 'column2', 'column3', 'column4', 'column5']
print([item for item in b if item not in a])

c = list(itertools.combinations(b, 1))
d = [a + list(item) for item in c]
print(d)