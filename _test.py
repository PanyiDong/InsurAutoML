import numpy as np
import pandas as pd
import itertools

a = ['column1', 'column2']
b = ['column3', 'column4', 'column5']

c = list(itertools.combinations(b, 1))
d = [a + list(item) for item in c]
print(d)