from re import A
import numpy as np
import pandas as pd

a = pd.DataFrame({
    'name' : [4, 2, 5, 7, 8, 9, 9],
    'value' : [1, 2, 3, 2, 1, 2, 2]
})
c = pd.DataFrame({
    'response' : [1, 2, 3, 4, 5, 6, 7]
})

b = a.groupby(['name', 'value']).size().div(len(a))
d = a.groupby('name').size().div(len(a))
print(d.loc[d.index == b.index[0][0]])
