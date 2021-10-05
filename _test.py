import numpy as np
import pandas as pd

a = pd.DataFrame({
    'name' : [1, np.nan, 2, np.nan],
    'value' : [np.nan, 4, 5, 6]
})

print(a.isnull().astype(int))