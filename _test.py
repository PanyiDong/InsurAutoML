from re import A
import numpy as np
import pandas as pd

def uniform_sampler(low = 0, high = 1, size = (1, 1)) :
        
    # allows only change row size with (n, )
    # cannot handle (, n)
    try :
        if size[0] == None :
            size[0] == 1
        elif size[1] == None :
            size[1] == 1
    except IndexError :
        size = (size[0], 1)
        
    return np.random.uniform(low = low, high = high, size = size)

a = pd.DataFrame({
    '1' : [1, 2, 3, np.nan],
    '2' : [4, 5, np.nan, 7],
    '3' : [7, 8, np.nan, 19]
})

maskedarr1 = np.ma.array(a.loc[:, ['2', '3']], mask=np.isnan(a.loc[:, ['2', '3']]))
maskedarr2 = np.ma.array(a.loc[:, ['1']], mask=np.isnan(a.loc[:, ['1']]))
print(np.ma.cov(maskedarr1.T, y = maskedarr2.T))

a.loc[2, a.loc[2, :].isnull()] = [3, 4]
print(a)
