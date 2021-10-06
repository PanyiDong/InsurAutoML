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
    '1' : [1, 2, 3, 4],
    '2' : [4, 5, 6, 7]
})
print((a['1'] != a['2']).astype(int).sum())