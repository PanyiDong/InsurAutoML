"""
File Name: _base.pyx
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_c/_base.pyx
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:16:52 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2022, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""




import warnings
import cython
import numpy as np
import pandas as pd

cimport numpy as cnp

cnp.import_array()

ctypedef cnp.npy_float FLOAT
ctypedef cnp.npy_intp INTP

# cimport numpy as сnp
# cdef сnp.ndarray[double, ndim=1] c_arr

#######################################
# An object to be used in Python

# set response to [0, 1] class, random guess at 0.5
cdef int random_guess(
    float number, 
    int seed= 1,
):    
    if seed != None:
        np.random.seed(seed)
    if number > 0.5:
        return 1
    elif number < 0.5:
        return 0
    else:
        return np.random.randint(0, 2)
    
# Return random index of a list (unique values only)
# from total draw n, default total = n
cdef cnp.ndarray random_index(
    int n, 
    int total=0, 
    int seed=1
):
    if seed is not None:
        np.random.seed(seed)
    if total == 0:
        total = n
    cdef cnp.ndarray output = np.zeros(total, dtype=int)
    
    if total < n:
        raise ValueError(
            "Total number of samples must be greater than or equal to the number of samples to be drawn. Got total={}, n={}".format(total, n)
        )
    elif n == 0 :
        raise ValueError(
            "Number of samples to be drawn must be greater than 0. Got n={}.".format(n)
        )
    
    output = np.random.choice(total, n, replace=False)
    
    return output

# Return randomly shuffle of a list (unique values only)
cdef cnp.ndarray random_list(
    cnp.ndarray[FLOAT, ndim = 1, mode = "c"] vlist, 
    int seed=1
):
    if seed != None:
        np.random.seed(seed)
    
    cdef cnp.ndarray output = np.zeros(len(vlist), dtype=float)
    output = np.random.permutation(vlist)

    return output

# Return location of minimum values
cdef int minloc(
    cnp.ndarray[FLOAT, ndim = 1, mode = "c"] vlist,
):
    
    # make sure input is np.array
    if not isinstance(vlist, np.ndarray):
        vlist = np.array(vlist)
        
    if len(vlist) == 0:
        raise ValueError("Invalid List!")
    else :
        # check whether multi-dimensional array
        if len((<object> vlist).shape) > 1:
            warnings.warn(
                "Input is multi-dimensional array, return min location of the first dimension."
            )
        
        return np.argmin(vlist, axis=0)


# Return location of maximum values
cdef int maxloc(
    cnp.ndarray[FLOAT, ndim = 1, mode = "c"] vlist,                         
):
    
    # make sure input is np.array
    if not isinstance(vlist, np.ndarray):
        vlist = np.array(vlist)
        
    if len(vlist) == 0:
        raise ValueError("Invalid List!")
    else :
        # check whether multi-dimensional array
        if len((<object> vlist).shape) > 1:
            warnings.warn(
                "Input is multi-dimensional array, return max location of the first dimension."
            )
        
        return np.argmax(vlist, axis=0)