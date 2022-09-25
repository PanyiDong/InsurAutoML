"""
File: _base.pyx
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_base.pyx
File Created: Saturday, 24th September 2022 11:13:27 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 25th September 2022 11:12:40 am
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

import cython
import numpy as np

cimport numpy as cnp

cnp.import_array()

# cimport numpy as сnp
# cdef сnp.ndarray[double, ndim=1] c_arr

#######################################
# An object to be used in Python

# set response to [0, 1] class, random guess at 0.5
cpdef int random_guess(
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
cpdef cnp.ndarray random_index(
    int n, 
    int total=0, 
    int seed=1
):
    if seed is not None:
        np.random.seed(seed)
    if total == 0:
        total = n
    cdef cnp.ndarray output = np.zeros(total, dtype=float)
    cdef cnp.ndarray vlist = np.array([i for i in range(total)])
    for idx, _ in enumerate(range(n)):
        # np.random.seed(int(datetime.now().strftime("%H%M%S")))
        index = np.random.randint(0, high=len(vlist), size=1)[0]
        output[idx] = vlist[index]
        vlist = np.delete(vlist, index)

    return output