"""
File Name: twinreduction.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/ext/twinreduction/twinreduction/twinreduction.py
File Created: Thursday, 6th November 2025 5:56:27 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 6th November 2025 8:02:14 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025 - 2025, Panyi Dong

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

import numpy as np
import math
from ...twinreduction.twinning_cpp import twin_cpp


def _data_format(data):
    const_cols = np.all(data == data[0, :], axis=0)
    data = data[:, np.invert(const_cols)]
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    if data.data.c_contiguous:
        return data
    else:
        return np.copy(data, order="C")


def twin(data, r: int = 10, u1: int = None, leaf_size: int = 8):
	"""Return indices of twin blocks in the data.

	Parameters
	----------
	data : ndarray
		2-D input data (n x p)
	r : int
		inverse splitting ratio (e.g., r=5 for 80/20)
	u1 : int, optional
		starting index for deterministic behavior
	leaf_size : int
		kd-tree leaf size

	Returns
	-------
	list[np.ndarray]
		A list of 1-D numpy arrays (dtype=uint64), each containing indices of a twin block
	"""

	if type(data) != np.ndarray or len(data.shape) != 2:
		raise Exception("data is expected to be a 2 dimensional numpy ndarray")

	if np.isnan(data).any() or np.isinf(data).any():
		raise Exception("data cannot contain nan or infinity")

	if u1 is None:
		u1 = np.random.randint(data.shape[0])
	elif u1 not in range(data.shape[0]):
		raise Exception("u1 should be a row index such that 0 <= u1 < data.shape[0]")

	if r not in range(2, math.floor(data.shape[0] / 2) + 1):
		raise Exception("r should be an integer such that 2 <= r <= data.shape[0]/2")

	data = _data_format(data)
	# twin_cpp now returns a list of numpy arrays, each containing indices for a twin block.
	groups = twin_cpp(data, r, u1, leaf_size)
	# normalize dtype and memory layout for each group's array
	return np.array([np.asarray(g) for g in groups])


def TwinReduction(data, missing_idx, r: int = 10, u1: int = None, leaf_size: int = 8):
	"""Use missing indices to perform data reduction via twinning.

	Parameters
	----------
	data : ndarray
		Input data used to compute twin blocks
	missing_idx : array_like
		Boolean 1-D mask of length n*p (flattened) indicating missing positions
	r, u1, leaf_size : as in :func:`twin`

	Returns
	-------
	ndarray
		Flattened array of retained indices after reduction
	"""
	# convert data to numpy array if needed
	data = np.array(data) if not isinstance(data, np.ndarray) else data
	# Get twin indices (list of 1-D numpy arrays)
	twin_idx = twin(data, r, u1, leaf_size)
	# Vectorized mapping: get boolean mask for twin positions referenced by missing_idx
	# Sum along axis 1 to get a boolean mask for each twin block
	twin_missing = missing_idx[twin_idx].sum(axis = 1)
	# If the twin block have both missing and observed, remove entire block
	idx_reduced = np.array([twin_idx[i] for i in range(len(twin_idx)) if twin_missing[i] == 0 or twin_missing[i] == len(twin_idx[i])]).flatten()
	
	return data[idx_reduced, :]


