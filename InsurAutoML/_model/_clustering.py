"""
File: _clustering.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_model/_clustering.py
File Created: Monday, 20th June 2022 12:21:59 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 20th June 2022 12:25:31 am
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

from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    SpectralBiclustering,
    MiniBatchKMeans,
    MeanShift,
    OPTICS,
    SpectralClustering,
    SpectralBiclustering,
    SpectralCoclustering,
)

clusterings = {
    "AffinityPropagation": AffinityPropagation,
    "AgglomerativeClustering": AgglomerativeClustering,
    "Birch": Birch,
    "DBSCAN": DBSCAN,
    "KMeans": KMeans,
    "SpectralBiclustering": SpectralBiclustering,
    "MiniBatchKMeans": MiniBatchKMeans,
    "MeanShift": MeanShift,
    "OPTICS": OPTICS,
    "SpectralClustering": SpectralClustering,
    "SpectralBiclustering": SpectralBiclustering,
    "SpectralCoclustering": SpectralCoclustering,
}
