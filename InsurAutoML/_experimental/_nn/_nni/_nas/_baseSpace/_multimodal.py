"""
File Name: _multimodal.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_experimental/_nn/_nni/_nas/_baseSpace/_multimodal.py
File Created: Sunday, 13th November 2022 5:15:00 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:26:21 pm
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

"""
List of methods:
                    Txt           Cat           Con
FusToken:      tokenize      encoding      encoding
FusEmbed:     embedding     embedding     embedding
FusModel:         model         model         model

FusToken: Txt/Cat/Con are only tokenized to numerics and fuse those heads together for whole model training.
FusEmbed: Txt/Cat/Con are further embedded with a few layers of NNs and fuse those embeddings together for later half of model training.
FusModel: Txt/Cat/Con are almost trained separately and just fused together for unified prediction.
"""

"""
List of classes need for multimodal neural architecture search.
1. entire network space (whole vs lint)
2. heads (before fuse)
3. prediction (after fuse for FusModel)
"""
