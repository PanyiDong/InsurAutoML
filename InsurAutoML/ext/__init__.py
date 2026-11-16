"""
File Name: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/ext/__init__.py
File Created: Thursday, 6th November 2025 6:55:05 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 6th November 2025 8:00:25 pm
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

from .._optional import INFORMED_EXTENSIONS, InformedNotInstalledError
from .._optional import INFORMED_INSTALL_CMD, INFORMED_REINSTALL_CMD

try:
    from .suppsplit import SPlit
except ImportError as error:
    # compiled extensions are only built with the informed installation
    raise InformedNotInstalledError(
        "InsurAutoML.ext requires the compiled extension {}, built by the "
        "InformedAutoML version installation:\n    {}\nIf InsurAutoML is already installed, "
        "rebuild it with:\n    {}".format(
            ", ".join(module for _, module in INFORMED_EXTENSIONS),
            INFORMED_INSTALL_CMD,
            INFORMED_REINSTALL_CMD,
        )
    ) from error
