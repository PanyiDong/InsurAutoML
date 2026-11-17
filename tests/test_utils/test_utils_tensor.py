"""
File Name: test_utils_tensor.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /tests/test_utils/test_utils_tensor.py
File Created: Monday, 16th November 2026 7:12:56 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:37:11 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2026, Panyi Dong

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

import os

import pytest

torch = pytest.importorskip("torch")

from InsurAutoML.utils.tensor import (  # noqa: E402
    repackage_hidden,
    CustomTensorDataset,
    ListTensorDataset,
    DictTensorDataset,
    SerialTensorDataset,
)


def test_repackage_hidden_tensor_and_tuple():
    w = torch.ones(3, requires_grad=True)
    h = w * 2
    assert h.requires_grad
    out = repackage_hidden(h)
    assert not out.requires_grad
    assert torch.equal(out, h.detach())

    h_tuple = (w * 2, (w * 3, w * 4))
    out_tuple = repackage_hidden(h_tuple)
    assert isinstance(out_tuple, tuple) and isinstance(out_tuple[1], tuple)
    assert not out_tuple[0].requires_grad
    assert not out_tuple[1][1].requires_grad
    assert torch.equal(out_tuple[1][1], torch.full((3,), 4.0))


@pytest.mark.parametrize("fmt", ["tuple", "list", "dict"])
def test_custom_tensor_dataset(fmt):
    inputs = torch.arange(20, dtype=torch.float).reshape(10, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    ds = CustomTensorDataset(inputs, labels, format=fmt)
    assert len(ds) == 10
    assert ds.inputSize == 2
    assert ds.outputSize == 3
    item = ds[3]
    if fmt == "dict":
        assert torch.equal(item["input"], inputs[3]) and item["label"] == 0
    else:
        assert isinstance(item, tuple if fmt == "tuple" else list)
        assert torch.equal(item[0], inputs[3]) and item[1] == 0


def test_custom_tensor_dataset_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        CustomTensorDataset(torch.zeros(5, 2), torch.zeros(4))


@pytest.mark.parametrize("fmt", ["tuple", "list", "dict"])
def test_list_tensor_dataset(tmp_path, fmt):
    a = torch.arange(10, dtype=torch.float).reshape(5, 2)
    b = torch.arange(15, dtype=torch.float).reshape(5, 3)
    labels = torch.tensor([1, 0, 1, 0, 1])
    ds = ListTensorDataset([a, b], labels, path=str(tmp_path), format=fmt)
    assert ds._input_format == "list"
    assert os.path.exists(tmp_path / "input" / "0.pt")
    assert os.path.exists(tmp_path / "input" / "1.pt")
    assert os.path.exists(tmp_path / "labels.pt")
    assert len(ds) == 5

    item = ds[2]
    inp, lab = (item["input"], item["label"]) if fmt == "dict" else item
    if fmt != "dict":
        assert isinstance(item, tuple if fmt == "tuple" else list)
    assert torch.equal(inp[0], a[2]) and torch.equal(inp[1], b[2])
    assert lab == 1


def test_list_tensor_dataset_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        ListTensorDataset([torch.zeros(5, 2)], torch.zeros(3), path=str(tmp_path))


@pytest.mark.parametrize("fmt", ["tuple", "list", "dict"])
def test_dict_tensor_dataset(tmp_path, fmt):
    inputs = {"con": torch.ones(4, 2), "cat": torch.zeros(4, 3)}
    labels = torch.tensor([0, 1, 0, 1])
    ds = DictTensorDataset(inputs, labels, path=str(tmp_path), format=fmt)
    assert ds._input_format == "dict"
    assert os.path.exists(tmp_path / "input" / "con.pt")
    assert os.path.exists(tmp_path / "input" / "cat.pt")
    assert len(ds) == 4

    item = ds[1]
    inp, lab = (item["input"], item["label"]) if fmt == "dict" else item
    assert torch.equal(inp[0], torch.ones(2))
    assert torch.equal(inp[1], torch.zeros(3))
    assert lab == 1


def test_dict_tensor_dataset_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        DictTensorDataset({"a": torch.zeros(5, 2)}, torch.zeros(3), path=str(tmp_path))


def test_serial_tensor_dataset(tmp_path):
    labels = torch.tensor([0, 1, 2])

    # single tensor -> stored as a one-element list
    single = SerialTensorDataset(torch.eye(3), labels, path=str(tmp_path / "single"))
    assert single._input_format == "list"
    assert len(single) == 3
    inp, lab = single[2]
    assert len(inp) == 1 and torch.equal(inp[0], torch.eye(3)[2]) and lab == 2

    # list of tensors
    lst = SerialTensorDataset(
        [torch.eye(3), torch.ones(3, 2)], labels, path=str(tmp_path / "list")
    )
    assert lst._input_format == "list"
    assert len(lst[0][0]) == 2

    # dict of tensors
    dct = SerialTensorDataset(
        {"x": torch.eye(3)}, labels, path=str(tmp_path / "dict"), format="dict"
    )
    assert dct._input_format == "dict"
    item = dct[1]
    assert torch.equal(item["input"][0], torch.eye(3)[1]) and item["label"] == 1
