"""
File Name: test_utils_metadata_extra.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_utils/test_utils_metadata_extra.py
File Created: Monday, 16th November 2026 12:00:00 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 12:00:00 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2026, Panyi Dong

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

import numpy as np
import pandas as pd
import pytest

from InsurAutoML.constant import UNIQUE_FULLTYPE
from InsurAutoML.utils.metadata import (
    ChunkMetaData,
    MetaData,
    get_details,
    meta_map_int,
    meta_map_object,
)


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "int_cat": rng.integers(0, 3, size=n),
            "int_num": np.arange(n) + 1000 * seed,
            "float": rng.normal(size=n),
            "obj_cat": rng.choice(["a", "b", "c"], size=n),
            "obj_txt": ["this is sentence {} {}".format(seed, i) for i in range(n)],
        }
    )


def test_meta_map_helpers():
    assert meta_map_int(pd.Series([0, 1] * 50)) == "Categorical"
    assert meta_map_int(pd.Series(range(100))) == "Numerical"

    # few unique values -> categorical
    assert meta_map_object(pd.Series(["a", "b"] * 50)) == "Categorical"
    # many unique long strings -> text
    assert (
        meta_map_object(pd.Series(["w x y {}".format(i) for i in range(50)])) == "Text"
    )
    # many unique short strings -> categorical
    assert (
        meta_map_object(pd.Series(["w{}".format(i) for i in range(50)]))
        == "Categorical"
    )
    # many unique non-string objects: str accessor fails -> categorical
    assert meta_map_object(pd.Series(list(range(50)), dtype=object)) == "Categorical"


def test_get_details_get_and_merge():
    num = pd.Series([1.0, 2.0, 3.0])
    det = get_details("Float", "").get(num)
    assert det["length"] == 3 and det["mean"] == pytest.approx(2.0)
    assert det["min"] == 1.0 and det["max"] == 3.0

    new = pd.Series([4.0, 5.0])
    merged = get_details("Float", "").merge(new, det)
    full = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert merged["length"] == 5
    assert merged["mean"] == pytest.approx(full.mean())
    assert merged["std"] == pytest.approx(np.std(full))
    assert merged["min"] == 1.0 and merged["max"] == 5.0

    cat = pd.Series(["a", "b", "a"])
    det = get_details("Object", "Categorical").get(cat)
    assert det == {"unique_count": {"a": 2, "b": 1}}
    merged = get_details("Object", "Categorical").merge(pd.Series(["b", "c"]), det)
    assert merged == {"unique_count": {"a": 2, "b": 2, "c": 1}}

    assert get_details("Object", "Text").get(cat) == {}
    assert get_details("Datetime", "").merge(cat, {}) == {}


def test_metadata_requires_get_first():
    meta = MetaData()
    with pytest.raises(AttributeError, match="Metadata not generated"):
        repr(meta)
    with pytest.raises(AttributeError):
        str(meta)
    with pytest.raises(AttributeError):
        meta.update(pd.DataFrame({"a": [1]}))


def test_metadata_fulltypes_and_repr():
    df = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = MetaData(df)

    assert meta.metadata[("Int", "Categorical")] == ["int_cat"]
    assert meta.metadata[("Int", "Numerical")] == ["int_num"]
    assert meta.metadata[("Float", "")] == ["float"]
    assert meta.metadata[("Object", "Categorical")] == ["obj_cat"]
    assert meta.metadata[("Object", "Text")] == ["obj_txt"]
    assert str(meta) == repr(meta) == str(meta.metadata)
    assert meta.details["float"]["length"] == len(df)
    assert set(meta.details["obj_cat"]["unique_count"]) == {"a", "b", "c"}


def test_get_fulltype_special_dtypes(tmp_path):
    # float holding int values warns and is registered as Int
    with pytest.warns(UserWarning, match="actually int"):
        assert MetaData._get_fulltype(pd.Series([1.0, 2.0, np.nan] * 20, name="f")) == (
            "Int",
            "Categorical",
        )
    with pytest.warns(UserWarning, match="actually int"):
        assert MetaData._get_fulltype(
            pd.Series(np.arange(200, dtype=float), name="f")
        ) == ("Int", "Numerical")
    # genuine floats stay Float
    assert MetaData._get_fulltype(pd.Series([0.5, 1.25, 2.0], name="f")) == (
        "Float",
        "",
    )

    dt = pd.Series(pd.date_range("2020-01-01", periods=5), name="d")
    assert MetaData._get_fulltype(dt) == ("Datetime", "")

    # strings of existing file paths are registered as Path
    files = []
    for i in range(20):
        f = tmp_path / "file_{}.txt".format(i)
        f.write_text("x")
        files.append(str(f))
    paths = pd.Series(files, name="p")
    assert MetaData._get_fulltype(paths) == ("Path", "")
    meta = MetaData(pd.DataFrame({"p": paths, "x": np.arange(20) * 0.5}))
    assert meta.metadata == {("Path", ""): ["p"], ("Float", ""): ["x"]}
    assert meta.details["p"] == {}
    # non-existing paths are ordinary strings
    missing = pd.Series([str(tmp_path / "nope.txt")] * 20, name="p")
    assert MetaData._get_fulltype(missing) == ("Object", "Categorical")

    # first value cannot be checked as a path (TypeError swallowed)
    mixed = pd.Series([None, "a", "b"] * 20, dtype=object, name="m")
    assert MetaData._get_fulltype(mixed) == ("Object", "Categorical")

    cat = pd.Series(pd.Categorical(["a", "b"] * 20), name="c")
    assert MetaData._get_fulltype(cat) == ("Object", "Categorical")

    with pytest.raises(TypeError, match="not supported"):
        MetaData._get_fulltype(pd.Series([True, False], name="b"))


def test_metadata_from_others():
    arr = np.column_stack([np.arange(50), np.random.default_rng(0).normal(size=50)])
    meta = MetaData()
    result = meta.get(arr)
    # numpy array converted to dataframe with generated column names
    # (float array: the integer-valued column is detected as Int)
    assert result == {("Int", "Numerical"): ["col_0"], ("Float", ""): ["col_1"]}
    assert meta.details["col_0"]["max"] == 49

    with pytest.raises(TypeError, match="cannot be converted"):
        MetaData().get(5)


def test_check_fulltype_and_get_key_errors():
    with pytest.raises(TypeError, match="not supported"):
        MetaData._check_fulltype({("Bad", "Type"): ["a"]}, ref=UNIQUE_FULLTYPE)

    with pytest.raises(ValueError, match="Expected 1 key, got 0"):
        MetaData._get_key("missing", {("Int", "Numerical"): ["a"]})


def test_metadata_update_and_force_update():
    df = _frame()
    meta = MetaData(df)

    # change obj_cat into a numerical column and update metadata
    new = df.copy()
    new["obj_cat"] = np.arange(len(new)) * 0.5
    meta.update(new, names=["obj_cat"])
    assert ("Object", "Categorical") not in meta.metadata
    assert meta.metadata[("Float", "")] == ["float", "obj_cat"]
    assert meta.details["obj_cat"]["max"] == pytest.approx(new["obj_cat"].max())

    # update all columns without names
    meta.update(new)
    assert sorted(sum(meta.metadata.values(), [])) == sorted(new.columns)

    # force_update with a string name
    meta.force_update("int_num", [("Int", "Categorical")])
    assert "int_num" in meta.metadata[("Int", "Categorical")]
    assert ("Int", "Numerical") not in meta.metadata
    assert meta.details["int_num"] == {}

    # force_update with a single fulltype tuple (wrapped like a string)
    meta.force_update("float", ("Int", "Numerical"))
    assert meta.metadata[("Int", "Numerical")] == ["float"]
    assert meta.metadata[("Int", "Categorical")] == ["int_cat", "int_num"]
    assert ("Float", "") in meta.metadata  # obj_cat still Float
    meta.force_update(["float"], ("Float", ""))
    assert meta.metadata[("Float", "")] == ["obj_cat", "float"]

    # force_update with list inputs
    meta.force_update(["int_num"], [("Int", "Numerical")])
    assert meta.metadata[("Int", "Numerical")] == ["int_num"]

    with pytest.raises(ValueError, match="Length of names and fulltypes"):
        meta.force_update(["int_num", "float"], [("Int", "Numerical")])


def test_metadata_merge():
    df1, df2 = _frame(seed=0), _frame(seed=1)
    meta = MetaData(df1)
    meta.merge(df2)

    full = pd.concat([df1, df2], ignore_index=True)
    # metadata keys unchanged, no duplicated registrations
    assert meta.metadata == MetaData(df1).metadata
    assert meta.details["float"]["length"] == len(full)
    assert meta.details["float"]["mean"] == pytest.approx(full["float"].mean())
    assert meta.details["float"]["std"] == pytest.approx(np.std(full["float"]))
    assert meta.details["int_num"]["min"] == full["int_num"].min()
    assert meta.details["int_num"]["max"] == full["int_num"].max()
    assert meta.details["obj_cat"]["unique_count"] == dict(
        full["obj_cat"].value_counts()
    )

    # type mismatch raises
    bad = df2.copy()
    bad["float"] = ["x y z {}".format(i) for i in range(len(bad))]
    with pytest.raises(TypeError, match="Type/subtype mismatch"):
        meta.merge(bad, names=["float"])


def test_chunk_metadata(tmp_path):
    df = pd.concat(
        [_frame(seed=0), _frame(seed=1), _frame(n=50, seed=2)], ignore_index=True
    )
    # make the int_num column numerical in every chunk
    df["int_num"] = np.arange(len(df))
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    meta = ChunkMetaData(str(path), chunksize=200)
    ref = MetaData(df)

    assert meta.metadata == ref.metadata
    for col in ["float", "int_num"]:
        assert meta.details[col]["length"] == len(df)
        assert meta.details[col]["mean"] == pytest.approx(ref.details[col]["mean"])
        assert meta.details[col]["std"] == pytest.approx(ref.details[col]["std"])
        assert meta.details[col]["min"] == ref.details[col]["min"]
        assert meta.details[col]["max"] == ref.details[col]["max"]
    assert (
        meta.details["int_cat"]["unique_count"]
        == ref.details["int_cat"]["unique_count"]
    )
