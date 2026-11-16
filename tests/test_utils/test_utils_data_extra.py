"""
File Name: test_utils_data_extra.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_utils/test_utils_data_extra.py
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

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from InsurAutoML.utils import data as udata


def test_str2list_str2dict():
    assert udata.str2list("[1, 2, 'a']") == [1, 2, "a"]
    # a single name is a one-element list (not iterated character by character)
    assert udata.str2list("RandomForest") == ["RandomForest"]
    assert udata.str2list("not a list(") == ["not a list("]
    assert udata.str2list(["A", "B"]) == ["A", "B"]
    assert udata.str2list(("A",)) == ["A"]
    assert udata.str2dict('{"a": 1, "b": [1, 2]}') == {"a": 1, "b": [1, 2]}
    assert udata.str2dict({"a": 1}) == {"a": 1}
    with pytest.raises(ValueError):
        udata.str2dict("{bad json}")
    with pytest.raises(ValueError):
        udata.str2dict("[1, 2]")


def test_train_test_split():
    X = pd.DataFrame({"a": np.arange(20)})
    y = pd.Series(np.arange(20) * 10)
    X_train, X_test, y_train, y_test = udata.train_test_split(X, y, test_perc=0.25, seed=1)

    assert len(X_train) == 15 and len(X_test) == 5
    assert set(X_train["a"]).isdisjoint(set(X_test["a"]))
    assert set(X_train["a"]) | set(X_test["a"]) == set(range(20))
    np.testing.assert_array_equal(y_test.values, X_test["a"].values * 10)
    # deterministic given the seed
    again = udata.train_test_split(X, y, test_perc=0.25, seed=1)
    pd.testing.assert_frame_equal(again[1], X_test)


def test_as_dataframe():
    conv = udata.as_dataframe()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    arr = conv.to_array(df)
    np.testing.assert_array_equal(arr, [[1, 3], [2, 4]])

    # default: restore the stored array/columns
    pd.testing.assert_frame_equal(conv.to_df(), df)
    out = conv.to_df(arr * 2, columns=np.array(["x", "y"]))
    assert list(out.columns) == ["x", "y"]
    assert out["y"].tolist() == [6, 8]

    with pytest.raises(TypeError, match="Input should be dataframe"):
        conv.to_array(arr)
    with pytest.raises(TypeError, match="Input should be numpy array"):
        conv.to_df([[1, 2]])
    with pytest.raises(ValueError, match="does not match"):
        conv.to_df(arr, columns=["only_one"])


def test_formatting_roundtrip():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, np.nan],
            "cat": ["John", np.nan, "Amy", "John"],
        }
    )
    fmt = udata.formatting()
    fmt.fit(df)
    assert fmt.unique_table["cat"] == ["John", "Amy"]
    assert df["cat"].tolist()[0] == 0 and df["cat"].tolist()[2] == 1
    assert pd.isna(df["cat"].iloc[1])

    # unseen numeric value mapped to closest seen class
    df.loc[2, "cat"] = 0.9
    fmt.refit(df)
    assert df["cat"].tolist()[0] == "John" and df["cat"].tolist()[2] == "Amy"
    assert df["num"].dtype == np.float64

    # not inplace: data untouched by fit, refit returns new values
    raw = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]})
    fmt2 = udata.formatting(columns=["cat", "num"], inplace=False)
    fmt2.fit(raw)
    assert raw["cat"].tolist() == ["a", "b", "a"]
    coded = pd.DataFrame({"cat": [1, 0, 1], "num": [1.0, 2.0, 3.0]})
    back = fmt2.refit(coded)
    assert back["cat"].tolist() == ["b", "a", "b"]
    assert back["num"].dtype == raw["num"].dtype

    # numpy input converted to dataframe
    fmt3 = udata.formatting()
    fmt3.fit(np.array([[1, 2], [3, 4]]))
    assert fmt3.columns == [0, 1]
    with pytest.raises(TypeError, match="Expect a dataframe"):
        udata.formatting().fit(5)


def test_unify_nan():
    data = pd.DataFrame(
        {
            "a": ["novalue", "x", "y"],
            "b": ["1", "None", "3"],
            "c": [1, 2, 3],
        }
    )
    out = udata.unify_nan(data.copy())
    assert "a_useNA" in out and "b_useNA" in out and "c_useNA" not in out
    assert out["a_useNA"].isna().tolist() == [True, False, False]

    out = udata.unify_nan(data.copy(), columns="b", replace=True)
    assert out["b"].isna().tolist() == [False, True, False]
    assert list(out.columns) == ["a", "b", "c"]


def test_remove_index_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = pd.DataFrame(
        {
            "a": [1, np.nan, np.nan, np.nan],
            "b": [1, 2, np.nan, 4],
            "c": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    # numpy input, all columns, default threshold 1
    out = udata.remove_index_columns(base.values.copy())
    assert out.shape == (4, 2)

    # columns given via index (axis=1), threshold list, save removed columns
    out = udata.remove_index_columns(
        base.copy(), index=["a", "b"], threshold=[0.5, 0.5], save=True
    )
    assert list(out.columns) == ["b", "c"]
    saved = pd.read_csv(tmp_path / "Removing_data(Limited_Information).csv")
    assert list(saved.columns) == ["a"]

    with pytest.raises(ValueError, match="Columns and threshold"):
        udata.remove_index_columns(base.copy(), threshold=[0.5])

    # rows, scalar threshold
    out = udata.remove_index_columns(base.copy(), axis=0, threshold=0.9, reset_index=False)
    assert list(out.index) == [0, 1, 3]

    # rows given via columns (axis=0), threshold list, save removed rows
    out = udata.remove_index_columns(
        base.copy(), columns=[1, 2], axis=0, threshold=[0.5, 1.0], save=True
    )
    assert len(out) == 2
    assert list(out.index) == [0, 1]
    saved = pd.read_csv(tmp_path / "Removing_data(Limited_Information).csv")
    assert len(saved) == 2

    with pytest.raises(ValueError, match="Indexes and threshold"):
        udata.remove_index_columns(base.copy(), axis=0, threshold=[0.5])

    with pytest.raises(TypeError, match="Expect a dataframe"):
        udata.remove_index_columns(5)


def test_get_missing_matrix_type_error():
    with pytest.raises(TypeError, match="Expect a array"):
        udata.get_missing_matrix([[1, 2]])
    np.testing.assert_array_equal(
        udata.get_missing_matrix(np.array([[1, np.nan]]), missing=0), [[1, 0]]
    )
    df = pd.DataFrame({"a": [1, None], "b": ["novalue", "x"]})
    np.testing.assert_array_equal(udata.get_missing_matrix(df), [[0, 1], [1, 0]])


def test_assign_classes_probabilities():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3], [0.0, 0.1, 0.9]])
    np.testing.assert_array_equal(udata.assign_classes(probs), [1, 0, 2])


def test_is_imbalance():
    balanced = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 2, 3, 4]})
    imbalanced = pd.DataFrame({"a": [0, 0, 0, 1], "b": [1, 2, 3, 4]})
    single = pd.DataFrame({"a": [1, 1, 1, 1]})

    assert udata.is_imbalance(balanced, 0.6) is False
    assert udata.is_imbalance(balanced, 0.6, value=True) == (None, None)
    assert udata.is_imbalance(imbalanced, 0.6) is True
    assert udata.is_imbalance(imbalanced, 0.6, value=True) == ("a", 0)
    assert udata.is_imbalance(single, 0.6) is False
    assert udata.is_imbalance(single, 0.6, value=True) == (None, None)


def test_link_table():
    table = pd.DataFrame({"x": [0, 3], "y": [0, 4]})
    sample = pd.DataFrame({"x": [0], "y": [0]})
    assert udata.LinkTable(sample, table, norm="l2") == [[0.0, 25.0]]
    assert udata.LinkTable(sample, table, norm="l1") == [[0.0, 7.0]]
    with pytest.raises(ValueError, match="Not same size"):
        udata.LinkTable(sample[["x"]], table)


def test_extreme_class():
    X = pd.DataFrame(
        {
            "const": [1] * 10,
            "mostly": [0] * 9 + [1],
            "mixed": [0, 1] * 5,
        }
    )
    assert list(udata.ExtremeClass().cut(X).columns) == ["mostly", "mixed"]
    assert list(udata.ExtremeClass(extreme_threshold=0.9).cut(X).columns) == ["mixed"]
    # input not modified
    assert list(X.columns) == ["const", "mostly", "mixed"]


def test_softmax():
    out = udata.softmax(np.array([0.0, 100.0, -100.0]))
    np.testing.assert_allclose(out, [[0.5, 0.5], [0, 1], [1, 0]])

    out = udata.softmax(np.array([[1.0, 1.0], [0.0, np.log(3)]]))
    np.testing.assert_allclose(out, [[0.5, 0.5], [0.25, 0.75]])


def test_plot_high_dim_cluster(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = np.repeat([0, 1], 20)

    path = tmp_path / "plots"
    udata.plotHighDimCluster(X, y, plot=True, method="PCA", dim=2, save=True, path=str(path))
    assert (path / "HighDim.png").exists()

    udata.plotHighDimCluster(X, y, plot=False, method="PCA", dim=3)
    udata.plotHighDimCluster(X, y, plot=False, method="TSNE", dim=2)

    X_big = rng.normal(size=(110, 101))
    udata.plotHighDimCluster(X_big, np.arange(110) % 2, plot=False, method="TSNE", dim=2)

    with pytest.raises(ValueError, match="high dimensional"):
        udata.plotHighDimCluster(X[:, :3], y)
    with pytest.raises(ValueError, match="same length"):
        udata.plotHighDimCluster(X, y[:10])


def test_feature_type():
    assert udata.feature_type(pd.Series([0, 1] * 50)) == "numerical_categorical"
    assert udata.feature_type(pd.Series(np.arange(100) * 1.5)) == "continuous"
    assert udata.feature_type(pd.Series(["a", "b"] * 50)) == "string_categorical"
    assert udata.feature_type(pd.Series(["w {}".format(i) for i in range(100)])) == "text"


def test_text_preprocessing():
    assert udata.text_preprocessing("Hello (remove me) World! [x] 42-Times") == [
        "hello",
        "world",
        "42",
        "times",
    ]


@pytest.fixture
def corpus():
    return pd.Series(
        ["the cat sat on the mat", "the dog ate the cat", "a mat (old) for the dog"] * 4,
        name="review",
    )


def test_text2vec_word2vec(corpus):
    pytest.importorskip("gensim")

    out = udata.text2vec(corpus, dim=8, how="mean")
    assert out.shape == (len(corpus), 8)
    assert list(out.columns) == ["review_{}".format(i) for i in range(8)]
    # identical sentences get identical vectors
    np.testing.assert_allclose(out.iloc[0].values, out.iloc[3].values)

    torch = pytest.importorskip("torch")
    pt = udata.text2vec(corpus, dim=4, return_type="pt")
    assert isinstance(pt, torch.Tensor) and tuple(pt.shape) == (len(corpus), 4)


def test_text2vec_pretrained(corpus, monkeypatch):
    gensim = pytest.importorskip("gensim")
    import gensim.downloader
    from gensim.models import KeyedVectors

    with pytest.raises(ValueError, match="Pretrained model must be one of"):
        udata.text2vec(corpus, pretrained="not-a-model")

    # avoid downloading: serve small fixed keyed vectors
    words = sorted({w for s in corpus for w in udata.text_preprocessing(s)})
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(words, np.arange(len(words) * 3, dtype=np.float32).reshape(-1, 3))
    requested = []

    def fake_load(name):
        requested.append(name)
        return kv

    monkeypatch.setattr(gensim.downloader, "load", fake_load)

    out = udata.text2vec(corpus, pretrained="glove-twitter-25", dim=3, how="sum")
    assert requested == ["glove-twitter-25"]
    expected = sum(kv[w] for w in udata.text_preprocessing(corpus.iloc[0]))
    np.testing.assert_allclose(out.iloc[0].values, expected)


def test_text2vec_dataframe_input(corpus, monkeypatch):
    pytest.importorskip("gensim")
    import gensim.downloader
    from gensim.models import KeyedVectors

    # fixed pretrained vectors make the output deterministic
    other = pd.Series(["dog dog", "cat mat", "the a"] * 4, name="title")
    words = sorted(
        {w for s in pd.concat([corpus, other]) for w in udata.text_preprocessing(s)}
    )
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(words, np.arange(len(words) * 3, dtype=np.float32).reshape(-1, 3))
    monkeypatch.setattr(gensim.downloader, "load", lambda name: kv)
    kwargs = dict(pretrained="glove-twitter-25", dim=3, how="sum")

    # single text column dataframe behaves like its Series
    from_series = udata.text2vec(corpus, **kwargs)
    from_frame = udata.text2vec(corpus.to_frame(), **kwargs)
    pd.testing.assert_frame_equal(from_frame, from_series)
    assert list(from_frame.columns) == ["review_0", "review_1", "review_2"]

    # multiple text columns: vectorized per column (element-wise) and concatenated
    df = pd.DataFrame({"review": corpus, "title": other})
    out = udata.text2vec(df, **kwargs)
    assert out.shape == (len(df), 6)
    assert list(out.columns) == ["review_0", "review_1", "review_2",
                                 "title_0", "title_1", "title_2"]
    pd.testing.assert_frame_equal(out[["review_0", "review_1", "review_2"]], from_series)
    np.testing.assert_allclose(out.iloc[0, 3:].values, kv["dog"] * 2)

    torch = pytest.importorskip("torch")
    pt = udata.text2vec(df, return_type="pt", **kwargs)
    assert isinstance(pt, torch.Tensor) and tuple(pt.shape) == (len(df), 6)
    np.testing.assert_allclose(pt.numpy(), out.values, rtol=1e-6)
    pt1 = udata.text2vec(corpus.to_frame(), return_type="pt", **kwargs)
    np.testing.assert_allclose(pt1.numpy(), from_series.values, rtol=1e-6)
