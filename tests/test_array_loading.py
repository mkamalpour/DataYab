"""Loading coverage for .arr array files."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from backend.core import DataYabCore


def save_numpy(path, array):
    with open(path, "wb") as stream:
        np.save(stream, array)


@pytest.fixture
def frame():
    return pd.DataFrame({"amount": [1.0, 2.0, 3.0], "group": ["a", "b", "c"]})


def test_numpy_matrix_loads_with_generated_columns(tmp_path, frame):
    path = tmp_path / "matrix.arr"
    save_numpy(path, frame[["amount"]].to_numpy())
    core = DataYabCore()
    result = core.load_arr(path)
    assert result["success"]
    assert list(core.df.columns) == ["column_1"]
    assert core.df["column_1"].tolist() == [1.0, 2.0, 3.0]


def test_numpy_structured_array_keeps_field_names(tmp_path):
    path = tmp_path / "records.arr"
    structured = np.array(
        [(1.0, "a"), (2.0, "b")], dtype=[("amount", "f8"), ("group", "U1")]
    )
    save_numpy(path, structured)
    core = DataYabCore()
    assert core.load_arr(path)["success"]
    assert list(core.df.columns) == ["amount", "group"]


def test_arrow_and_feather_files_load(tmp_path, frame):
    table = pa.table(frame)
    arrow_path = tmp_path / "table.arr"
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    feather_path = tmp_path / "feather.arr"
    frame.to_feather(feather_path)

    for path in (arrow_path, feather_path):
        core = DataYabCore()
        result = core.load_arr(path)
        assert result["success"], result["message"]
        assert core.df["amount"].tolist() == [1.0, 2.0, 3.0]


def test_delimited_text_arr_loads_like_csv(tmp_path):
    path = tmp_path / "text.arr"
    path.write_text("amount;group\n1;a\n2;b\n3;c\n", encoding="utf-8")
    core = DataYabCore()
    assert core.load_arr(path)["success"]
    assert list(core.df.columns) == ["amount", "group"]


def test_load_file_dispatches_arr(tmp_path):
    path = tmp_path / "sample.arr"
    save_numpy(path, np.array([5.0, 6.0]))
    core = DataYabCore()
    result = core.load_file(path)
    assert result["success"]
    assert core.df["value"].tolist() == [5.0, 6.0]


def test_invalid_arr_fails_without_replacing_dataset(tmp_path):
    core = DataYabCore()
    core.data_loader._set_dataset(pd.DataFrame({"keep": [1, 2]}))
    path = tmp_path / "broken.arr"
    path.write_bytes(b"not an array")
    result = core.load_arr(path)
    assert not result["success"]
    assert core.df["keep"].tolist() == [1, 2]


ARFF = """@relation demo

@attribute amount numeric
@attribute group {a,b}
@attribute label {-1,1}

@data
1,a,-1
2,b,1
?,a,-1
"""


def test_arff_loads_numeric_and_nominal(tmp_path):
    path = tmp_path / "table.arff"
    path.write_text(ARFF, encoding="utf-8")
    core = DataYabCore()
    result = core.load_arff(path)
    assert result["success"], result["message"]
    assert list(core.df.columns) == ["amount", "group", "label"]
    assert core.df["amount"].dtype.kind == "f"
    assert core.df["amount"].isna().sum() == 1
    assert core.df["group"].tolist() == ["a", "b", "a"]
    assert core.df["label"].tolist() == [-1, 1, -1]


def test_load_file_dispatches_arff(tmp_path):
    path = tmp_path / "sample.arff"
    path.write_text(ARFF, encoding="utf-8")
    assert DataYabCore().load_file(path)["success"]
