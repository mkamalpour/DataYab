"""State transitions and unusual but supported table inputs."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from backend.core import DataYabCore


def trained(problem="classification", constant=False):
    rng = np.random.default_rng(8)
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({"x": rng.normal(size=60), "group": ["a", "b"] * 30,
        "target": np.ones(60) if constant else ([0, 1] * 30 if problem == "classification" else rng.normal(size=60))}))
    core.split_data("target", problem_type=problem)
    core.train("Decision Tree")
    return core


def test_constant_regression_model_renders_without_fabricated_skill():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    core = trained("regression", constant=True)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit/app.py"))
    app.session_state["core"] = core
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert core.get_metrics()["rmse_skill"] is None


@pytest.mark.parametrize("kind", ["nullable", "all_missing", "single_column", "dates", "constant"])
def test_unusual_tables_render(kind):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    frames = {
        "nullable": pd.DataFrame({"x": pd.Series([1, None] * 10, dtype="Int64"), "flag": pd.Series([True, None] * 10, dtype="boolean")}),
        "all_missing": pd.DataFrame({"x": [np.nan] * 20, "y": [None] * 20}),
        "single_column": pd.DataFrame({"value": range(20)}),
        "dates": pd.DataFrame({"date": pd.date_range("2026-01-01", periods=20, tz="UTC"), "value": range(20)}),
        "constant": pd.DataFrame({"x": [1.] * 20, "y": [1.] * 20}),
    }
    core = DataYabCore()
    core._set_dataset(frames[kind])
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit/app.py"))
    app.session_state["core"] = core
    app.run(timeout=30)
    assert not app.exception and not app.error


def test_failed_split_keeps_working_model_and_feature_state(monkeypatch):
    core = trained()
    previous = core._raw_train
    model = core.model
    signature = core._split_signature
    def fail():
        raise ValueError("preprocessing failure")
    monkeypatch.setattr(core.feature_engineer, "_refresh_feature_view", fail)
    with pytest.raises(ValueError, match="preprocessing failure"):
        core.split_data("target", test_size=.3)
    assert core._raw_train is previous and core.model is model
    assert core._split_signature == signature


def test_invalid_import_keeps_previous_model(tmp_path):
    core = trained()
    path = tmp_path / "model.joblib"
    core.save_pipeline(path)
    package = joblib.load(path)
    package["schema"] = [None]
    joblib.dump(package, path)
    model = core.model
    with pytest.raises(ValueError):
        core.load_pipeline(path)
    assert core.model is model


def test_prediction_metadata_cannot_be_overwritten_by_input_columns():
    core = trained()
    frame = core.df.rename(columns={"x": "predicted", "group": "original_index"})
    core._set_dataset(frame)
    core.split_data("target")
    core.train("Decision Tree")
    result = core.predict_on_test_set()
    np.testing.assert_array_equal(result["predicted"].astype(str), core.target_encoder.inverse_transform(core.y_pred))


def test_no_dataset_split_raises_actionable_error():
    with pytest.raises(ValueError, match="dataset"):
        DataYabCore().split_data("target")


@pytest.mark.parametrize("problem", ["classification", "regression"])
@pytest.mark.parametrize("action", ["Split & Train", "Learning Curve", "Run Cross-Validation", "Generate Model Card", "Partial Dependence"])
def test_training_and_evaluation_actions_across_tasks(problem, action):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    core = trained(problem)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit/app.py"))
    app.session_state["core"] = core
    app.run(timeout=30)
    # Training must use the intended target rather than the first input column.
    if action == "Split & Train":
        next(widget for widget in app.selectbox if widget.label == "Target Column").select("target")
        next(widget for widget in app.selectbox if widget.label == "Problem Type").select(problem)
        app.run(timeout=30)
    next(button for button in app.button if button.label.endswith(action)).click().run(timeout=30)
    assert not app.exception and not app.error, [item.value for item in app.error]


@pytest.mark.parametrize("indices", [[-1], [10000], ["0"], [True]])
def test_invalid_prediction_positions_are_rejected(indices):
    with pytest.raises(ValueError, match="row positions"):
        trained().predict_on_test_set(indices)
