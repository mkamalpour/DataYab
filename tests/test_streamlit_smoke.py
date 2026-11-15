"""Repeatable rendering checks for the local Streamlit workbench."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from backend.core import DataYabCore

APP = Path(__file__).resolve().parents[1] / "frontend" / "streamlit" / "app.py"


@pytest.mark.parametrize("state", ["empty", "numeric", "categorical", "trained"])
def test_workbench_states_render(state):
    app = AppTest.from_file(str(APP))
    if state != "empty":
        core = DataYabCore()
        frame = pd.DataFrame({"x": np.arange(40), "target": [0, 1] * 20})
        if state == "categorical":
            frame = pd.DataFrame({"group": ["a", "b"] * 20, "label": ["x", "y"] * 20})
        core._set_dataset(frame)
        if state == "trained":
            core.split_data("target")
            core.train("Decision Tree")
        app.session_state["core"] = core
        app.session_state["data_loaded"] = True
        app.session_state["split_done"] = state == "trained"
        app.session_state["model_trained"] = state == "trained"
    app.run(timeout=20)
    assert not app.exception
    assert len(app.tabs) >= 12


def test_advanced_buttons_compute_and_persist_results():
    pytest.importorskip("statsmodels")
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=60),
        "x": np.sin(np.arange(60) * 2 * np.pi / 7) + np.arange(60) / 10,
        "y": np.cos(np.arange(60)),
    }))
    app = AppTest.from_file(str(APP))
    app.session_state["core"] = core
    app.session_state["data_loaded"] = True
    app.run(timeout=20)
    for label, key in [("Run clustering", "cluster_result"),
                       ("Detect anomalies", "anomaly_result"),
                       ("Run time series analysis", "ts_result")]:
        next(button for button in app.button if button.label == label).click().run(timeout=30)
        assert not app.exception
        assert not app.error
        assert key in app.session_state
