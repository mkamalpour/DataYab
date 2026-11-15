"""Regression coverage for user-facing actions and application diagnostics."""
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from backend.core import DataYabCore
from backend.logging_utils import get_logger, recent_log


def make_core(problem="classification"):
    rng = np.random.default_rng(18)
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({"x": rng.normal(size=100), "noise": rng.normal(size=100),
        "empty": [None] * 100, "target": [0, 1] * 50 if problem == "classification" else rng.normal(size=100)}))
    core.split_data("target", problem_type=problem)
    core.train("Decision Tree")
    return core


@pytest.mark.parametrize("problem", ["classification", "regression"])
def test_deployment_accepts_valid_components(problem):
    core = make_core(problem)
    result = core.validate_before_deployment()
    assert result["prediction_test"] == "PASS"
    assert not any("required component" in item for item in result["errors"])
    assert result["summary"].startswith(str(len(result["errors"])))


def test_correlation_and_outlier_reports_handle_missing_and_categorical_columns():
    core = make_core()
    assert not core.get_top_correlations().empty
    core._set_dataset(pd.DataFrame({"a": [None] * 10, "b": ["x"] * 10}))
    assert core.get_top_correlations().empty
    assert core.detect_all_outliers().empty


def test_logs_capture_operation_tracebacks_without_recording_prediction_inputs():
    core = make_core()
    logger = get_logger()
    count = len(logger.handlers)
    assert get_logger() is logger and len(logger.handlers) == count
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
    capture = Capture()
    logger.addHandler(capture)
    try:
        with pytest.raises(ValueError):
            core.train("invalid-model")
        core.predict({"x": 987654321., "noise": 0., "empty": "private-category-value"})
    finally:
        logger.removeHandler(capture)
    assert any(record.exc_info and "train failed" in record.getMessage() for record in records)
    assert all("private-category-value" not in record.getMessage() for record in records)
    assert isinstance(recent_log(128), str)


def test_cli_error_is_reported_without_unbound_sys_exception():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "app.py"), "--cli", "--file", "missing.csv"],
                            capture_output=True, text=True, cwd=root)
    assert result.returncode == 1
    assert "DataYab:" in result.stderr and "UnboundLocalError" not in result.stderr


@pytest.mark.parametrize("label", ["Show Statistics", "Show Top Correlations", "Show Health Score",
    "Generate Insights", "Generate Model Card", "Calculate Permutation Importance", "Run Cross-Validation",
    "Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance", "Predict",
    "Validate for Deployment", "Data Quality Report", "Show Feature Scores", "Partial Dependence"])
def test_workbench_actions_complete(label):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit/app.py"))
    app.session_state["core"] = make_core()
    app.run(timeout=30)
    button = next(button for button in app.button if button.label.endswith(label))
    button.click().run(timeout=30)
    assert not app.exception
    assert not app.error, [(item.value) for item in app.error]


def test_invalid_action_is_logged_and_other_tabs_stay_available():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit/app.py"))
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]}))
    app.session_state["core"] = core
    app.run(timeout=30)
    next(button for button in app.button if button.label.endswith("Split & Train")).click().run(timeout=30)
    assert not app.exception and app.error
    assert len(app.tabs) >= 12


def test_imported_model_card_does_not_require_training_data(tmp_path):
    core = make_core()
    path = tmp_path / "model.joblib"
    core.save_pipeline(path)
    imported = DataYabCore()
    imported.load_pipeline(path)
    card = imported.generate_model_card()
    assert card["cross_validation"] is None
    assert any("Imported" in warning for warning in card["warnings"])
