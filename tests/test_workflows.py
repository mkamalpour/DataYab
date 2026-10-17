"""Regression coverage using synthetic data only."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.core import DataYabCore
from frontend import web


@pytest.fixture
def core():
    c = DataYabCore()
    x = np.arange(120, dtype=float)
    c._set_dataset(
        pd.DataFrame(
            {"amount": x, "group": ["a", "b", "c"] * 40, "target": ["no", "yes"] * 60}
        )
    )
    return c


def test_split_imputation_uses_training_rows(core):
    core.df.loc[::7, "amount"] = np.nan
    core.df.loc[0, "target"] = None
    core.split_data("target")
    core.train("Logistic Regression")
    prep = core.model.named_steps["prepare"]
    first = prep.transformers_[0][1]
    assert (
        first.named_steps["impute"].statistics_[0] == core._raw_train["amount"].median()
    )
    assert len(core.y_train) + len(core.y_test) == 119
    assert set(core._raw_train.index).isdisjoint(core._raw_test.index)
    assert core.predict({"amount": None, "group": "unseen"})["prediction"] in [
        "no",
        "yes",
    ]


@pytest.mark.parametrize("transform", ["none", "pca", "selection"])
def test_model_roundtrip_without_dataset(core, tmp_path, transform):
    core.split_data("target")
    if transform == "pca":
        core.apply_pca(1)
    elif transform == "selection":
        core.select_features_manual(["amount"])
    core.train("Random Forest", n_estimators=10)
    inputs = {"amount": 12, "group": "a"}
    expected = core.predict(inputs)
    path = tmp_path / "model.joblib"
    core.save_pipeline(path)
    loaded = DataYabCore()
    loaded.load_pipeline(path)
    assert loaded.df is None
    assert loaded.predict(inputs) == expected
    assert loaded.get_input_form_schema() == core.get_input_form_schema()


def test_clean_invalidates_models_but_keeps_target(core):
    core.split_data("target")
    core.train("Decision Tree")
    core.auto_clean(target_col="target")
    assert core.target_col == "target"
    assert core.model is None and core.X_train is None


def test_failed_load_preserves_model(core, tmp_path):
    core.split_data("target")
    core.train("Decision Tree")
    previous = core.model
    path = tmp_path / "nested.json"
    path.write_text('[{"x": [1, 2]}, {"x": [3]}]')
    assert not core.load_json(path)["success"]
    assert core.model is previous


def test_regression_cv_and_stacking(core):
    core.df["target"] = core.df["amount"] * 2.5 + 3
    core.split_data("target", problem_type="regression")
    core.train("Linear Regression")
    assert core.get_metrics()["r2"] > 0.99
    assert core.get_cross_val_score(3)["mean"] > 0.99
    core.train("Ridge")
    result = core.train_stacked_model("Linear Regression", "Ridge", "Ridge")
    assert result["success"]
    assert core.get_metrics()["r2"] > 0.99


def test_smote_requires_numeric_features(core):
    core.split_data("target")
    with pytest.raises(ValueError, match="categorical|numeric"):
        core.train("Random Forest", use_smote=True)


def test_model_selection_and_reset(core):
    core.split_data("target")
    core.train("Decision Tree")
    core.train("Logistic Regression")
    core.select_model("Decision Tree")
    assert core.model_name == "Decision Tree"
    core.reset_to_original()
    assert core.model is None and not core.trained_models


def test_api_workflow_and_errors(core):
    web.core = DataYabCore()
    client = TestClient(web.app)
    assert client.get("/api/evaluate").status_code == 400
    csv = core.df.to_csv(index=False)
    assert (
        client.post(
            "/api/upload", files={"file": ("../../sample.csv", csv, "text/csv")}
        ).status_code
        == 200
    )
    assert web.core.file_path == "sample.csv"
    assert (
        client.post(
            "/api/train", json={"target": "target", "model": "Decision Tree"}
        ).status_code
        == 200
    )
    assert client.post("/api/feature/pca?components=1").status_code == 200
    assert (
        client.post(
            "/api/train", json={"target": "target", "model": "Decision Tree"}
        ).status_code
        == 200
    )
    assert (
        client.post("/api/predict", json={"amount": 12, "group": "unknown"}).status_code
        == 200
    )
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.get("/api/cross-validation").status_code == 200
    assert client.get("/api/export/report").status_code == 200
    assert client.post("/api/reset").status_code == 200
    assert client.get("/api/evaluate").status_code == 400


def test_api_rejects_foreign_origin_and_large_request():
    client = TestClient(web.app)
    assert (
        client.post("/api/reset", headers={"origin": "https://example.com"}).status_code
        == 403
    )
    assert client.get("/", headers={"host": "example.com"}).status_code == 400
    assert (
        client.post(
            "/api/upload", headers={"content-length": str(60 * 1024 * 1024)}
        ).status_code
        == 413
    )


def test_report_escapes_user_data(core, tmp_path):
    core.file_path = "<script>alert(1)</script>"
    core.df["<script>"] = "<img src=x onerror=alert(1)>"
    path = tmp_path / "report.html"
    core.export_report(path)
    report = path.read_text(encoding="utf-8")
    assert "<script>" not in report
    assert "<img src=x" not in report


def test_single_column_csv_is_not_split_on_letters(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("amount\n10\n20\n30\n")
    core = DataYabCore()
    assert core.load_csv(path)["success"]
    assert list(core.df.columns) == ["amount"]
    assert core.df["amount"].tolist() == [10, 20, 30]
