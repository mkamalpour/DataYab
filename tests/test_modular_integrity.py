"""Integration checks for modular state and honest model evaluation."""
import numpy as np
import pandas as pd
import pytest

from backend.core import DataYabCore


def trained_core():
    core = DataYabCore()
    rng = np.random.default_rng(12)
    core._set_dataset(pd.DataFrame({"x": rng.normal(size=80), "noise": rng.normal(size=80),
                                    "target": [0, 1] * 40}))
    core.split_data("target")
    core.train("Decision Tree")
    return core


def test_feature_changes_invalidate_models_and_failed_changes_restore_state():
    core = trained_core()
    model = core.model
    with pytest.raises(ValueError):
        core.apply_extraction("pca", 100)
    assert core.model is model
    core.apply_selection("manual", features=["x"])
    assert core.model is None and not core.trained_models
    assert core.X_train.shape[1] == 1


def test_replacement_resplit_and_reset_clear_stale_state():
    core = trained_core()
    core.split_data("target", test_size=.3)
    assert core.model is None
    core.train("Decision Tree")
    core.reset_to_original()
    assert core.model is None and core.X_train is None and core._split_signature is None
    assert core.target_col == ""


def test_grid_search_uses_pipeline_parameters_and_training_rows_only():
    core = trained_core()
    holdout = core._raw_test.copy(deep=True)
    result = core.tune_with_gridsearch("Decision Tree", {"max_depth": [1, 2]}, cv_folds=3)
    assert result["success"] and np.isfinite(result["best_score"])
    assert core.model.named_steps["model"].max_depth in (1, 2)
    pd.testing.assert_frame_equal(core._raw_test, holdout)
    assert core.model.named_steps["prepare"].transformers_[0][1].named_steps["impute"].statistics_[0] == pytest.approx(core._raw_train.x.median())


def test_selection_is_disclosed_in_cv_results():
    core = trained_core()
    core.apply_selection("rf", k=1)
    core.train("Decision Tree")
    assert "conditional" in core.get_cross_val_score(3)["warning"]


def test_holdout_predictions_retain_source_rows():
    core = trained_core()
    predictions = core.predict_on_test_set()
    assert len(predictions) == len(core.y_test)


def test_categorical_only_stats_and_duplicate_columns():
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({"group": ["a", None, "b"]}))
    assert core.get_stats().loc["group", "missing"] == 1
    with pytest.raises(ValueError, match="unique"):
        core._set_dataset(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    assert list(core.df) == ["group"]


def test_zscore_constant_columns_do_not_hide_outliers():
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({"x": [0.] * 39 + [100.], "constant": [1.] * 40}))
    result = core.advanced_analytics.detect_anomalies_zscore()
    assert result["success"] and result["anomaly_indices"] == [39]


@pytest.mark.parametrize("method", ["run_kmeans", "run_dbscan", "run_hierarchical", "run_gmm",
    "detect_anomalies_isolation_forest", "detect_anomalies_lof", "detect_anomalies_one_class_svm",
    "detect_anomalies_zscore", "detect_anomalies_iqr"])
def test_advanced_algorithms_return_aligned_rows_without_mutating_data(method):
    core = trained_core()
    original = core.df.copy(deep=True)
    result = getattr(core.advanced_analytics, method)(columns=["x", "noise"])
    assert result["success"]
    assert result["result_df"].index.equals(original.index)
    pd.testing.assert_frame_equal(core.df, original)


def test_time_series_decomposition_and_forecast():
    pytest.importorskip("statsmodels")
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({"date": pd.date_range("2025-01-01", periods=60),
                                    "value": np.sin(np.arange(60) * 2 * np.pi / 7) + np.arange(60) / 10}))
    analytics = core.advanced_analytics
    assert analytics.set_time_index("date")["success"]
    decomposition = analytics.time_series_decomposition("value", period=7)
    assert decomposition["success"] and len(decomposition["decomposition"]["observed"]) == 60
    forecast = analytics.forecast_arima("value", steps=3)
    assert forecast["success"] and len(forecast["forecast"]) == 3
    assert np.isfinite(forecast["confidence_intervals"]).all()
