import numpy as np
import pandas as pd
import pytest
from backend.core import DataYabCore


def test_regression_baseline_is_training_only_and_visible_in_metrics():
    core = DataYabCore()
    x = np.arange(120, dtype=float)
    core._set_dataset(pd.DataFrame({'x': x, 'target': 4 * x + 10}))
    core.split_data('target', problem_type='regression')
    core.train('Linear Regression')
    report = core.get_metrics()
    expected = np.sqrt(np.mean((core.y_test - np.mean(core.y_train)) ** 2))
    assert report['baseline_rmse'] == pytest.approx(expected)
    assert report['rmse_skill'] > .99


def test_classification_reports_majority_baseline_and_macro_metrics():
    core = DataYabCore()
    x = np.arange(120, dtype=float)
    core._set_dataset(pd.DataFrame({'x': x, 'target': ['a'] * 90 + ['b'] * 30}))
    core.split_data('target', problem_type='classification')
    core.train('Logistic Regression')
    report = core.get_metrics()
    assert report['majority_baseline_accuracy'] == .75
    assert report['balanced_accuracy'] > .8 and report['macro_f1'] > .8
