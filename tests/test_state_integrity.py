import numpy as np
import pandas as pd
import pytest
from backend.core import DataYabCore


@pytest.fixture
def trained():
    core=DataYabCore()
    core._set_dataset(pd.DataFrame({'x':np.arange(40),'group':['a','b']*20,'target':[0,1]*20}))
    core.split_data('target');core.train('Decision Tree')
    return core


def test_failed_candidate_preserves_working_model_and_predictions(trained):
    previous=trained.model
    prediction=trained.predict({'x':3,'group':'a'})
    with pytest.raises(ValueError):trained.train('KNN',n_neighbors=1000)
    assert trained.model is previous and trained.model_name=='Decision Tree'
    assert trained.predict({'x':3,'group':'a'})==prediction
    assert trained.get_trained_models()==['Decision Tree']


def test_failed_cleaning_restores_encoder_metadata_and_log(trained):
    original=trained.df.copy(deep=True);log=trained.get_log();model=trained.model
    with pytest.raises(KeyError):trained.encode_categorical(['group','missing_column'])
    pd.testing.assert_frame_equal(trained.df,original)
    assert trained.model is model and trained.label_encoders=={}
    assert trained.get_log()==log
    result=trained.encode_categorical(method='unsupported')
    assert not result['success'] and trained.model is model


def test_nested_array_load_is_rejected_before_resetting_old_model(trained):
    old_model=trained.model;old_data=trained.df.copy(deep=True)
    with pytest.raises(ValueError,match='Nested'):
        trained._set_dataset(pd.DataFrame({'nested':[np.array([1,2]),np.array([3,4])]}))
    assert trained.model is old_model
    pd.testing.assert_frame_equal(trained.df,old_data)


def test_observations_handle_entirely_missing_numeric_and_target_columns():
    core=DataYabCore();core._set_dataset(pd.DataFrame({'number':[np.nan]*12,'target':[None]*12}))
    core.target_col='target'
    result=core.generate_observations()
    assert result and any('missing' in item.lower() for item in result)


def test_drop_missing_columns_respects_selection_and_protects_default_target():
    core = DataYabCore()
    original = pd.DataFrame({'keep': [1., np.nan, np.nan, np.nan],
                             'drop': [2., np.nan, np.nan, np.nan],
                             'target': [0., np.nan, np.nan, np.nan]})
    core._set_dataset(original)
    core.target_col = 'target'
    core.handle_missing('drop_cols', columns=['drop'])
    pd.testing.assert_frame_equal(core.df, original.drop(columns=['drop']))
    core.handle_missing('drop_cols')
    pd.testing.assert_frame_equal(core.df, original[['target']])
    assert core.target_col == 'target'


def test_iterative_missing_strategy_is_usable_without_external_import_setup():
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({'a': [1., 2., np.nan, 4.], 'b': [2., 4., 6., 8.]}))
    result = core.handle_missing('iterative')
    assert result['success'] and not core.df.isna().any().any()
    assert core.df.loc[2, 'a'] == pytest.approx(3., abs=.01)



def test_insights_after_cleaning_infers_categorical_target_without_a_training_split():
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({'x': [1, 2, 3, 4], 'target': ['a', 'b', 'a', 'b']}))
    core.auto_clean(target_col='target')
    assert core.problem_type == ''
    insights = core.generate_insights()
    assert any('CLASSIFICATION' in line for line in insights)
    assert core._raw_train is None


def test_failed_selected_drop_preserves_entire_trained_state(trained):
    original = trained.df.copy(deep=True)
    model = trained.model
    signature = trained._split_signature
    raw_train = trained._raw_train
    logs = trained.get_log()
    prediction = trained.predict({'x': 3, 'group': 'a'})
    with pytest.raises(KeyError):
        trained.handle_missing('drop_cols', ['group', 'missing_column'])
    pd.testing.assert_frame_equal(trained.df, original)
    assert trained.model is model and trained._raw_train is raw_train
    assert trained._split_signature == signature
    assert trained.get_log() == logs and trained.target_col == 'target'
    assert trained.predict({'x': 3, 'group': 'a'}) == prediction



def test_desktop_blank_numeric_prediction_matches_backend_imputation():
    from types import SimpleNamespace
    from frontend.desktop import PredictTab
    core = DataYabCore()
    core._set_dataset(pd.DataFrame({'x': np.arange(40), 'target': 100 + np.arange(40)}))
    core.split_data('target', problem_type='regression')
    core.train('Linear Regression')
    expected = core.predict({'x': ''})
    assert expected['prediction'] != pytest.approx(core.predict({'x': '0'})['prediction'])
    results = []
    tab = SimpleNamespace(app=SimpleNamespace(core=core, run_threaded=lambda work, done: done(work())),
                          input_entries={'x': SimpleNamespace(get=lambda: '')},
                          display_prediction=results.append)
    PredictTab.predict_manual(tab)
    assert results == [expected]
