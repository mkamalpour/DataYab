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
