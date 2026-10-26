"""Profile CLI regressions using tiny disposable tables."""
import json
from tools.profile_data import ultimate_profile


def test_boolean_columns_are_profiled_without_numeric_quartile_arithmetic(tmp_path):
    path = tmp_path / 'table.csv'
    path.write_text('flag,value\nTrue,1\nFalse,2\nTrue,3\n', encoding='utf-8')
    profile = ultimate_profile(path)
    assert profile['columns']['flag']['basic']['dtype'] == 'bool'
    assert profile['columns']['flag']['top_values'] == {'True': 2, 'False': 1}
    json.dumps(profile, allow_nan=False)


def test_profile_undefined_statistics_become_json_null(tmp_path):
    path = tmp_path / 'table.csv'
    path.write_text('name,value\nsingle,1\n', encoding='utf-8')
    profile = ultimate_profile(path)
    assert profile['columns']['value']['numeric']['std'] is None
    assert profile['columns']['value']['numeric']['skewness'] is None
    assert json.loads(json.dumps(profile, allow_nan=False)) == profile
