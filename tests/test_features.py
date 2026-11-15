"""Separate feature selection and feature extraction coverage."""

import numpy as np
import pandas as pd
import pytest

from backend.core import DataYabCore


@pytest.fixture
def core():
    generator = np.random.default_rng(42)
    signal = generator.normal(size=160)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "redundant": signal * 3,
            "noise": generator.normal(size=160),
            "group": ["a", "b"] * 80,
            "constant": 1.0,
            "target": (signal > 0).astype(int),
        }
    )
    instance = DataYabCore()
    instance.data_loader._set_dataset(frame)
    instance.split_data("target")
    return instance


def test_ranking_methods_return_original_columns(core):
    for method in ("rf", "mutual_info", "anova", "correlation"):
        scores = core.get_feature_scores(method, 3)
        assert len(scores) == 3
        assert "signal" in set(scores.index) | {
            "redundant"
        }, f"{method} missed the signal column"
        assert set(scores.index) <= set(core._all_input_names)


def test_selection_methods_select_and_restore(core):
    result = core.apply_selection("anova", 2)
    assert result["total"] == 2
    assert set(result["selected"]) <= {"signal", "redundant", "noise", "group"}
    status = core.get_feature_status()
    assert status["selected"] == 2
    assert core.X_train.shape[1] == 2

    restored = core.restore_selection()
    assert restored["total"] == 5
    assert core.get_feature_status()["selected"] == 5

    rfe = core.apply_selection("rfe", 3)
    assert rfe["total"] == 3
    core.restore_selection()

    variance = core.apply_selection("variance")
    assert "constant" not in variance["selected"]
    assert variance["total"] == 4


def test_manual_selection_rejects_component_names(core):
    core.apply_extraction("pca", 2)
    with pytest.raises(ValueError, match="original input columns"):
        core.apply_selection("manual", features=["PC1"])


def test_selection_and_extraction_are_independent(core):
    core.apply_selection("manual", features=["signal", "noise"])
    core.apply_extraction("pca", 2)
    status = core.get_feature_status()
    assert status["extraction"] == "pca"
    assert core.feature_names == ["PC1", "PC2"]

    core.restore_extraction()
    status = core.get_feature_status()
    assert status["extraction"] is None
    assert status["selected"] == 2
    assert core.feature_names == ["signal", "noise"]

    core.restore_selection()
    assert core.get_feature_status()["selected"] == 5


@pytest.mark.parametrize(
    "method,components,prefix",
    [("pca", 2, "PC"), ("svd", 2, "SVD"), ("lda", 1, "LD")],
)
def test_extraction_methods_build_components(core, method, components, prefix):
    result = core.apply_extraction(method, components)
    assert result["success"]
    assert core.feature_names == [f"{prefix}{i + 1}" for i in range(components)]
    core.train("Decision Tree", max_depth=3)
    assert "extract" in core.model.named_steps
    prediction = core.predict({column: 0 for column in core._all_input_names})
    assert "prediction" in prediction


def test_extraction_validation(core):
    with pytest.raises(ValueError, match="positive whole number"):
        core.apply_extraction("pca", 0)
    with pytest.raises(ValueError, match="Extraction methods"):
        core.apply_extraction("umap", 2)
    with pytest.raises(ValueError, match="classification"):
        regression = DataYabCore()
        regression._set_dataset(
            pd.DataFrame({"x": np.arange(30.0), "y": np.arange(30.0) * 2})
        )
        regression.split_data("y", problem_type="regression")
        regression.apply_extraction("lda", 1)


def test_extraction_is_inside_cross_validation(core):
    core.apply_extraction("pca", 2)
    core.train("Decision Tree")
    assert "extract" in core.model.named_steps
    score = core.get_cross_val_score(3)
    assert 0 <= score["mean"] <= 1


def test_dataset_quality_reports_duplicates_and_balance():
    instance = DataYabCore()
    instance.data_loader._set_dataset(
        pd.DataFrame(
            {
                "amount": [1, 1, 2, 3, 4, 4],
                "label": ["no", "no", "yes", "yes", "no", "no"],
            }
        )
    )
    instance.target_col = "label"
    quality = instance.get_dataset_quality()
    assert quality["duplicate_rows"] == 2
    assert quality["class_balance"] == {"no": 4, "yes": 2}
    assert quality["missing_total"] == 0


def test_auto_clean_keeps_duplicates():
    instance = DataYabCore()
    instance.data_loader._set_dataset(pd.DataFrame({"x": [1, 1, 2], "y": [3, 3, 4]}))
    result = instance.auto_clean()
    assert len(instance.df) == 3
    assert result["duplicates_detected"] == 1
    assert result["duplicates_removed"] == 0
    instance.remove_duplicates()
    assert len(instance.df) == 2


def test_arff_export_roundtrip(core, tmp_path):
    path = tmp_path / "cleaned.arff"
    core.export_cleaned_arff(path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("@relation")
    assert "group {a,b}" in text
    reloaded = DataYabCore()
    result = reloaded.load_arff(path)
    assert result["success"], result["message"]
    assert reloaded.df.shape == core.df.shape
    assert list(reloaded.df.columns) == list(core.df.columns)
    assert set(reloaded.df["group"]) == {"a", "b"}
