"""Pipeline management for saving/loading complete ML pipelines."""

from typing import Optional, Dict, Any
import copy
import joblib
import importlib.metadata
import html
from pathlib import Path

from ..storage import atomic_target


def runtime_versions() -> Dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "scikit-learn", "pandas", "joblib", "imbalanced-learn"]
    }


class PipelineManager:
    """Handles saving and loading complete ML pipelines."""

    def __init__(self, data_loader, feature_engineer, model_trainer, model_evaluator):
        self.dl = data_loader
        self.fe = feature_engineer
        self.mt = model_trainer
        self.me = model_evaluator

    def _require_model(self):
        if self.mt.model is None:
            raise ValueError("Train or import a model first.")

    def _require_data(self):
        if self.dl.df is None:
            raise ValueError("No dataset loaded.")

    def save_pipeline(self, file_path: str) -> Dict:
        self._require_model()
        package = {
            "format": "datayab-v2",
            "versions": runtime_versions(),
            "model": self.mt.model,
            "model_name": self.mt.model_name,
            "target_col": self.fe.target_col,
            "problem_type": self.fe.problem_type,
            "target_encoder": self.fe.target_encoder,
            "schema": self.mt._model_schema,
            "input_names": self.mt._active_input_names,
            "feature_names": self.fe.feature_names,
            # Feature engineering state
            "selected_input_names": self.fe._selected_input_names,
            "all_input_names": self.fe._all_input_names,
            "extraction_method": self.fe._extraction_method,
            "extraction_components": self.fe._extraction_components,
            "numeric_columns": self.fe._numeric_columns,
            "scale": self.fe._scale,
            "scaler_type": self.fe._scaler_type,
            # Data info
            "file_path": self.dl.file_path,
            "data_shape": self.dl.df.shape if self.dl.df is not None else None,
        }
        with atomic_target(file_path) as temporary:
            joblib.dump(package, temporary)
        return {"success": True, "path": str(file_path)}

    def load_pipeline(self, file_path: str) -> Dict:
        package = joblib.load(file_path)
        if not isinstance(package, dict) or package.get("format") != "datayab-v2":
            raise ValueError(
                "Legacy/incomplete model: retrain and export with this DataYab version."
            )
        if package.get("versions") != runtime_versions():
            raise ValueError(
                "Model dependency versions differ. Use its original environment or retrain."
            )
        required_keys = {
            "model_name", "target_col", "problem_type", "target_encoder",
            "input_names", "feature_names", "schema", "model"
        }
        if not required_keys.issubset(package) or not package.get("schema") or not hasattr(package.get("model"), "predict"):
            raise ValueError("Incomplete prediction package.")

        # Reset all state
        self.mt._reset_learning()
        self.fe._raw_train = None
        self.fe._raw_test = None
        self.fe.y_train = None
        self.fe.y_test = None
        self.fe._selected_input_names = []
        self.fe._all_input_names = []
        self.fe._extraction_method = None
        self.fe._extraction_components = None

        # Load model
        self.mt.model = package["model"]
        self.mt.model_name = package["model_name"]
        self.fe.target_col = package["target_col"]
        self.fe.problem_type = package["problem_type"]
        self.fe.target_encoder = package["target_encoder"]
        self.mt._model_schema = package["schema"]
        self.mt._active_input_names = package["input_names"]
        self.fe.feature_names = package["feature_names"]

        # Restore feature engineering state
        self.fe._selected_input_names = package.get("selected_input_names", package["input_names"])
        self.fe._all_input_names = package.get("all_input_names", package["input_names"])
        self.fe._extraction_method = package.get("extraction_method")
        self.fe._extraction_components = package.get("extraction_components")
        self.fe._numeric_columns = package.get("numeric_columns", [])
        self.fe._scale = package.get("scale", True)
        self.fe._scaler_type = package.get("scaler_type", "standard")

        self.mt.trained_models = {
            self.mt.model_name: {
                "model": self.mt.model,
                "feature_names": self.fe.feature_names,
                "input_names": self.mt._active_input_names,
                "schema": self.mt._model_schema,
            }
        }
        self.mt.available_models = [self.mt.model_name]
        return {"success": True, "model": self.mt.model_name}

    def export_report(self, file_path: str) -> Dict:
        self._require_data()
        title = html.escape(Path(self.dl.file_path).name or "Dataset")
        rows = [
            f"<h1>DataYab report: {title}</h1>",
            f"<p>{len(self.dl.df):,} rows / {len(self.dl.df.columns)} columns</p>",
        ]
        rows.append(self.dl.df.describe(include="all").transpose().to_html(escape=True))
        rows.append(
            "<h2>Dataset observations</h2><ul>"
            + "".join(
                "<li>" + html.escape(str(item)) + "</li>"
                for item in self.generate_observations()
            )
            + "</ul>"
        )
        if self.mt.model is not None and self.fe._raw_test is not None:
            rows.append(
                "<h2>Reserved holdout evaluation</h2><pre>"
                + html.escape(str(self.me.get_metrics()))
                + "</pre>"
            )
        rows.append(
            "<p>Exploratory analysis: confirm model and feature choices on fresh data.</p>"
        )
        document = (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>DataYab report</title><style>body{font:15px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#172b4d}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:8px;border:1px solid #dce5eb}pre{white-space:pre-wrap}</style>'
            + "".join(rows)
            + "</html>"
        )
        with atomic_target(file_path) as temporary:
            temporary.write_text(document, encoding="utf-8")
        return {"success": True, "path": str(file_path)}

    def generate_observations(self, n: int = 10):
        from .insights import InsightsGenerator
        ig = InsightsGenerator(self.dl, self.fe, self.mt)
        return ig.generate_observations(n)

    def get_pipeline_summary(self) -> Dict:
        """Get a summary of the current pipeline state."""
        return {
            "model_name": self.mt.model_name,
            "problem_type": self.fe.problem_type,
            "target_column": self.fe.target_col,
            "data_shape": self.dl.df.shape if self.dl.df is not None else None,
            "train_shape": self.fe._raw_train.shape if self.fe._raw_train is not None else None,
            "test_shape": self.fe._raw_test.shape if self.fe._raw_test is not None else None,
            "features": {
                "total": len(self.fe._all_input_names),
                "selected": len(self.fe._selected_input_names),
                "selected_features": self.fe._selected_input_names,
                "extraction": self.fe._extraction_method,
                "extraction_components": self.fe._extraction_components,
                "final_feature_names": self.fe.feature_names,
            },
            "preprocessing": {
                "scale": self.fe._scale,
                "scaler_type": self.fe._scaler_type,
                "missing_strategy": "median (numeric) / constant (categorical)",
                "encoding": "ordinal",
            },
            "available_models": self.mt.available_models,
            "metrics": self.mt.metrics,
        }
