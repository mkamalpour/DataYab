"""Model training, evaluation, and hyperparameter tuning module."""

from typing import Optional, Dict, Any, List, Tuple
import time
import copy
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, precision_recall_curve
)
import importlib.util
import importlib.metadata
import joblib


def model_catalog(problem: str) -> Dict:
    """Get available models for a problem type."""
    if problem == "classification":
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier, XGBRegressor
        models["XGBoost"] = (
            XGBClassifier if problem == "classification" else XGBRegressor
        )(random_state=42, n_jobs=1, verbosity=0)
    return models


def runtime_versions() -> Dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "scikit-learn", "pandas", "joblib", "imbalanced-learn"]
    }


def _schema_for(raw_train: pd.DataFrame, names: List[str], numeric_columns: List[str]) -> List[Dict]:
    schema = []
    for name in names:
        col = raw_train[name]
        if name in numeric_columns:
            finite = col.dropna()
            schema.append({
                "name": name,
                "type": "numeric",
                "min": float(finite.min()) if len(finite) else 0.0,
                "max": float(finite.max()) if len(finite) else 0.0,
                "mean": float(finite.mean()) if len(finite) else 0.0,
            })
        else:
            schema.append({
                "name": name,
                "type": "categorical",
                "options": col.dropna().unique().tolist()[:100] or ["Unknown"],
            })
    return schema


class ModelTrainer:
    """Handles model training and management."""

    def __init__(self, feature_engineer):
        self.fe = feature_engineer
        self.model = None
        self.model_name = ""
        self.eval_model = None
        self.eval_model_name = None
        self.trained_models: Dict = {}
        self.available_models: List[str] = []
        self.metrics: Dict = {}
        self.feature_importance: Dict = {}
        self.y_pred = None
        self.y_proba = None
        self.misclassified_indices: List = []
        self._model_schema: List = []
        self._used_feature_selection = False
        self._selected_feature_names = None
        self._selected_features_indices = None

        self._active_input_names = []

    def _reset_learning(self):
        self.model = None
        self.model_name = ""
        self.eval_model = None
        self.eval_model_name = None
        self.trained_models = {}
        self.available_models = []
        self.metrics = {}
        self.feature_importance = {}
        self.y_pred = None
        self.y_proba = None
        self.misclassified_indices = []
        self._model_schema = []
        self._used_feature_selection = False
        self._selected_feature_names = None
        self._selected_features_indices = None

    def train(
        self,
        model_name: str = "Random Forest",
        use_smote: bool = False,
        class_weight: Optional[str] = None,
        **kwargs
    ) -> Dict:
        if self.fe._raw_train is None:
            raise ValueError("Split data first.")
        models = model_catalog(self.fe.problem_type)
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        model = models[model_name]
        if kwargs:
            model.set_params(**kwargs)
        if class_weight not in (None, "none", "balanced"):
            raise ValueError("Class weight must be balanced or none.")
        if class_weight == "balanced":
            if "class_weight" not in model.get_params():
                raise ValueError("This model does not support class weights.")
            model.set_params(class_weight="balanced")

        pipeline = self.fe._training_pipeline(model, use_smote)
        start = time.monotonic()
        pipeline.fit(self.fe._raw_train, self.fe.y_train)

        previous_models = dict(self.trained_models)
        previous_available = list(self.available_models)

        self.trained_models[model_name] = {
            "model": pipeline,
            "feature_names": list(self.fe.feature_names),
            "input_names": list(self.fe._selected_input_names),
            "schema": _schema_for(self.fe._raw_train, self.fe._selected_input_names, self.fe._numeric_columns),
        }
        self.available_models = list(self.trained_models)

        try:
            self.select_model(model_name)
        except Exception:
            self.trained_models.clear()
            self.trained_models.update(previous_models)
            self.available_models = previous_available
            raise

        elapsed = round(time.monotonic() - start, 3)
        return {
            "success": True,
            "model": model_name,
            "elapsed": elapsed,
            "params": model.get_params(),
        }

    def get_trained_models(self) -> List[str]:
        return list(self.trained_models)

    def select_model(self, model_name: str) -> Dict:
        if model_name not in self.trained_models:
            raise ValueError("Choose a trained model.")
        previous = self.__dict__.copy()
        try:
            bundle = self.trained_models[model_name]
            self.model = bundle["model"]
            self.model_name = model_name
            self.eval_model = None
            self.eval_model_name = model_name
            self._model_schema = copy.deepcopy(bundle["schema"])
            self._active_input_names = list(bundle["input_names"])
            self.fe.feature_names = list(bundle["feature_names"])
            self._selected_feature_names = list(self.fe.feature_names)
            self._selected_features_indices = None
            self._used_feature_selection = False
            self._compute_predictions()
            self._extract_feature_importance()
            return {"success": True, "model": model_name}
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

    def set_evaluation_model(self, model_name: str) -> Dict:
        return self.select_model(model_name)

    def get_evaluation_model(self) -> str:
        return self.model_name

    def _require_model(self):
        if self.model is None:
            raise ValueError("No model trained or selected.")

    def _compute_predictions(self):
        if self.model is None or self.fe._raw_test is None:
            return
        self.y_pred = self.model.predict(self.fe._raw_test)
        self.y_proba = (
            self.model.predict_proba(self.fe._raw_test)
            if hasattr(self.model, "predict_proba")
            else None
        )
        if self.fe.problem_type == "classification":
            self.misclassified_indices = np.flatnonzero(self.y_pred != self.fe.y_test).tolist()
        else:
            self.misclassified_indices = []

    def _extract_feature_importance(self):
        self.feature_importance = {}
        estimator = (
            self.model.named_steps.get("model")
            if hasattr(self.model, "named_steps")
            else None
        )
        if estimator is None:
            return
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            values = np.abs(estimator.coef_)
            values = values.mean(axis=0) if values.ndim > 1 else values
        else:
            return
        self.feature_importance = {
            n: float(v) for n, v in zip(self.fe.feature_names, values)
        }

    def train_all_models(
        self, use_smote: bool = False, class_weight: Optional[str] = None, **kwargs
    ) -> pd.DataFrame:
        if self.fe._raw_train is None:
            raise ValueError("Split data first.")
        rows = []
        best = None
        metric = "Accuracy" if self.fe.problem_type == "classification" else "R²"

        for name in model_catalog(self.fe.problem_type):
            try:
                self.train(name, use_smote=use_smote, class_weight=class_weight, **kwargs)
                cv = self.get_cross_val_score(cv=3)
                rows.append({
                    "Model": name,
                    metric: cv["mean"],
                    "CV Std": cv["std"],
                    "Evaluation": "Training cross-validation",
                })
                if best is None or cv["mean"] > best[0]:
                    best = (cv["mean"], name)
            except (ValueError, TypeError) as error:
                from ..logging_utils import get_logger
                get_logger().warning("Model comparison skipped %s: %s", name, error)

        if best is None:
            raise ValueError("No model could be compared. Check sample count and options.")

        self.select_model(best[1])
        return (
            pd.DataFrame(rows)
            .sort_values(metric, ascending=False)
            .reset_index(drop=True)
        )

    def train_stacked_model(
        self, base_model_1: str, base_model_2: str, meta_model: str = "Logistic Regression"
    ) -> Dict:
        if self.fe._raw_train is None:
            raise ValueError("Split and train base models first.")
        if base_model_1 == base_model_2 or any(
            n not in self.trained_models for n in [base_model_1, base_model_2]
        ):
            raise ValueError("Choose two distinct trained base models.")

        models = model_catalog(self.fe.problem_type)
        if meta_model not in models:
            if self.fe.problem_type == "regression" and meta_model == "Logistic Regression":
                meta_model = "Ridge"
            else:
                raise ValueError("Meta model is incompatible with this task.")

        estimators = [
            ("first", clone(self.trained_models[base_model_1]["model"])),
            ("second", clone(self.trained_models[base_model_2]["model"])),
        ]
        cls = (
            StackingClassifier if self.fe.problem_type == "classification" else StackingRegressor
        )
        ensemble = cls(
            estimators=estimators,
            final_estimator=models[meta_model],
            cv=self._cv_splitter(3),
            n_jobs=1,
        )
        start = time.monotonic()
        ensemble.fit(self.fe._raw_train, self.fe.y_train)

        name = f"Stacked: {base_model_1} + {base_model_2}"
        first = self.trained_models[base_model_1]
        self.trained_models[name] = {
            "model": ensemble,
            "feature_names": list(first["feature_names"]),
            "input_names": list(first["input_names"]),
            "schema": copy.deepcopy(first["schema"]),
        }
        self.available_models = list(self.trained_models)
        self.select_model(name)

        return {
            "success": True,
            "model": name,
            "base_models": [base_model_1, base_model_2],
            "meta_model": meta_model,
            "elapsed": round(time.monotonic() - start, 3),
        }

    def _cv_splitter(self, cv: int):
        if type(cv) is not int or cv < 2:
            raise ValueError("Use at least two CV folds.")
        limit = (
            int(np.bincount(self.fe.y_train).min())
            if self.fe.problem_type == "classification"
            else len(self.fe.y_train) // 2
        )
        folds = min(cv, limit)
        if folds < 2:
            raise ValueError("Not enough training rows per class for cross-validation.")
        return (
            StratifiedKFold(folds, shuffle=True, random_state=42)
            if self.fe.problem_type == "classification"
            else KFold(folds, shuffle=True, random_state=42)
        )

    def get_cross_val_score(self, cv: int = 5) -> Dict:
        if self.model is None:
            raise ValueError("Train or import a model first.")
        if self.fe._raw_train is None:
            raise ValueError("Load/split a dataset to evaluate this imported model.")

        splitter = self._cv_splitter(cv)
        scoring = "accuracy" if self.fe.problem_type == "classification" else "r2"
        scores = cross_val_score(
            clone(self.model),
            self.fe._raw_train,
            self.fe.y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        result = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
            "metric": scoring,
            "folds": splitter.n_splits,
        }
        if self.fe._selection_data_driven:
            result["warning"] = (
                "Feature choices were explored on training data; CV is conditional on those choices."
            )
        return result


class ModelEvaluator:
    """Handles model evaluation and metrics computation."""

    def __init__(self, model_trainer, feature_engineer):
        self.mt = model_trainer
        self.fe = feature_engineer

    def get_metrics(self) -> Dict:
        self.mt._require_model()
        if self.fe._raw_test is None:
            raise ValueError("No holdout data available for this imported model.")
        self.mt._compute_predictions()
        pred = self.mt.y_pred

        if self.fe.problem_type == "classification":
            result = {
                "accuracy": float(accuracy_score(self.fe.y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(self.fe.y_test, pred)),
                "macro_f1": float(f1_score(self.fe.y_test, pred, average='macro', zero_division=0)),
                "precision": float(
                    precision_score(self.fe.y_test, pred, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(self.fe.y_test, pred, average="weighted", zero_division=0)
                ),
                "f1": float(
                    f1_score(self.fe.y_test, pred, average="weighted", zero_division=0)
                ),
                "classification_report": classification_report(
                    self.fe.y_test, pred,
                    target_names=[str(c) for c in self.fe.target_encoder.classes_],
                    labels=np.arange(len(self.fe.target_encoder.classes_)),
                    zero_division=0,
                ),
            }
            labels, counts = np.unique(self.fe.y_train, return_counts=True)
            majority = labels[np.argmax(counts)]
            result['majority_baseline_accuracy'] = float(np.mean(np.asarray(self.fe.y_test) == majority))

            if self.mt.y_proba is not None:
                if self.mt.y_proba.shape[1] == 2:
                    result["roc_auc"] = float(roc_auc_score(self.fe.y_test, self.mt.y_proba[:, 1]))
                else:
                    result["roc_auc"] = float(roc_auc_score(self.fe.y_test, self.mt.y_proba, multi_class="ovr"))
        else:
            mse = float(mean_squared_error(self.fe.y_test, pred))
            result = {
                "r2": float(r2_score(self.fe.y_test, pred)),
                "mae": float(mean_absolute_error(self.fe.y_test, pred)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
            }
            baseline = np.full(len(self.fe.y_test), float(np.mean(self.fe.y_train)))
            result['baseline_mae'] = float(mean_absolute_error(self.fe.y_test, baseline))
            result['baseline_rmse'] = float(np.sqrt(mean_squared_error(self.fe.y_test, baseline)))
            result['rmse_skill'] = 1 - result['rmse'] / result['baseline_rmse'] if result['baseline_rmse'] else None

        self.mt.metrics = result
        return result

    def get_confusion_matrix(self) -> Dict:
        if self.fe.problem_type != "classification" or self.fe._raw_test is None:
            raise ValueError("A classification holdout is required.")
        self.mt._compute_predictions()
        labels = self.fe.target_encoder.classes_.tolist()
        return {
            "matrix": confusion_matrix(
                self.fe.y_test, self.mt.y_pred, labels=np.arange(len(labels))
            ).tolist(),
            "labels": labels,
        }

    def get_misclassified_samples(self, max_samples: int = 20) -> pd.DataFrame:
        self.get_metrics()
        if self.fe.problem_type != "classification":
            raise ValueError("Classification only.")
        rows = self.fe.data_loader.df.loc[
            self.fe._test_indices[self.mt.misclassified_indices[:max_samples]]
        ].copy()
        if len(rows):
            positions = self.mt.misclassified_indices[:max_samples]
            rows["Actual"] = self.fe.target_encoder.inverse_transform(
                self.fe.y_test[positions]
            )
            rows["Predicted"] = self.fe.target_encoder.inverse_transform(
                self.mt.y_pred[positions]
            )
        return rows


class HyperparameterTuner:
    """Handles hyperparameter optimization using Optuna or GridSearch."""

    def __init__(self, model_trainer, feature_engineer):
        self.mt = model_trainer
        self.fe = feature_engineer

    def tune_with_optuna(
        self,
        model_name: str,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        cv_folds: int = 3,
        use_smote: bool = False,
        class_weight: Optional[str] = None,
    ) -> Dict:
        """Tune hyperparameters using Optuna."""
        try:
            import optuna
        except ImportError:
            return {"success": False, "message": "Optuna not installed. Install with: pip install optuna"}

        if self.fe._raw_train is None:
            raise ValueError("Split data first.")

        models = model_catalog(self.fe.problem_type)
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")

        supported = {"Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "SVR", "KNN", "XGBoost"}
        if model_name not in supported:
            raise ValueError("No Optuna search space for this model. Use grid search.")

        def objective(trial):
            params = {}
            if model_name == "Random Forest":
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 30)
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
                params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
            elif model_name == "Gradient Boosting":
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            elif model_name == "Logistic Regression":
                params["C"] = trial.suggest_float("C", 0.01, 100, log=True)
            elif model_name in ("SVM", "SVR"):
                params["C"] = trial.suggest_float("C", 0.01, 100, log=True)
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            elif model_name == "KNN":
                params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 30)
                params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
            elif model_name == "XGBoost":
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            else:
                return 0.0

            model = models[model_name]
            model.set_params(**params)
            if class_weight == "balanced" and "class_weight" in model.get_params():
                model.set_params(class_weight="balanced")

            pipeline = self.fe._training_pipeline(model, use_smote)

            splitter = self.mt._cv_splitter(cv_folds)
            scoring = "accuracy" if self.fe.problem_type == "classification" else "r2"
            scores = cross_val_score(
                pipeline, self.fe._raw_train, self.fe.y_train,
                cv=splitter, scoring=scoring, n_jobs=1, error_score="raise"
            )
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        best_score = study.best_value

        # Train final model with best params
        self.mt.train(model_name, use_smote=use_smote, class_weight=class_weight, **best_params)

        return {
            "success": True,
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": len(study.trials),
            "model": model_name,
        }

    def tune_with_gridsearch(
        self,
        model_name: str,
        param_grid: Dict,
        cv_folds: int = 3,
        use_smote: bool = False,
        class_weight: Optional[str] = None,
    ) -> Dict:
        """Tune hyperparameters using GridSearchCV."""
        from sklearn.model_selection import GridSearchCV

        if self.fe._raw_train is None:
            raise ValueError("Split data first.")

        models = model_catalog(self.fe.problem_type)
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")

        model = models[model_name]
        if class_weight == "balanced" and "class_weight" in model.get_params():
            model.set_params(class_weight="balanced")

        # Public grids may use estimator names or full pipeline parameter names.
        normalized_grid = {
            key if "__" in key else f"model__{key}": value
            for key, value in param_grid.items()
        }
        pipeline = self.fe._training_pipeline(model, use_smote)
        splitter = self.mt._cv_splitter(cv_folds)
        scoring = "accuracy" if self.fe.problem_type == "classification" else "r2"

        grid_search = GridSearchCV(
            pipeline, normalized_grid, cv=splitter, scoring=scoring,
            n_jobs=1, error_score="raise", verbose=0
        )
        grid_search.fit(self.fe._raw_train, self.fe.y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Train final model with best params
        # Retain the fitted best pipeline, including any preprocessing parameters.
        best_model = grid_search.best_estimator_
        previous = self.mt.__dict__.copy()
        self.mt.trained_models = dict(self.mt.trained_models)
        self.mt.trained_models[model_name] = {
            "model": best_model,
            "feature_names": list(self.fe.feature_names),
            "input_names": list(self.fe._selected_input_names),
            "schema": _schema_for(self.fe._raw_train, self.fe._selected_input_names, self.fe._numeric_columns),
        }
        try:
            self.mt.select_model(model_name)
            self.mt.available_models = list(self.mt.trained_models)
        except Exception:
            self.mt.__dict__.clear()
            self.mt.__dict__.update(previous)
            raise

        return {
            "success": True,
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": grid_search.cv_results_,
            "model": model_name,
        }

    def get_model_recommendation(self, target: str = None) -> Dict:
        """Generate recommendation for best model strategy."""
        target_column = target or self.fe.target_col
        recommendation = {
            "problem_type": self.fe.problem_type if target_column else "Unknown",
            "dataset_size": len(self.fe.data_loader.df) if self.fe.data_loader.df is not None else 0,
            "target_col": target_column or "Not set",
            "recommended_models": [],
            "recommended_strategies": [],
            "notes": [],
        }

        if not target_column:
            return recommendation

        problem = recommendation["problem_type"]
        n_rows = len(self.fe.data_loader.df) if self.fe.data_loader.df is not None else 0

        if problem == "classification":
            recommendation["recommended_models"] = [
                "Random Forest", "XGBoost", "Gradient Boosting"
            ]
            if target_column in self.fe.data_loader.df.columns:
                target_dist = self.fe.data_loader.df[target_column].value_counts(normalize=True)
                if len(target_dist) == 2:
                    ratio = target_dist.iloc[0] / target_dist.iloc[1]
                    if ratio > 3:
                        recommendation["recommended_strategies"].extend([
                            "Use SMOTE for imbalanced classification",
                            "Set class_weight='balanced'",
                        ])
                        recommendation["notes"].append(f"Class imbalance detected: {ratio:.1f}:1 ratio")
            if n_rows < 1000:
                recommendation["notes"].append("Small dataset – consider cross-validation (CV=5 or more)")
        else:
            recommendation["recommended_models"] = [
                "Random Forest", "XGBoost", "Gradient Boosting"
            ]
            recommendation["notes"].append("Regression task – evaluate using R², RMSE, and MAE")

        if "XGBoost" in model_catalog("classification"):
            recommendation["notes"].append("XGBoost available – often provides best performance")

        return recommendation
