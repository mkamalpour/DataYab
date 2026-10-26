"""Train-only preprocessing and complete prediction pipelines."""

from .storage import atomic_target

import copy
import time
from pathlib import Path
import importlib.util
import importlib.metadata
import html
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
    precision_recall_curve,
)


def model_catalog(problem):
    if problem == "classification":
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=1
            ),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=1
            ),
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


def runtime_versions():
    return {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
            "joblib",
            "imbalanced-learn",
        ]
    }


class ModelingMixin:
    def _reset_learning(self):
        self._invalidate_models()
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self._raw_train = self._raw_test = None
        self.target_col = ""
        self.problem_type = ""
        self.feature_names = []
        self._selected_input_names = None
        self._pca_components = None
        self._split_signature = None
        self.target_encoder = None
        self.scaler = None
        self.label_encoders = {}
        self._test_indices = []
        self._all_input_names = []
        self._selection_data_driven = False

    def _invalidate_models(self):
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

    def _set_dataset(self, df, path=""):
        if (
            not isinstance(df, pd.DataFrame)
            or df.empty
            or len(df.columns) > 2000
            or len(df) > 1000000
        ):
            raise ValueError(
                "Choose a nonempty table with at most 1,000,000 rows and 2,000 columns."
            )
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        for col in df.select_dtypes(include="object"):
            if df[col].map(lambda value: isinstance(value, (list, dict, set, tuple, np.ndarray))).any():
                raise ValueError(
                    "Nested values are not supported; flatten arrays and objects first."
                )
        if not df.columns.is_unique:
            raise ValueError("Column names must be unique.")
        if any(not c.strip() for c in df.columns):
            raise ValueError("Column names cannot be blank.")
        self._reset_learning()
        self.df = df.reset_index(drop=True)
        self.original_df = self.df.copy(deep=True)
        self.file_path = str(path)
        self.insights = []
        self._log(f"Loaded {len(df):,} rows / {len(df.columns)} columns")
        return {
            "success": True,
            "message": f"Loaded {len(df):,} rows / {len(df.columns)} columns",
            "preview": self.df.head(10),
            "info": self.get_info(),
        }

    def load_csv(self, file_path, sep=None):
        import csv

        try:
            for encoding in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    delimiter = sep
                    if delimiter is None:
                        with open(file_path, encoding=encoding, newline="") as stream:
                            sample = stream.read(65536)
                        try:
                            delimiter = (
                                csv.Sniffer()
                                .sniff(sample, delimiters=",;\t|")
                                .delimiter
                            )
                        except csv.Error:
                            delimiter = ","
                    return self._set_dataset(
                        pd.read_csv(file_path, sep=delimiter, encoding=encoding),
                        file_path,
                    )
                except UnicodeDecodeError:
                    continue
            raise ValueError("Cannot decode this CSV.")
        except Exception as error:
            return {"success": False, "message": f"CSV load failed: {error}"}

    def load_excel(self, file_path, sheet_name=0):
        try:
            return self._set_dataset(
                pd.read_excel(file_path, sheet_name=sheet_name), file_path
            )
        except Exception as error:
            return {"success": False, "message": f"Excel load failed: {error}"}

    def load_json(self, file_path):
        try:
            import json

            with open(file_path, encoding="utf-8-sig") as stream:
                value = json.load(stream)
            if (
                isinstance(value, dict)
                and len(value) == 1
                and isinstance(next(iter(value.values())), list)
            ):
                value = next(iter(value.values()))
            df = (
                pd.json_normalize(value)
                if isinstance(value, list)
                else pd.DataFrame(value)
            )
            return self._set_dataset(df, file_path)
        except Exception as error:
            return {"success": False, "message": f"JSON load failed: {error}"}

    def load_parquet(self, file_path):
        try:
            return self._set_dataset(pd.read_parquet(file_path), file_path)
        except Exception as error:
            return {"success": False, "message": f"Parquet load failed: {error}"}

    def load_file(self, file_path):
        loader = {
            ".csv": self.load_csv,
            ".xlsx": self.load_excel,
            ".xls": self.load_excel,
            ".json": self.load_json,
            ".parquet": self.load_parquet,
        }.get(Path(file_path).suffix.lower())
        return (
            loader(file_path)
            if loader
            else {
                "success": False,
                "message": "Supported: CSV, XLSX, XLS, JSON, Parquet.",
            }
        )

    def reset_to_original(self):
        if self.original_df is None:
            raise ValueError("No original dataset is available.")
        self._set_dataset(self.original_df.copy(deep=True), self.file_path)
        return {
            "success": True,
            "message": "Restored original data and cleared models.",
        }

    def auto_detect_problem_type(self, target):
        col = self.df[target].dropna()
        if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            return "classification"
        if col.nunique() <= 20 and np.equal(col, np.floor(col)).all():
            return "classification"
        return "regression"

    @staticmethod
    def _canonical_frame(frame, numeric):
        frame = frame.copy()
        for col in frame:
            if col in numeric:
                frame[col] = (
                    pd.to_numeric(frame[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .astype(float)
                )
            else:
                frame[col] = (
                    frame[col]
                    .map(lambda value: np.nan if pd.isna(value) else str(value))
                    .astype(object)
                )
        return frame

    def split_data(
        self,
        target,
        test_size=0.2,
        scale=True,
        scaler_type="standard",
        problem_type="auto",
    ):
        self._require_data()
        if target not in self.df:
            raise ValueError(f"Target column {target!r} does not exist.")
        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            raise ValueError("Test fraction must be between zero and one.")
        if scaler_type not in ("standard", "minmax"):
            raise ValueError("Scaler must be standard or minmax.")
        if problem_type not in ("auto", "classification", "regression"):
            raise ValueError("Unknown problem type.")
        signature = (target, float(test_size), bool(scale), scaler_type, problem_type)
        if signature == self._split_signature and self._raw_train is not None:
            return {
                "success": True,
                "problem_type": self.problem_type,
                "train_size": len(self.y_train),
                "test_size": len(self.y_test),
                "features": len(self.feature_names),
                "feature_names": self.feature_names,
            }
        df = self.df.loc[self.df[target].notna()].copy()
        if len(df) < 10 or len(df.columns) < 2:
            raise ValueError(
                "Training needs ten labelled rows and at least one feature."
            )
        problem = (
            self.auto_detect_problem_type(target)
            if problem_type == "auto"
            else problem_type
        )
        X = df.drop(columns=[target])
        numeric = [col for col in X if pd.api.types.is_numeric_dtype(X[col])]
        X = self._canonical_frame(X, numeric)
        yraw = df[target]
        if problem == "classification":
            values = yraw.astype(str)
            counts = values.value_counts()
            if len(counts) < 2 or counts.min() < 2:
                raise ValueError(
                    "Each of at least two classes needs two labelled rows."
                )
            ntest = int(np.ceil(len(values) * test_size))
            if min(ntest, len(values) - ntest) < len(counts):
                raise ValueError("Train/test sizes must include every class.")
            stratify = values
        else:
            values = pd.to_numeric(yraw, errors="coerce")
            if not np.isfinite(values).all():
                raise ValueError("Regression targets must be finite numbers.")
            if int(np.ceil(len(values) * test_size)) < 2:
                raise ValueError("Regression evaluation needs two holdout rows.")
            stratify = None
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=test_size, stratify=stratify, random_state=42
        )
        encoder = (
            LabelEncoder().fit(values.iloc[train_idx])
            if problem == "classification"
            else None
        )
        yt = (
            encoder.transform(values.iloc[train_idx])
            if encoder is not None
            else values.iloc[train_idx].to_numpy(dtype=float)
        )
        yv = (
            encoder.transform(values.iloc[test_idx])
            if encoder is not None
            else values.iloc[test_idx].to_numpy(dtype=float)
        )
        self._reset_learning()
        self.target_col = target
        self.problem_type = problem
        self.target_encoder = encoder
        self._raw_train = X.iloc[train_idx].copy()
        self._raw_test = X.iloc[test_idx].copy()
        self._numeric_columns = numeric
        self._all_input_names = list(X.columns)
        self._selected_input_names = list(X.columns)
        self._scale = bool(scale)
        self._scaler_type = scaler_type
        self._test_indices = df.index.to_numpy()[test_idx]
        self.y_train = yt
        self.y_test = yv
        self._split_signature = signature
        self._refresh_feature_view()
        self._log(
            f"Split before preprocessing: train={len(yt)}, holdout={len(yv)}, excluded unlabelled={len(self.df) - len(df)}, task={problem}"
        )
        return {
            "success": True,
            "problem_type": problem,
            "train_size": len(yt),
            "test_size": len(yv),
            "features": len(self.feature_names),
            "feature_names": self.feature_names,
        }

    def _preprocessor(self):
        transformers = []
        for i, col in enumerate(self._selected_input_names):
            if col in self._numeric_columns:
                steps = [
                    (
                        "impute",
                        SimpleImputer(strategy="median", keep_empty_features=True),
                    )
                ]
                if self._scale:
                    steps.append(
                        (
                            "scale",
                            StandardScaler()
                            if self._scaler_type == "standard"
                            else MinMaxScaler(),
                        )
                    )
            else:
                steps = [
                    (
                        "impute",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="Unknown",
                            keep_empty_features=True,
                        ),
                    ),
                    (
                        "encode",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            transformers.append((f"column_{i}", Pipeline(steps), [col]))
        return ColumnTransformer(
            transformers, remainder="drop", verbose_feature_names_out=False
        )

    def _refresh_feature_view(self):
        prep = self._preprocessor()
        self.X_train = prep.fit_transform(self._raw_train)
        self.X_test = prep.transform(self._raw_test)
        self.feature_names = list(self._selected_input_names)
        if self._pca_components is not None:
            pca = PCA(n_components=self._pca_components, random_state=42)
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)
            self.feature_names = [f"PC{i + 1}" for i in range(self._pca_components)]
            return pca
        return None

    def _training_pipeline(self, estimator, use_smote=False):
        steps = [("prepare", self._preprocessor())]
        if self._pca_components is not None:
            steps.append(
                ("pca", PCA(n_components=self._pca_components, random_state=42))
            )
        if use_smote:
            if self.problem_type != "classification":
                raise ValueError("SMOTE is only for classification.")
            if any(
                col not in self._numeric_columns for col in self._selected_input_names
            ):
                raise ValueError(
                    "SMOTE requires numeric features here; use balanced class weights for mixed categorical data."
                )
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as SamplingPipeline

            steps.extend(
                [
                    ("resample", SMOTE(random_state=42, k_neighbors=1)),
                    ("model", estimator),
                ]
            )
            return SamplingPipeline(steps)
        steps.append(("model", estimator))
        return Pipeline(steps)

    def train(
        self, model_name="Random Forest", use_smote=False, class_weight=None, **kwargs
    ):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        models = model_catalog(self.problem_type)
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
        pipeline = self._training_pipeline(model, use_smote)
        start = time.monotonic()
        pipeline.fit(self._raw_train, self.y_train)
        previous_models = dict(self.trained_models)
        previous_available = list(self.available_models)
        self.trained_models[model_name] = {
            "model": pipeline,
            "feature_names": list(self.feature_names),
            "input_names": list(self._selected_input_names),
            "schema": self._schema_for(self._selected_input_names),
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
        self._log(f"Trained {model_name} in {elapsed}s")
        return {
            "success": True,
            "model": model_name,
            "elapsed": elapsed,
            "params": model.get_params(),
        }

    def _compute_predictions(self):
        if self.model is None or self._raw_test is None:
            return
        self.y_pred = self.model.predict(self._raw_test)
        self.y_proba = (
            self.model.predict_proba(self._raw_test)
            if hasattr(self.model, "predict_proba")
            else None
        )
        self.misclassified_indices = (
            np.flatnonzero(self.y_pred != self.y_test).tolist()
            if self.problem_type == "classification"
            else []
        )

    def _compute_predictions_with_selection(self, X_test=None, selected_indices=None):
        self._compute_predictions()

    def get_trained_models(self):
        return list(self.trained_models)

    def select_model(self, model_name):
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
            self.feature_names = list(bundle["feature_names"])
            self._selected_feature_names = list(self.feature_names)
            self._selected_features_indices = None
            self._used_feature_selection = False
            self._compute_predictions()
            self._extract_feature_importance()
            return {"success": True, "model": model_name}
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

    def set_evaluation_model(self, model_name):
        return self.select_model(model_name)

    def get_evaluation_model(self):
        return self.model_name

    def select_features_manual(self, feature_list):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        if (
            not feature_list
            or len(set(feature_list)) != len(feature_list)
            or any(f not in self._all_input_names for f in feature_list)
        ):
            raise ValueError(
                "Choose unique original input columns, not PCA component names."
            )
        self._invalidate_models()
        self._selected_input_names = list(feature_list)
        self._pca_components = None
        self._selection_data_driven = True
        self._refresh_feature_view()
        self._selected_feature_names = list(self.feature_names)
        self._log(
            "Feature choices changed; models cleared. CV is conditional on manually explored feature choices. Holdout not used."
        )
        return True

    def apply_pca(self, n_components=2):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        maximum = min(len(self._selected_input_names), len(self.y_train))
        if type(n_components) is not int or not 1 <= n_components <= maximum:
            raise ValueError(f"PCA components must be 1-{maximum}.")
        self._invalidate_models()
        self._pca_components = n_components
        pca = self._refresh_feature_view()
        self._selected_feature_names = list(self.feature_names)
        return {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": pca.explained_variance_ratio_.cumsum().tolist(),
            "n_components": n_components,
        }

    def restore_from_pca(self):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        self._invalidate_models()
        self._selected_input_names = list(self._all_input_names)
        self._pca_components = None
        self._selection_data_driven = False
        self._refresh_feature_view()
        self._selected_feature_names = list(self.feature_names)
        return True

    def get_fast_feature_importance(self, top_n=20):
        if self.X_train is None:
            raise ValueError("Split data first.")
        model = (
            RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
            if self.problem_type == "classification"
            else RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1)
        )
        model.fit(self.X_train, self.y_train)
        return (
            pd.Series(model.feature_importances_, index=self.feature_names)
            .sort_values(ascending=False)
            .head(top_n)
        )

    def get_mutual_information_top(self, k=10):
        if self.X_train is None:
            raise ValueError("Split data first.")
        function = (
            mutual_info_classif
            if self.problem_type == "classification"
            else mutual_info_regression
        )
        scores = function(
            self.X_train,
            self.y_train,
            random_state=42,
            n_neighbors=min(3, len(self.y_train) - 1),
        )
        return (
            pd.Series(scores, index=self.feature_names)
            .sort_values(ascending=False)
            .head(k)
        )

    def _cv_splitter(self, cv):
        if type(cv) is not int or cv < 2:
            raise ValueError("Use at least two CV folds.")
        limit = (
            int(np.bincount(self.y_train).min())
            if self.problem_type == "classification"
            else len(self.y_train) // 2
        )
        folds = min(cv, limit)
        if folds < 2:
            raise ValueError("Not enough training rows per class for cross-validation.")
        return (
            StratifiedKFold(folds, shuffle=True, random_state=42)
            if self.problem_type == "classification"
            else KFold(folds, shuffle=True, random_state=42)
        )

    def get_cross_val_score(self, cv=5):
        self._require_model()
        if self._raw_train is None:
            raise ValueError("Load/split a dataset to evaluate this imported model.")
        splitter = self._cv_splitter(cv)
        scoring = "accuracy" if self.problem_type == "classification" else "r2"
        scores = cross_val_score(
            clone(self.model),
            self._raw_train,
            self.y_train,
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
        if self._selection_data_driven:
            result["warning"] = (
                "Feature choices were explored on training data; CV is conditional on those choices. Holdout not used."
            )
        return result

    def train_all_models(self, use_smote=False, class_weight=None, **kwargs):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        rows = []
        best = None
        metric = "Accuracy" if self.problem_type == "classification" else "R²"
        for name in model_catalog(self.problem_type):
            try:
                self.train(name, use_smote=use_smote, class_weight=class_weight)
                cv = self.get_cross_val_score(cv=3)
                rows.append(
                    {
                        "Model": name,
                        metric: cv["mean"],
                        "CV Std": cv["std"],
                        "Evaluation": "Training cross-validation",
                    }
                )
                if best is None or cv["mean"] > best[0]:
                    best = (cv["mean"], name)
            except (ValueError, TypeError) as error:
                self._log(f"{name} excluded: {error}")
        if best is None:
            raise ValueError(
                "No model could be compared. Check sample count and options."
            )
        self.select_model(best[1])
        self._log(
            "Comparison ranked training CV scores; selected model holdout is separate."
        )
        return (
            pd.DataFrame(rows)
            .sort_values(metric, ascending=False)
            .reset_index(drop=True)
        )

    def train_stacked_model(
        self, base_model_1, base_model_2, meta_model="Logistic Regression"
    ):
        if self._raw_train is None:
            raise ValueError("Split and train base models first.")
        if base_model_1 == base_model_2 or any(
            n not in self.trained_models for n in [base_model_1, base_model_2]
        ):
            raise ValueError("Choose two distinct trained base models.")
        models = model_catalog(self.problem_type)
        if meta_model not in models:
            if (
                self.problem_type == "regression"
                and meta_model == "Logistic Regression"
            ):
                meta_model = "Ridge"
            else:
                raise ValueError("Meta model is incompatible with this task.")
        estimators = [
            ("first", clone(self.trained_models[base_model_1]["model"])),
            ("second", clone(self.trained_models[base_model_2]["model"])),
        ]
        cls = (
            StackingClassifier
            if self.problem_type == "classification"
            else StackingRegressor
        )
        ensemble = cls(
            estimators=estimators,
            final_estimator=models[meta_model],
            cv=self._cv_splitter(3),
            n_jobs=1,
        )
        start = time.monotonic()
        ensemble.fit(self._raw_train, self.y_train)
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

    def get_metrics(self):
        self._require_model()
        if self._raw_test is None:
            raise ValueError("No holdout data is available for this imported model.")
        self._compute_predictions()
        pred = self.y_pred
        if self.problem_type == "classification":
            result = {
                "accuracy": float(accuracy_score(self.y_test, pred)),
                'balanced_accuracy': float(balanced_accuracy_score(self.y_test, pred)),
                'macro_f1': float(f1_score(self.y_test, pred, average='macro', zero_division=0)),
                "precision": float(
                    precision_score(
                        self.y_test, pred, average="weighted", zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(self.y_test, pred, average="weighted", zero_division=0)
                ),
                "f1": float(
                    f1_score(self.y_test, pred, average="weighted", zero_division=0)
                ),
                "classification_report": classification_report(
                    self.y_test,
                    pred,
                    target_names=[str(c) for c in self.target_encoder.classes_],
                    labels=np.arange(len(self.target_encoder.classes_)),
                    zero_division=0,
                ),
            }
            labels, counts = np.unique(self.y_train, return_counts=True)
            majority = labels[np.argmax(counts)]
            result['majority_baseline_accuracy'] = float(np.mean(np.asarray(self.y_test) == majority))
            if self.y_proba is not None:
                result["roc_auc"] = (
                    float(roc_auc_score(self.y_test, self.y_proba[:, 1]))
                    if self.y_proba.shape[1] == 2
                    else float(
                        roc_auc_score(self.y_test, self.y_proba, multi_class="ovr")
                    )
                )
        else:
            mse = float(mean_squared_error(self.y_test, pred))
            result = {
                "r2": float(r2_score(self.y_test, pred)),
                "mae": float(mean_absolute_error(self.y_test, pred)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
            }
            baseline = np.full(len(self.y_test), float(np.mean(self.y_train)))
            result['baseline_mae'] = float(mean_absolute_error(self.y_test, baseline))
            result['baseline_rmse'] = float(np.sqrt(mean_squared_error(self.y_test, baseline)))
            result['rmse_skill'] = 1 - result['rmse'] / result['baseline_rmse'] if result['baseline_rmse'] else None
        self.metrics = result
        return result

    def get_confusion_matrix(self):
        self._require_model()
        if self.problem_type != "classification" or self._raw_test is None:
            raise ValueError("A classification holdout is required.")
        self._compute_predictions()
        labels = self.target_encoder.classes_.tolist()
        return {
            "matrix": confusion_matrix(
                self.y_test, self.y_pred, labels=np.arange(len(labels))
            ).tolist(),
            "labels": labels,
        }

    def get_permutation_importance(self, top_n=15):
        self._require_model()
        if self._raw_test is None:
            raise ValueError("A holdout dataset is required.")
        result = permutation_importance(
            self.model,
            self._raw_test,
            self.y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        return (
            pd.DataFrame(
                {
                    "feature": self._all_input_names,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

    def get_misclassified_samples(self, max_samples=20):
        self.get_metrics()
        if self.problem_type != "classification":
            raise ValueError("Classification only.")
        rows = self.df.loc[
            self._test_indices[self.misclassified_indices[:max_samples]]
        ].copy()
        if len(rows):
            positions = self.misclassified_indices[:max_samples]
            rows["Actual"] = self.target_encoder.inverse_transform(
                self.y_test[positions]
            )
            rows["Predicted"] = self.target_encoder.inverse_transform(
                self.y_pred[positions]
            )
        return rows

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
            n: float(v) for n, v in zip(self.feature_names, values)
        }

    def _schema_for(self, names):
        schema = []
        for name in names:
            col = self._raw_train[name]
            if name in self._numeric_columns:
                finite = col.dropna()
                schema.append(
                    {
                        "name": name,
                        "type": "numeric",
                        "min": float(finite.min()) if len(finite) else 0.0,
                        "max": float(finite.max()) if len(finite) else 0.0,
                        "mean": float(finite.mean()) if len(finite) else 0.0,
                    }
                )
            else:
                schema.append(
                    {
                        "name": name,
                        "type": "categorical",
                        "options": col.dropna().unique().tolist()[:100],
                    }
                )
        return schema

    def get_input_form_schema(self):
        self._require_model()
        return copy.deepcopy(self._model_schema)

    def predict(self, input_dict):
        self._require_model()
        missing = [
            item["name"]
            for item in self._model_schema
            if item["name"] not in input_dict
        ]
        if missing:
            raise ValueError("Missing prediction fields: " + ", ".join(missing))
        row = {}
        for item in self._model_schema:
            name = item["name"]
            value = input_dict[name]
            if item["type"] == "numeric":
                try:
                    value = np.nan if pd.isna(value) or value == "" else float(value)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"{name}: enter a number or leave blank."
                    ) from None
                if np.isinf(value):
                    raise ValueError(f"{name}: infinity is not supported.")
            else:
                value = np.nan if pd.isna(value) or value == "" else str(value)
            row[name] = value
        X = pd.DataFrame([row])
        raw = self.model.predict(X)[0]
        label = (
            self.target_encoder.inverse_transform([int(raw)])[0]
            if self.target_encoder is not None
            else raw
        )
        result = {
            "prediction": label.item() if isinstance(label, np.generic) else label,
            "raw": raw.item() if isinstance(raw, np.generic) else raw,
        }
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            labels = self.target_encoder.inverse_transform(
                self.model.classes_.astype(int)
            )
            result["confidence"] = float(proba.max())
            result["probabilities"] = {str(c): float(p) for c, p in zip(labels, proba)}
        return result

    def predict_from_csv(self, file_path, output_path=None):
        self._require_model()
        frame = pd.read_csv(file_path)
        rows = []
        missing = {item["name"] for item in self._model_schema} - set(frame.columns)
        if missing:
            raise ValueError(
                "Batch file is missing columns: " + ", ".join(sorted(missing))
            )
        for idx, row in frame.iterrows():
            try:
                result = self.predict(row.to_dict())
                rows.append(
                    {
                        "sample_id": idx,
                        "prediction": result["prediction"],
                        "confidence": result.get("confidence"),
                    }
                )
            except ValueError as error:
                rows.append(
                    {
                        "sample_id": idx,
                        "prediction": None,
                        "confidence": None,
                        "error": str(error),
                    }
                )
        result = pd.DataFrame(rows)
        if output_path:
            with atomic_target(output_path) as temporary:
                result.to_csv(temporary, index=False)
        return result

    def save_pipeline(self, file_path):
        self._require_model()
        package = {
            "format": "datayab-v2",
            "versions": runtime_versions(),
            "model": self.model,
            "model_name": self.model_name,
            "target_col": self.target_col,
            "problem_type": self.problem_type,
            "target_encoder": self.target_encoder,
            "schema": self._model_schema,
            "input_names": self._active_input_names,
            "feature_names": self.feature_names,
        }
        with atomic_target(file_path) as temporary:
            joblib.dump(package, temporary)
        return {"success": True, "path": str(file_path)}

    def load_pipeline(self, file_path):
        package = joblib.load(file_path)
        if not isinstance(package, dict) or package.get("format") != "datayab-v2":
            raise ValueError(
                "Legacy/incomplete model: retrain and export with this DataYab version."
            )
        if package.get("versions") != runtime_versions():
            raise ValueError(
                "Model dependency versions differ. Use its original environment or retrain."
            )
        if (
            not {
                "model_name",
                "target_col",
                "problem_type",
                "target_encoder",
                "input_names",
                "feature_names",
            }.issubset(package)
            or not package.get("schema")
            or not hasattr(package.get("model"), "predict")
        ):
            raise ValueError("Incomplete prediction package.")
        self._reset_learning()
        self.model = package["model"]
        self.model_name = package["model_name"]
        self.target_col = package["target_col"]
        self.problem_type = package["problem_type"]
        self.target_encoder = package["target_encoder"]
        self._model_schema = package["schema"]
        self._active_input_names = package["input_names"]
        self.feature_names = package["feature_names"]
        self.trained_models = {
            self.model_name: {
                "model": self.model,
                "feature_names": self.feature_names,
                "input_names": self._active_input_names,
                "schema": self._model_schema,
            }
        }
        self.available_models = [self.model_name]
        return {"success": True, "model": self.model_name}

    def _curve(self, kind):
        self.get_metrics()
        from matplotlib import pyplot as plt
        from .core import _fig_to_base64

        fig, ax = plt.subplots(figsize=(7, 4))
        if kind == "residuals":
            if self.problem_type != "regression":
                plt.close(fig)
                raise ValueError("Residuals require regression.")
            ax.scatter(self.y_pred, self.y_test - self.y_pred, alpha=0.6)
            ax.axhline(0, color="gray", linestyle="--")
            ax.set(xlabel="Prediction", ylabel="Residual", title="Holdout residuals")
        else:
            if (
                self.problem_type != "classification"
                or self.y_proba is None
                or self.y_proba.shape[1] != 2
            ):
                plt.close(fig)
                raise ValueError(
                    "This curve requires binary classification with probability estimates."
                )
            if kind == "roc":
                x, y, _ = roc_curve(self.y_test, self.y_proba[:, 1])
                ax.plot(x, y)
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set(
                    xlabel="False positive rate",
                    ylabel="True positive rate",
                    title="Holdout ROC",
                )
            else:
                precision, recall, _ = precision_recall_curve(
                    self.y_test, self.y_proba[:, 1]
                )
                ax.plot(recall, precision)
                ax.set(
                    xlabel="Recall",
                    ylabel="Precision",
                    title="Holdout precision-recall",
                )
        fig.tight_layout()
        return _fig_to_base64(fig)

    def plot_roc_curve(self):
        return self._curve("roc")

    def plot_precision_recall_curve(self):
        return self._curve("pr")

    def plot_residuals(self):
        return self._curve("residuals")

    def export_report(self, file_path):
        self._require_data()
        title = html.escape(Path(self.file_path).name or "Dataset")
        rows = [
            f"<h1>DataYab report: {title}</h1>",
            f"<p>{len(self.df):,} rows / {len(self.df.columns)} columns</p>",
        ]
        rows.append(self.df.describe(include="all").transpose().to_html(escape=True))
        rows.append(
            "<h2>Dataset observations</h2><ul>"
            + "".join(
                "<li>" + html.escape(str(item)) + "</li>"
                for item in self.generate_observations()
            )
            + "</ul>"
        )
        if self.model is not None and self._raw_test is not None:
            rows.append(
                "<h2>Reserved holdout evaluation</h2><pre>"
                + html.escape(str(self.get_metrics()))
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
