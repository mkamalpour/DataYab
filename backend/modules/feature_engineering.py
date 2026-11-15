"""Feature engineering module (selection, extraction, transformation)."""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, RFE
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.inspection import permutation_importance


class FeatureEngineer:
    """Handles feature selection, extraction, and preprocessing pipelines."""

    def __init__(self, data_loader):
        self.data_loader = data_loader

        # State for training
        self._raw_train = None
        self._raw_test = None
        self.y_train = None
        self.y_test = None
        self._numeric_columns = []
        self._all_input_names = []
        self._selected_input_names = []
        self._extraction_method = None
        self._extraction_components = None
        self._scale = True
        self._scaler_type = "standard"
        self._selection_data_driven = False
        self.feature_names = []
        self.target_encoder = None
        self.problem_type = ""
        self.target_col = ""
        self._split_signature = None
        self.X_train = None
        self.X_test = None
        self._pca_components = None

    def _require_split(self):
        if self._raw_train is None:
            raise ValueError("Split data first using split_data().")

    def _canonical_frame(self, frame: pd.DataFrame, numeric: List[str]) -> pd.DataFrame:
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
                    .map(lambda v: np.nan if pd.isna(v) else str(v))
                    .astype(object)
                )
        return frame

    def split_data(
        self,
        target: str,
        test_size: float = 0.2,
        scale: bool = True,
        scaler_type: str = "standard",
        problem_type: str = "auto"
    ) -> dict:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        if target not in self.data_loader.df:
            raise ValueError(f"Target column {target!r} does not exist.")
        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            raise ValueError("Test fraction must be between zero and one.")
        if scaler_type not in ("standard", "minmax"):
            raise ValueError("Scaler must be standard or minmax.")
        if problem_type not in ("auto", "classification", "regression"):
            raise ValueError("Unknown problem type.")

        signature = (target, float(test_size), bool(scale), scaler_type, problem_type)
        if signature == getattr(self, '_split_signature', None) and self._raw_train is not None:
            return {
                "success": True,
                "problem_type": self.problem_type,
                "train_size": len(self.y_train),
                "test_size": len(self.y_test),
                "features": len(self.feature_names),
                "feature_names": self.feature_names,
            }

        df = self.data_loader.df.loc[self.data_loader.df[target].notna()].copy()
        if len(df) < 10 or len(df.columns) < 2:
            raise ValueError("Training needs ten labelled rows and at least one feature.")

        # Auto-detect problem type
        col = df[target].dropna()
        if problem_type == "auto":
            if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                problem = "classification"
            elif col.nunique() <= 20 and np.equal(col, np.floor(col)).all():
                problem = "classification"
            else:
                problem = "regression"
        else:
            problem = problem_type

        X = df.drop(columns=[target])
        numeric = [col for col in X if pd.api.types.is_numeric_dtype(X[col])]
        X = self._canonical_frame(X, numeric)
        yraw = df[target]

        if problem == "classification":
            values = yraw.astype(str)
            counts = values.value_counts()
            if len(counts) < 2 or counts.min() < 2:
                raise ValueError("Each of at least two classes needs two labelled rows.")
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

        self.target_encoder = encoder
        self.target_col = target
        self.problem_type = problem
        self._raw_train = X.iloc[train_idx].copy()
        self._raw_test = X.iloc[test_idx].copy()
        self._train_indices = self._raw_train.index.to_numpy()
        self._test_indices = self._raw_test.index.to_numpy()
        self._numeric_columns = numeric
        self._all_input_names = list(X.columns)
        self._selected_input_names = list(X.columns)
        self._extraction_method = None
        self._extraction_components = None
        self._selection_data_driven = False
        self._scale = bool(scale)
        self._scaler_type = scaler_type
        self.y_train = yt
        self.y_test = yv
        self._split_signature = signature

        self._refresh_feature_view()

        return {
            "success": True,
            "problem_type": problem,
            "train_size": len(yt),
            "test_size": len(yv),
            "features": len(self.feature_names),
            "feature_names": self.feature_names,
        }

    def _preprocessor(self, names: Optional[List[str]] = None):
        names = self._selected_input_names if names is None else names
        transformers = []
        for i, col in enumerate(names):
            if col in self._numeric_columns:
                steps = [
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
                ]
                if self._scale:
                    steps.append((
                        "scale",
                        StandardScaler() if self._scaler_type == "standard" else MinMaxScaler()
                    ))
            else:
                steps = [
                    ("impute", SimpleImputer(
                        strategy="constant", fill_value="Unknown", keep_empty_features=True
                    )),
                    ("encode", OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )),
                ]
            transformers.append((f"column_{i}", Pipeline(steps), [col]))
        return ColumnTransformer(
            transformers, remainder="drop", verbose_feature_names_out=False
        )

    def _extraction_transformer(self):
        method = self._extraction_method
        if method is None:
            return None
        components = self._extraction_components
        if method == "pca":
            return PCA(n_components=components, random_state=42)
        if method == "svd":
            return TruncatedSVD(n_components=components, random_state=42)
        if method == "lda":
            return LinearDiscriminantAnalysis(n_components=components)
        raise ValueError(f"Unknown feature extraction method: {method}")

    def _refresh_feature_view(self):
        prep = self._preprocessor()
        self._raw_train_transformed = prep.fit_transform(self._raw_train)
        self._raw_test_transformed = prep.transform(self._raw_test)
        self.feature_names = list(self._selected_input_names)

        transformer = self._extraction_transformer()
        if transformer is None:
            self.X_train = self._raw_train_transformed
            self.X_test = self._raw_test_transformed
            return None

        if isinstance(transformer, LinearDiscriminantAnalysis):
            self.X_train = transformer.fit_transform(self._raw_train_transformed, self.y_train)
        else:
            self.X_train = transformer.fit_transform(self._raw_train_transformed)
        self.X_test = transformer.transform(self._raw_test_transformed)

        prefix = {"pca": "PC", "svd": "SVD", "lda": "LD"}[self._extraction_method]
        self.feature_names = [f"{prefix}{i + 1}" for i in range(self.X_train.shape[1])]
        return transformer

    def _training_pipeline(self, estimator, use_smote: bool = False):
        steps = [("prepare", self._preprocessor())]
        transformer = self._extraction_transformer()
        if transformer is not None:
            steps.append(("extract", transformer))
        if use_smote:
            if self.problem_type != "classification":
                raise ValueError("SMOTE is only for classification.")
            if any(
                col not in self._numeric_columns for col in self._selected_input_names
            ):
                raise ValueError(
                    "SMOTE requires numeric features; use balanced class weights for mixed data."
                )
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as SamplingPipeline
            steps.extend([
                ("resample", SMOTE(random_state=42, k_neighbors=1)),
                ("model", estimator),
            ])
            return SamplingPipeline(steps)
        steps.append(("model", estimator))
        return Pipeline(steps)

    # ==================== FEATURE SELECTION ====================

    def _selection_matrix(self):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        names = list(self._all_input_names)
        prep = self._preprocessor(names)
        return prep.fit_transform(self._raw_train[names]), names

    def _feature_score_series(self, method: str):
        method = str(method).lower()
        X, names = self._selection_matrix()

        if method == "rf":
            model = (
                RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
                if self.problem_type == "classification"
                else RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1)
            )
            model.fit(X, self.y_train)
            scores = model.feature_importances_
        elif method == "mutual_info":
            function = (
                mutual_info_classif
                if self.problem_type == "classification"
                else mutual_info_regression
            )
            scores = function(
                X, self.y_train,
                random_state=42,
                n_neighbors=min(3, len(self.y_train) - 1),
            )
        elif method == "anova":
            function = (
                f_classif if self.problem_type == "classification" else f_regression
            )
            scores = np.zeros(X.shape[1])
            variable = np.flatnonzero(np.var(X, axis=0) > 0)
            if len(variable):
                values = function(X[:, variable], self.y_train)[0]
                scores[variable] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        elif method == "correlation":
            target = np.asarray(self.y_train, dtype=float)
            scores = []
            for index in range(X.shape[1]):
                column = np.asarray(X[:, index], dtype=float)
                scores.append(
                    abs(np.corrcoef(column, target)[0, 1])
                    if column.std() > 0 and target.std() > 0 else 0.0
                )
            scores = np.nan_to_num(np.asarray(scores), nan=0.0)
        else:
            raise ValueError("Ranking methods: rf, mutual_info, anova, correlation.")
        return pd.Series(scores, index=names).sort_values(ascending=False)

    def get_feature_scores(self, method: str = "rf", top_n: int = 20):
        return self._feature_score_series(method).head(top_n)

    def _validate_input_names(self, names: List[str]):
        if (
            not names
            or len(set(names)) != len(names)
            or any(name not in self._all_input_names for name in names)
        ):
            raise ValueError(
                "Choose unique original input columns, not extracted component names."
            )

    def _set_selected_names(self, names: List[str]):
        self._selected_input_names = list(names)
        self._selection_data_driven = True
        self._extraction_method = None
        self._extraction_components = None
        self._refresh_feature_view()

    def apply_selection(self, method: str = "rf", k: int = 10, features: Optional[List[str]] = None):
        if self._raw_train is None:
            raise ValueError("Split data first.")

        method = str(method).lower()

        if method == "all":
            return self.restore_selection()
        if method == "manual":
            if not isinstance(features, list) or not features:
                raise ValueError("Provide a list of feature names.")
            names = [str(name) for name in features]
            self._validate_input_names(names)
        elif method == "variance":
            names = [
                column for column in self._all_input_names
                if self._raw_train[column].nunique(dropna=True) > 1
            ]
            if not names:
                raise ValueError("Every input column is constant.")
        elif method in ("rf", "mutual_info", "anova", "correlation"):
            if type(k) is not int or not 1 <= k <= len(self._all_input_names):
                raise ValueError(f"Keep between 1 and {len(self._all_input_names)} features.")
            names = list(self._feature_score_series(method).head(k).index)
        elif method == "rfe":
            if type(k) is not int or not 1 <= k <= len(self._all_input_names):
                raise ValueError(f"Keep between 1 and {len(self._all_input_names)} features.")
            X, names = self._selection_matrix()
            estimator = (
                LogisticRegression(max_iter=2000, random_state=42)
                if self.problem_type == "classification"
                else Ridge(alpha=1.0)
            )
            selector = RFE(estimator, n_features_to_select=k).fit(X, self.y_train)
            names = [names[i] for i in np.flatnonzero(selector.support_)]
        else:
            raise ValueError(
                "Selection methods: rf, mutual_info, anova, correlation, rfe, variance, manual."
            )

        self._set_selected_names(names)
        dropped = [name for name in self._all_input_names if name not in names]
        return {
            "success": True,
            "method": method,
            "selected": list(self._selected_input_names),
            "dropped": dropped,
            "total": len(self._selected_input_names),
        }

    def select_features_manual(self, feature_list: List[str]):
        return self.apply_selection("manual", features=feature_list)

    def restore_selection(self):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        self._selected_input_names = list(self._all_input_names)
        self._selection_data_driven = False
        self._extraction_method = None
        self._extraction_components = None
        self._refresh_feature_view()
        return {
            "success": True,
            "selected": list(self._selected_input_names),
            "total": len(self._selected_input_names),
        }

    # ==================== FEATURE EXTRACTION ====================

    def apply_extraction(self, method: str = "pca", n_components: int = 2):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        method = str(method).lower()
        if method not in ("pca", "svd", "lda"):
            raise ValueError("Extraction methods: pca, svd, lda.")
        if type(n_components) is not int or n_components < 1:
            raise ValueError("Components must be a positive whole number.")

        limit = min(len(self._selected_input_names), len(self.y_train))
        if method == "lda":
            if self.problem_type != "classification":
                raise ValueError("LDA extraction is only for classification.")
            limit = min(limit, len(np.unique(self.y_train)) - 1)
        if limit < 1 or n_components > limit:
            raise ValueError(f"{method.upper()} components must be between 1 and {max(limit, 0)}.")

        self._extraction_method = method
        self._extraction_components = n_components
        transformer = self._refresh_feature_view()

        info = {
            "success": True,
            "method": method,
            "n_components": len(self.feature_names),
            "features": list(self.feature_names),
        }
        if hasattr(transformer, "explained_variance_ratio_"):
            ratio = np.asarray(transformer.explained_variance_ratio_)
            info["explained_variance_ratio"] = [float(v) for v in ratio]
            info["cumulative_variance"] = [float(v) for v in ratio.cumsum()]
            info["total_variance"] = float(ratio.sum())
        return info

    def restore_extraction(self):
        if self._raw_train is None:
            raise ValueError("Split data first.")
        self._extraction_method = None
        self._extraction_components = None
        self._refresh_feature_view()
        return {"success": True, "features": list(self.feature_names)}

    def get_feature_status(self):
        return {
            "total": len(self._all_input_names),
            "selected": len(self._selected_input_names or []),
            "selected_features": list(self._selected_input_names or []),
            "extraction": self._extraction_method,
            "extraction_components": self._extraction_components,
            "feature_names": list(self.feature_names or []),
            "pca": self._extraction_method == "pca",
        }

    def get_fast_feature_importance(self, top_n: int = 20):
        if not hasattr(self, 'X_train') or self.X_train is None:
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

    def get_mutual_information_top(self, k: int = 10):
        if not hasattr(self, 'X_train') or self.X_train is None:
            raise ValueError("Split data first.")
        function = (
            mutual_info_classif
            if self.problem_type == "classification"
            else mutual_info_regression
        )
        scores = function(
            self.X_train, self.y_train,
            random_state=42,
            n_neighbors=min(3, len(self.y_train) - 1),
        )
        return (
            pd.Series(scores, index=self.feature_names)
            .sort_values(ascending=False)
            .head(k)
        )

    def get_permutation_importance(self, top_n: int = 15, n_repeats: int = 5):
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Train a model first.")
        if self._raw_test is None:
            raise ValueError("A holdout dataset is required.")
        result = permutation_importance(
            self.model, self._raw_test, self.y_test,
            n_repeats=n_repeats, random_state=42, n_jobs=1
        )
        return (
            pd.DataFrame({
                "feature": self._all_input_names,
                "importance": result.importances_mean,
                "std": result.importances_std,
            })
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
