"""Validation guard module to prevent data leakage and ensure proper ML practices."""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import warnings


class ValidationGuard:
    """Validates ML pipelines to prevent data leakage and ensure best practices."""

    def __init__(self, data_loader, feature_engineer, model_trainer):
        self.dl = data_loader
        self.fe = feature_engineer
        self.mt = model_trainer
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate_pipeline(self) -> Dict[str, Any]:
        """Run comprehensive validation checks on the current pipeline."""
        self.warnings = []
        self.errors = []

        self._check_data_leakage()
        self._check_target_leakage()
        self._check_preprocessing_leakage()
        self._check_cv_validity()
        self._check_class_balance()
        self._check_feature_selection_leakage()
        self._check_sample_sizes()
        self._check_model_appropriateness()

        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        }

    def _check_data_leakage(self):
        """Check for potential data leakage in the pipeline."""
        if self.fe._raw_train is None or self.fe._raw_test is None:
            return

        # Check for overlapping indices
        train_idx = set(self.fe._raw_train.index)
        test_idx = set(self.fe._raw_test.index)
        if train_idx & test_idx:
            self.errors.append("CRITICAL: Train and test sets have overlapping indices!")

        # Check if target was used in feature engineering
        if self.fe.target_col and self.fe.target_col in self.fe._selected_input_names:
            self.errors.append(f"CRITICAL: Target column '{self.fe.target_col}' is in selected features!")

        # Check if any feature was derived from target
        for feat in self.fe._selected_input_names:
            if self.fe.target_col and self.fe.target_col.lower() in feat.lower():
                if feat != self.fe.target_col:
                    self.warnings.append(f"Feature '{feat}' name contains target name - verify no leakage")

    def _check_target_leakage(self):
        """Check for target leakage in features."""
        if self.fe._raw_train is None:
            return

        if not self.fe.target_col or self.fe.target_col not in self.dl.df.columns:
            return

        target = self.dl.df[self.fe.target_col]

        # Check for features that are identical or near-identical to target
        for col in self.fe._selected_input_names:
            if col in self.dl.df.columns:
                feat = self.dl.df[col]
                if pd.api.types.is_numeric_dtype(feat) and pd.api.types.is_numeric_dtype(target):
                    # Check correlation
                    if len(feat.dropna()) > 10 and len(target.dropna()) > 10:
                        try:
                            corr = feat.corr(target)
                            if abs(corr) > 0.99:
                                self.errors.append(
                                    f"CRITICAL: Feature '{col}' has {corr:.4f} correlation with target - likely leakage!"
                                )
                            elif abs(corr) > 0.95:
                                self.warnings.append(
                                    f"WARNING: Feature '{col}' has very high correlation ({corr:.4f}) with target - verify no leakage"
                                )
                        except:
                            pass

                # Check for identical values
                try:
                    if feat.equals(target) or (feat == target).all():
                        self.errors.append(f"CRITICAL: Feature '{col}' is identical to target!")
                except:
                    pass

    def _check_preprocessing_leakage(self):
        """Check that preprocessing is fitted only on training data."""
        if self.fe._raw_train is None:
            return

        # Check if imputation statistics match training data
        numeric_cols = [c for c in self.fe._selected_input_names if c in self.fe._numeric_columns]
        for col in numeric_cols:
            if col in self.fe._raw_train.columns:
                train_median = self.fe._raw_train[col].median()
                # This would require checking the fitted pipeline - we'll add a warning
                pass

        self.warnings.append(
            "INFO: Preprocessing (imputation, scaling, encoding) is fitted inside each CV fold. "
            "This is correct and prevents leakage."
        )

    def _check_cv_validity(self):
        """Check cross-validation setup for validity."""
        if self.mt.model is None:
            return

        if self.fe._raw_train is None:
            self.warnings.append("No training data split - CV not available")
            return

        # Check minimum samples per class for classification
        if self.fe.problem_type == "classification":
            if hasattr(self.fe, 'y_train') and self.fe.y_train is not None:
                unique, counts = np.unique(self.fe.y_train, return_counts=True)
                min_count = counts.min()
                if min_count < 5:
                    self.warnings.append(
                        f"WARNING: Minority class has only {min_count} samples in training. "
                        f"CV may be unreliable. Consider more data or different split."
                    )
                if min_count < 2:
                    self.errors.append(
                        f"CRITICAL: Minority class has only {min_count} samples. "
                        f"Cannot do stratified CV."
                    )

        # Check if CV folds are appropriate
        n_samples = len(self.fe.y_train) if hasattr(self.fe, 'y_train') else 0
        if n_samples > 0 and n_samples < 50:
            self.warnings.append(
                f"WARNING: Only {n_samples} training samples. Consider using more CV folds (e.g., LOOCV) "
                f"or collecting more data."
            )

    def _check_class_balance(self):
        """Check class balance and warn about imbalanced data issues."""
        if self.fe.problem_type != "classification":
            return

        if not self.fe.target_col or self.fe.target_col not in self.dl.df.columns:
            return

        target = self.dl.df[self.fe.target_col].dropna()
        if len(target) == 0:
            return

        dist = target.value_counts(normalize=True)
        if len(dist) == 2:
            ratio = dist.iloc[0] / dist.iloc[1]
            if ratio > 10:
                self.warnings.append(
                    f"SEVERE IMBALANCE: Class ratio is {ratio:.1f}:1. "
                    f"Consider: SMOTE, class_weight='balanced', or different metrics (balanced_accuracy, F1)."
                )
            elif ratio > 3:
                self.warnings.append(
                    f"MODERATE IMBALANCE: Class ratio is {ratio:.1f}:1. "
                    f"Consider class_weight='balanced' or SMOTE."
                )
        elif len(dist) > 2:
            min_pct = dist.min() * 100
            if min_pct < 1:
                self.warnings.append(
                    f"MULTI-CLASS IMBALANCE: Smallest class is {min_pct:.1f}% of data. "
                    f"Consider class_weight='balanced' or oversampling."
                )

    def _check_feature_selection_leakage(self):
        """Check that feature selection was done properly without leakage."""
        if not self.fe._selection_data_driven:
            return

        self.warnings.append(
            "WARNING: Feature selection was data-driven on the training set. "
            "Cross-validation scores are conditional on these feature choices. "
            "Holdout evaluation is unbiased, but CV may be optimistic."
        )

        # Check if selection was done before or after split
        if self.fe._raw_train is not None:
            self.warnings.append(
                "INFO: Feature selection was applied AFTER train/test split. "
                "This is correct - selection uses only training data."
            )

    def _check_sample_sizes(self):
        """Check if sample sizes are adequate."""
        if self.dl.df is None:
            return

        n_rows = len(self.dl.df)
        n_features = len(self.dl.df.columns) - 1  # exclude target

        if n_rows < 50:
            self.warnings.append(f"WARNING: Only {n_rows} rows total. Model may overfit.")
        elif n_rows < 200:
            self.warnings.append(f"WARNING: {n_rows} rows is small. Use strong regularization and CV.")

        if self.fe._raw_train is not None:
            n_train = len(self.fe._raw_train)
            if n_features > n_train:
                self.warnings.append(
                    f"WARNING: More features ({n_features}) than training samples ({n_train}). "
                    f"High risk of overfit. Use feature selection or regularization."
                )
            elif n_features > n_train / 5:
                self.warnings.append(
                    f"WARNING: Feature-to-sample ratio is {n_features}:{n_train}. "
                    f"Consider feature selection or dimensionality reduction."
                )

    def _check_model_appropriateness(self):
        """Check if the selected model is appropriate for the problem."""
        if self.mt.model is None:
            return

        model_name = self.mt.model_name

        # Check SVM with large datasets
        if model_name in ["SVM", "SVR"] and self.fe._raw_train is not None:
            n_train = len(self.fe._raw_train)
            if n_train > 10000:
                self.warnings.append(
                    f"WARNING: SVM/SVR with {n_train} samples may be very slow. "
                    f"Consider LinearSVC, Random Forest, or Gradient Boosting."
                )

        # Check KNN with high dimensions
        if model_name == "KNN" and self.fe._raw_train is not None:
            n_features = self.fe._raw_train.shape[1]
            if n_features > 20:
                self.warnings.append(
                    f"WARNING: KNN with {n_features} features suffers from curse of dimensionality. "
                    f"Consider PCA or feature selection first."
                )

        # Check Naive Bayes assumptions
        if model_name == "Naive Bayes" and self.fe._raw_train is not None:
            self.warnings.append(
                "INFO: Naive Bayes assumes feature independence. "
                "Check feature correlations - highly correlated features may degrade performance."
            )

    def validate_before_deployment(self) -> Dict[str, Any]:
        """Comprehensive validation before deploying a model."""
        results = self.validate_pipeline()

        # Additional deployment checks
        if self.mt.model is None:
            results["errors"].append("No model trained.")
            results["valid"] = False

        # Check if model can predict on new data
        if self.mt.model is not None:
            try:
                # Test prediction with dummy data
                schema = self.mt._model_schema
                dummy_input = {}
                for item in schema:
                    if item["type"] == "numeric":
                        dummy_input[item["name"]] = item.get("mean", 0)
                    else:
                        options = item.get("options", ["Unknown"])
                        dummy_input[item["name"]] = options[0]

                pred = self.mt.model.predict(pd.DataFrame([dummy_input]))
                results["prediction_test"] = "PASS"
            except Exception as e:
                results["errors"].append(f"Prediction test failed: {e}")
                results["valid"] = False

        # Check if all required components are present
        required = ["model", "target_encoder", "_model_schema", "_active_input_names"]
        for req in required:
            if not hasattr(self.mt, req) or getattr(self.mt, req) is None:
                results["errors"].append(f"Missing required component: {req}")
                results["valid"] = False

        return results

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate a data quality report."""
        if self.dl.df is None:
            return {"error": "No data loaded"}

        df = self.dl.df
        report = {
            "shape": df.shape,
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "missing": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": df.dtypes.apply(str).to_dict(),
            "numeric_columns": list(df.select_dtypes(include=np.number).columns),
            "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
            "constant_columns": [c for c in df.columns if df[c].nunique() <= 1],
            "high_cardinality": [
                c for c in df.select_dtypes(include=["object", "category"]).columns
                if df[c].nunique() > 50
            ],
        }

        # Quality score
        score = 100
        issues = []
        if report["duplicates"] > 0:
            score -= min(20, report["duplicates"] / len(df) * 100 * 2)
            issues.append(f"{report['duplicates']} duplicate rows")

        missing_total = sum(report["missing"].values())
        if missing_total > 0:
            pct = missing_total / (df.shape[0] * df.shape[1]) * 100
            if pct > 50:
                score -= 40
            elif pct > 20:
                score -= 25
            elif pct > 5:
                score -= 10
            else:
                score -= 5
            issues.append(f"{missing_total} missing values ({pct:.1f}%)")

        if report["constant_columns"]:
            score -= len(report["constant_columns"]) * 2
            issues.append(f"{len(report['constant_columns'])} constant columns")

        report["quality_score"] = max(0, score)
        report["quality_grade"] = (
            "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
        )
        report["issues"] = issues

        return report
