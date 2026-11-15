"""Data cleaning operations module."""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


class DataCleaner:
    """Handles all data cleaning operations."""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def _require_data(self):
        if self.data_loader.df is None:
            raise ValueError("No dataset loaded.")

    def get_missing_report(self) -> pd.DataFrame:
        self._require_data()
        miss = self.data_loader.df.isnull().sum()
        pct = (miss / len(self.data_loader.df) * 100).round(2)
        report = pd.DataFrame(
            {"Column": miss.index, "Missing": miss.values, "Percent": pct.values}
        )
        return (
            report[report["Missing"] > 0]
            .sort_values("Missing", ascending=False)
            .reset_index(drop=True)
        )

    def handle_missing(self, strategy: str = "mean", columns: Optional[List[str]] = None) -> dict:
        self._require_data()
        target_col = getattr(self, 'target_col', None)
        cols = columns or [c for c in self.data_loader.df.columns if c != target_col]
        before = self.data_loader.df.isnull().sum().sum()

        if strategy == "drop_rows":
            self.data_loader.df.dropna(subset=cols, inplace=True)
        elif strategy == "drop_cols":
            thresh = int(len(self.data_loader.df) * 0.5) + 1
            dropped = [c for c in cols if self.data_loader.df[c].notna().sum() < thresh]
            self.data_loader.df.drop(columns=dropped, inplace=True)
        elif strategy == "mean":
            for c in cols:
                if pd.api.types.is_numeric_dtype(self.data_loader.df[c]):
                    self.data_loader.df[c] = self.data_loader.df[c].fillna(self.data_loader.df[c].mean())
        elif strategy == "median":
            for c in cols:
                if pd.api.types.is_numeric_dtype(self.data_loader.df[c]):
                    self.data_loader.df[c] = self.data_loader.df[c].fillna(self.data_loader.df[c].median())
        elif strategy == "mode":
            for c in cols:
                mode_val = self.data_loader.df[c].mode()
                if not mode_val.empty:
                    self.data_loader.df[c] = self.data_loader.df[c].fillna(mode_val[0])
        elif strategy == "ffill":
            self.data_loader.df[cols] = self.data_loader.df[cols].ffill()
        elif strategy == "bfill":
            self.data_loader.df[cols] = self.data_loader.df[cols].bfill()
        elif strategy == "zero":
            for c in cols:
                if pd.api.types.is_numeric_dtype(self.data_loader.df[c]):
                    self.data_loader.df[c] = self.data_loader.df[c].fillna(0)
        elif strategy == "unknown":
            for c in cols:
                if self.data_loader.df[c].dtype == object:
                    self.data_loader.df[c] = self.data_loader.df[c].fillna("Unknown")
        elif strategy == "knn":
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.data_loader.df[c])]
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
                self.data_loader.df[numeric_cols] = imputer.fit_transform(self.data_loader.df[numeric_cols])
        elif strategy == "iterative":
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.data_loader.df[c])]
            if numeric_cols:
                imputer = IterativeImputer(random_state=42, max_iter=10, keep_empty_features=True)
                self.data_loader.df[numeric_cols] = imputer.fit_transform(self.data_loader.df[numeric_cols])
        else:
            return {"success": False, "message": f"Unknown strategy: {strategy}"}

        after = self.data_loader.df.isnull().sum().sum()
        self.data_loader._log(f"Missing values: {before} → {after} (strategy: {strategy})")
        return {
            "success": True,
            "message": f"Handled missing: {before - after} values filled/dropped",
            "remaining": after,
        }

    def remove_duplicates(self) -> dict:
        self._require_data()
        before = len(self.data_loader.df)
        self.data_loader.df.drop_duplicates(inplace=True)
        self.data_loader.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.data_loader.df)
        self.data_loader._log(f"Removed {removed} duplicate rows")
        return {
            "success": True,
            "removed": removed,
            "message": f"Removed {removed} duplicate rows",
        }

    def encode_categorical(self, columns: Optional[List[str]] = None, method: str = "label") -> dict:
        self._require_data()
        if method not in {'label', 'onehot'}:
            return {'success': False, 'message': f'Unknown categorical encoding method: {method}'}

        target_col = getattr(self, 'target_col', None)
        if target_col:
            cols = columns or [
                c for c in self.data_loader.df.select_dtypes(include=["object", "category"]).columns
                if c != target_col
            ]
        else:
            cols = columns or list(self.data_loader.df.select_dtypes(include=["object", "category"]).columns)

        encoded = []
        if method == "label":
            for c in cols:
                le = LabelEncoder()
                self.data_loader.df[c] = le.fit_transform(self.data_loader.df[c].astype(str))
                self.label_encoders[c] = le
                encoded.append(c)
        elif method == "onehot":
            self.data_loader.df = pd.get_dummies(self.data_loader.df, columns=cols, drop_first=True)
            encoded = cols

        self.data_loader._log(f"Encoded {len(encoded)} categorical columns ({method})")
        return {
            "success": True,
            "encoded_columns": encoded,
            "message": f"Encoded: {encoded}",
        }

    def remove_constant_columns(self) -> dict:
        self._require_data()
        target_col = getattr(self, 'target_col', None)
        constant = [
            c for c in self.data_loader.df.columns
            if c != target_col and self.data_loader.df[c].nunique() <= 1
        ]
        self.data_loader.df.drop(columns=constant, inplace=True)
        self.data_loader._log(f"Removed {len(constant)} constant columns: {constant}")
        return {"success": True, "removed": constant, "message": f"Removed: {constant}"}

    def remove_high_cardinality(self, threshold: int = 50) -> dict:
        self._require_data()
        target_col = getattr(self, 'target_col', None)
        high = [
            c for c in self.data_loader.df.select_dtypes(include="object").columns
            if self.data_loader.df[c].nunique() > threshold and c != target_col
        ]
        self.data_loader.df.drop(columns=high, inplace=True)
        self.data_loader._log(f"Removed high-cardinality columns: {high}")
        return {"success": True, "removed": high}

    def detect_outliers(self, column: str) -> dict:
        self._require_data()
        if column not in self.data_loader.df.columns:
            return {"success": False, "message": f"Column '{column}' not found"}
        if not pd.api.types.is_numeric_dtype(self.data_loader.df[column]):
            return {"success": False, "message": f"Column '{column}' is not numeric"}

        Q1 = self.data_loader.df[column].quantile(0.25)
        Q3 = self.data_loader.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = self.data_loader.df[(self.data_loader.df[column] < lower) | (self.data_loader.df[column] > upper)]

        return {
            "success": True,
            "column": column,
            "count": len(outliers),
            "pct": round(len(outliers) / len(self.data_loader.df) * 100, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_indices": list(outliers.index),
        }

    def cap_outliers(self, column: str) -> dict:
        self._require_data()
        info = self.detect_outliers(column)
        if not info["success"]:
            return info
        self.data_loader.df[column] = self.data_loader.df[column].clip(
            lower=info["lower_bound"], upper=info["upper_bound"]
        )
        self.data_loader._log(f"Capped outliers in '{column}'")
        return {
            "success": True,
            "message": f"Capped {info['count']} outliers in '{column}'",
        }

    def detect_all_outliers(self) -> pd.DataFrame:
        self._require_data()
        results = []
        for c in self.data_loader.df.select_dtypes(include=np.number).columns:
            info = self.detect_outliers(c)
            if info["success"]:
                results.append({
                    "Column": c,
                    "Outliers": info["count"],
                    "Percent": info["pct"],
                    "Lower": info["lower_bound"],
                    "Upper": info["upper_bound"],
                })
        return pd.DataFrame(results).sort_values("Outliers", ascending=False)

    def detect_outliers_isolation_forest(self, contamination: float = 0.1) -> dict:
        self._require_data()
        numeric_cols = list(self.data_loader.df.select_dtypes(include=np.number).columns)
        if not numeric_cols:
            return {"success": False, "message": "No numeric columns found"}

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.data_loader.df[numeric_cols])
        n_outliers = (outlier_labels == -1).sum()
        outlier_indices = np.where(outlier_labels == -1)[0].tolist()

        return {
            "success": True,
            "method": "Isolation Forest",
            "count": n_outliers,
            "percent": round(n_outliers / len(self.data_loader.df) * 100, 2),
            "outlier_indices": outlier_indices,
            "contamination": contamination,
        }

    def remove_outliers_isolation_forest(self, contamination: float = 0.1) -> dict:
        self._require_data()
        info = self.detect_outliers_isolation_forest(contamination)
        if not info["success"]:
            return info

        numeric_cols = list(self.data_loader.df.select_dtypes(include=np.number).columns)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.data_loader.df[numeric_cols])
        self.data_loader.df = self.data_loader.df[outlier_labels != -1].reset_index(drop=True)
        self.data_loader._log(f"Removed {info['count']} outliers using Isolation Forest")
        return {
            "success": True,
            "message": f"Removed {info['count']} outliers",
            "removed": info["count"],
        }

    def fix_dtypes(self) -> dict:
        self._require_data()
        target_col = getattr(self, 'target_col', None)
        converted = []
        for col in self.data_loader.df.select_dtypes(include=["object", "string"]).columns:
            if col == target_col:
                continue
            try:
                values = pd.to_numeric(
                    self.data_loader.df[col].replace(r"^\s*$", np.nan, regex=True), errors="raise"
                )
                self.data_loader.df[col] = values.replace([np.inf, -np.inf], np.nan)
                converted.append(col)
            except (ValueError, TypeError):
                pass
        return {"success": True, "converted": converted}

    def auto_clean(self, target_col: Optional[str] = None) -> dict:
        self._require_data()
        if target_col and target_col not in self.data_loader.df:
            raise ValueError("Target column not found.")
        if target_col:
            self.target_col = target_col

        duplicates = int(self.data_loader.df.duplicated().sum())
        converted = self.fix_dtypes()["converted"]
        steps = [
            f"Detected {duplicates:,} duplicate rows; kept them. Use Remove Duplicates explicitly if unwanted.",
            f"Converted numeric text columns: {converted}",
            "Missing values and outliers retained: training imputes features inside each fold; targets never imputed.",
            "High-cardinality and constant columns retained for explicit review.",
        ]
        self.data_loader._log(
            "Conservative auto-clean complete; duplicates kept, no target imputation or automatic outlier clipping."
        )
        return {
            "success": True,
            "steps": steps,
            "shape": self.data_loader.df.shape,
            "duplicates_detected": duplicates,
            "duplicates_removed": 0,
        }

    def get_dataset_quality(self) -> dict:
        self._require_data()
        df = self.data_loader.df
        info = self.data_loader.get_info()
        duplicates = int(df.duplicated().sum())
        constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

        correlation_pairs = []
        numeric = df.select_dtypes(include=np.number)
        if numeric.shape[1] >= 2:
            corr = numeric.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1))
            for (left, right), value in upper.stack().sort_values(ascending=False).head(10).items():
                correlation_pairs.append({
                    "feature_a": str(left),
                    "feature_b": str(right),
                    "correlation": round(float(value), 4),
                })

        class_balance = {}
        target_col = getattr(self, 'target_col', None)
        if target_col and target_col in df.columns:
            counts = df[target_col].value_counts(dropna=True)
            class_balance = {str(k): int(v) for k, v in counts.head(25).items()}

        missing_columns = [
            {"column": str(c), "missing": int(v), "percent": round(v / len(df) * 100, 2)}
            for c, v in df.isnull().sum().items()
            if v > 0
        ]

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "duplicate_rows": duplicates,
            "duplicate_percent": round(duplicates / len(df) * 100, 2) if len(df) else 0.0,
            "constant_columns": [str(c) for c in constant],
            "missing_total": int(info["total_missing"]),
            "missing_columns": missing_columns,
            "target": target_col,
            "class_balance": class_balance,
            "top_correlations": correlation_pairs,
            "dtypes": info["dtypes"],
        }
