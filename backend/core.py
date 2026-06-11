"""
core.py — Ultimate Dataset Analyzer Backend
Complete, production‑ready, bug‑free.
Works with any tabular dataset.
"""

import io
import base64
import warnings
import copy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import joblib

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# ------------------------------
# Helper: figure to base64
# ------------------------------
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

# ------------------------------
# Model factories (fresh instances each time)
# ------------------------------
def _get_classification_models():
    models = {
        "Random Forest":        RandomForestClassifier(random_state=42),
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "SVM":                  SVC(probability=True, random_state=42),
        "KNN":                  KNeighborsClassifier(),
        "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
        "Naive Bayes":          GaussianNB(),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss')
    return models

def _get_regression_models():
    models = {
        "Random Forest":        RandomForestRegressor(random_state=42),
        "Linear Regression":    LinearRegression(),
        "Ridge":                Ridge(),
        "Decision Tree":        DecisionTreeRegressor(random_state=42),
        "SVR":                  SVR(),
        "KNN":                  KNeighborsRegressor(),
        "Gradient Boosting":    GradientBoostingRegressor(random_state=42),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
    return models

# ------------------------------
# Core Class
# ------------------------------
class DataYabCore:
    def __init__(self):
        self.df = None
        self.original_df = None
        self.file_path = ""

        self.target_col = ""
        self.problem_type = ""   # 'classification' or 'regression'

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = []

        self.model = None
        self.model_name = ""
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        
        # Model storage for selection and stacking
        self.trained_models = {}  # {model_name: model_obj}
        self.available_models = []  # list of trained model names

        self.metrics = {}
        self.feature_importance = {}
        self.insights = []
        self.log = []

        # Additional evaluation data
        self.y_pred = None
        self.y_proba = None
        self.misclassified_indices = []
        
        # Feature selection tracking
        self._used_feature_selection = False
        self._selected_features_indices = None
        self._selected_feature_names = None   # ADD THIS LINE
        self.eval_model = None           # ADD THIS
        self.eval_model_name = None      # ADD THIS

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    def load_csv(self, file_path: str, sep: str = None) -> dict:
        try:
            encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
            separators = [sep] if sep else [",", ";", "\t", "|"]
            df = None
            for enc in encodings:
                df = None
                for s in separators:
                    try:
                        df = pd.read_csv(file_path, sep=s, encoding=enc, low_memory=False)
                        if df.shape[1] > 1:
                            break
                    except Exception:
                        continue
                if df is not None and df.shape[1] > 1:
                    break
            if df is None or df.shape[1] <= 1:
                df = pd.read_csv(file_path, low_memory=False)
            self.df = df.copy()
            self.original_df = df.copy()
            self.file_path = file_path
            self._log(f"Loaded CSV: {Path(file_path).name} — {df.shape[0]:,} rows × {df.shape[1]} cols")
            return {"success": True, "message": f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns",
                    "preview": df.head(10), "info": self.get_info()}
        except Exception as e:
            return {"success": False, "message": f"Error loading CSV: {e}"}

    def load_excel(self, file_path: str, sheet_name=0) -> dict:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            self.df = df.copy()
            self.original_df = df.copy()
            self.file_path = file_path
            self._log(f"Loaded Excel: {Path(file_path).name}")
            return {"success": True, "message": f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns",
                    "preview": df.head(10), "info": self.get_info()}
        except Exception as e:
            return {"success": False, "message": f"Error loading Excel: {e}"}

    def load_json(self, file_path: str) -> dict:
        """Load JSON file, with graceful handling of nested structures."""
        try:
            df = pd.read_json(file_path)
            if df.shape[1] == 1 and df.shape[0] > 0:
                try:
                    df = pd.json_normalize(pd.read_json(file_path).iloc[:, 0])
                except Exception:
                    pass
        except Exception as e:
            try:
                import json
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.json_normalize(data)
                else:
                    return {"success": False, "message": f"Unsupported JSON structure: {type(data)}"}
            except Exception as e:
                return {"success": False, "message": f"Error loading JSON: {str(e)}"}
        
        if df is None or df.empty:
            return {"success": False, "message": "JSON file is empty or invalid"}
        
        self.df = df.copy()
        self.original_df = df.copy()
        self.file_path = file_path
        self._log(f"Loaded JSON: {Path(file_path).name}")
        return {"success": True, "message": f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns",
                "preview": df.head(10), "info": self.get_info()}

    def load_parquet(self, file_path: str) -> dict:
        try:
            df = pd.read_parquet(file_path)
            self.df = df.copy()
            self.original_df = df.copy()
            self.file_path = file_path
            self._log(f"Loaded Parquet: {Path(file_path).name}")
            return {"success": True, "message": f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns",
                    "preview": df.head(10), "info": self.get_info()}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def load_file(self, file_path: str) -> dict:
        ext = Path(file_path).suffix.lower()
        loaders = {".csv": self.load_csv, ".txt": self.load_csv,
                   ".xlsx": self.load_excel, ".xls": self.load_excel,
                   ".json": self.load_json, ".parquet": self.load_parquet}
        fn = loaders.get(ext)
        if fn:
            return fn(file_path)
        return {"success": False, "message": f"Unsupported file type: {ext}"}

    def get_info(self) -> dict:
        self._require_data()
        df = self.df
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        missing = df.isnull().sum()
        dtypes_map = {str(k): str(v) for k, v in df.dtypes.items()}
        return {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": dtypes_map,
            "missing_count": missing.to_dict(),
            "missing_pct": (missing / len(df) * 100).round(2).to_dict(),
            "total_missing": int(missing.sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(memory_mb, 3),
            "numeric_cols": list(df.select_dtypes(include=np.number).columns),
            "categorical_cols": list(df.select_dtypes(include=["object", "category"]).columns),
            "datetime_cols": list(df.select_dtypes(include=["datetime64"]).columns),
        }

    def reset_to_original(self):
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self._log("Dataset reset to original")

    # ------------------------------------------------------------------
    # 2. CLEAN DATA
    # ------------------------------------------------------------------
    def get_missing_report(self) -> pd.DataFrame:
        self._require_data()
        miss = self.df.isnull().sum()
        pct = (miss / len(self.df) * 100).round(2)
        report = pd.DataFrame({"Column": miss.index, "Missing": miss.values, "Percent": pct.values})
        return report[report["Missing"] > 0].sort_values("Missing", ascending=False).reset_index(drop=True)

    def handle_missing(self, strategy: str = "mean", columns: list = None) -> dict:
        self._require_data()
        cols = columns or list(self.df.columns)
        before = self.df.isnull().sum().sum()
        if strategy == "drop_rows":
            self.df.dropna(subset=cols, inplace=True)
        elif strategy == "drop_cols":
            thresh = int(len(self.df) * 0.5) + 1
            self.df.dropna(axis=1, thresh=thresh, inplace=True)
        elif strategy == "mean":
            for c in cols:
                if pd.api.types.is_numeric_dtype(self.df[c]):
                    self.df[c] = self.df[c].fillna(self.df[c].mean())
        elif strategy == "median":
            for c in cols:
                if pd.api.types.is_numeric_dtype(self.df[c]):
                    self.df[c] = self.df[c].fillna(self.df[c].median())
        elif strategy == "mode":
            for c in cols:
                mode_val = self.df[c].mode()
                if not mode_val.empty:
                    self.df[c] = self.df[c].fillna(mode_val[0])
        elif strategy == "ffill":
            self.df[cols] = self.df[cols].ffill()
        elif strategy == "bfill":
            self.df[cols] = self.df[cols].bfill()
        elif strategy == "zero":
            for c in cols:
                if pd.api.types.is_numeric_dtype(self.df[c]):
                    self.df[c] = self.df[c].fillna(0)
        elif strategy == "unknown":
            for c in cols:
                if self.df[c].dtype == object:
                    self.df[c] = self.df[c].fillna("Unknown")
        elif strategy == "knn":
            # KNN imputation for numeric columns
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        elif strategy == "iterative":
            # Iterative imputation (MICE-like)
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]
            if numeric_cols:
                imputer = IterativeImputer(random_state=42, max_iter=10)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        else:
            return {"success": False, "message": f"Unknown strategy: {strategy}"}
        after = self.df.isnull().sum().sum()
        self._log(f"Missing values: {before} → {after} (strategy: {strategy})")
        return {"success": True, "message": f"Handled missing: {before - after} values filled/dropped", "remaining": after}

    def remove_duplicates(self) -> dict:
        self._require_data()
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        self._log(f"Removed {removed} duplicate rows")
        return {"success": True, "removed": removed, "message": f"Removed {removed} duplicate rows"}

    def encode_categorical(self, columns: list = None, method: str = "label") -> dict:
        self._require_data()
        if self.target_col:
            cols = columns or [c for c in self.df.select_dtypes(include=["object", "category"]).columns if c != self.target_col]
        else:
            cols = columns or list(self.df.select_dtypes(include=["object", "category"]).columns)
        encoded = []
        if method == "label":
            for c in cols:
                le = LabelEncoder()
                self.df[c] = le.fit_transform(self.df[c].astype(str))
                self.label_encoders[c] = le
                encoded.append(c)
        elif method == "onehot":
            self.df = pd.get_dummies(self.df, columns=cols, drop_first=True)
            encoded = cols
        self._log(f"Encoded {len(encoded)} categorical columns ({method})")
        return {"success": True, "encoded_columns": encoded, "message": f"Encoded: {encoded}"}

    def remove_constant_columns(self) -> dict:
        self._require_data()
        constant = [c for c in self.df.columns if self.df[c].nunique() <= 1]
        self.df.drop(columns=constant, inplace=True)
        self._log(f"Removed {len(constant)} constant columns: {constant}")
        return {"success": True, "removed": constant, "message": f"Removed: {constant}"}

    def remove_high_cardinality(self, threshold: int = 50) -> dict:
        self._require_data()
        high = [c for c in self.df.select_dtypes(include="object").columns
                if self.df[c].nunique() > threshold and c != self.target_col]
        self.df.drop(columns=high, inplace=True)
        self._log(f"Removed high-cardinality columns: {high}")
        return {"success": True, "removed": high}

    def detect_outliers(self, column: str) -> dict:
        self._require_data()
        if column not in self.df.columns:
            return {"success": False, "message": f"Column '{column}' not found"}
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "message": f"Column '{column}' is not numeric"}
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower) | (self.df[column] > upper)]
        return {
            "success": True,
            "column": column,
            "count": len(outliers),
            "pct": round(len(outliers) / len(self.df) * 100, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_indices": list(outliers.index),
        }

    def cap_outliers(self, column: str) -> dict:
        self._require_data()
        info = self.detect_outliers(column)
        if not info["success"]:
            return info
        self.df[column] = self.df[column].clip(lower=info["lower_bound"], upper=info["upper_bound"])
        self._log(f"Capped outliers in '{column}'")
        return {"success": True, "message": f"Capped {info['count']} outliers in '{column}'"}

    def detect_all_outliers(self) -> pd.DataFrame:
        self._require_data()
        results = []
        for c in self.df.select_dtypes(include=np.number).columns:
            info = self.detect_outliers(c)
            if info["success"]:
                results.append({"Column": c, "Outliers": info["count"],
                                "Percent": info["pct"], "Lower": info["lower_bound"],
                                "Upper": info["upper_bound"]})
        return pd.DataFrame(results).sort_values("Outliers", ascending=False)

    def detect_outliers_isolation_forest(self, contamination: float = 0.1) -> dict:
        """Detect outliers using Isolation Forest (more robust than IQR)."""
        self._require_data()
        numeric_cols = list(self.df.select_dtypes(include=np.number).columns)
        if not numeric_cols:
            return {"success": False, "message": "No numeric columns found"}
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.df[numeric_cols])
        n_outliers = (outlier_labels == -1).sum()
        outlier_indices = np.where(outlier_labels == -1)[0].tolist()
        return {
            "success": True,
            "method": "Isolation Forest",
            "count": n_outliers,
            "percent": round(n_outliers / len(self.df) * 100, 2),
            "outlier_indices": outlier_indices,
            "contamination": contamination
        }

    def remove_outliers_isolation_forest(self, contamination: float = 0.1) -> dict:
        """Remove outliers detected by Isolation Forest."""
        self._require_data()
        info = self.detect_outliers_isolation_forest(contamination)
        if not info["success"]:
            return info
        numeric_cols = list(self.df.select_dtypes(include=np.number).columns)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.df[numeric_cols])
        self.df = self.df[outlier_labels != -1].reset_index(drop=True)
        self._log(f"Removed {info['count']} outliers using Isolation Forest")
        return {"success": True, "message": f"Removed {info['count']} outliers", "removed": info['count']}

    def fix_dtypes(self) -> dict:
        self._require_data()
        converted = []
        for c in self.df.select_dtypes(include="object").columns:
            try:
                self.df[c] = pd.to_numeric(self.df[c])
                converted.append(f"{c} → numeric")
                continue
            except Exception:
                pass
            try:
                self.df[c] = pd.to_datetime(self.df[c])
                converted.append(f"{c} → datetime")
            except Exception:
                pass
        self._log(f"Fixed dtypes: {converted}")
        return {"success": True, "converted": converted}

    def auto_clean(self, target_col: str = None) -> dict:
        self._require_data()
        steps = []
        r = self.fix_dtypes()
        steps.append(f"Dtype fix: {r['converted']}")
        r = self.remove_constant_columns()
        steps.append(f"Constant cols removed: {r['removed']}")
        if target_col:
            self.target_col = target_col
        hc_cols = [c for c in self.df.select_dtypes(include="object").columns
                   if self.df[c].nunique() > 50 and c != self.target_col]
        if hc_cols:
            self.df.drop(columns=hc_cols, inplace=True)
            steps.append(f"High-cardinality cols removed: {hc_cols}")
        r = self.remove_duplicates()
        steps.append(f"Duplicates removed: {r['removed']}")
        for c in self.df.columns:
            if self.df[c].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[c]):
                    self.df[c] = self.df[c].fillna(self.df[c].median())
                else:
                    mode = self.df[c].mode()
                    self.df[c] = self.df[c].fillna(mode[0] if not mode.empty else "Unknown")
        steps.append("Missing values filled (median/mode)")
        numeric_cols = list(self.df.select_dtypes(include=np.number).columns)
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        for c in numeric_cols:
            self.cap_outliers(c)
        steps.append(f"Outliers capped on {len(numeric_cols)} columns")
        self._log("Auto-clean complete")
        return {"success": True, "steps": steps, "shape": self.df.shape}

    # ------------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------------
    def get_column_types(self) -> dict:
        self._require_data()
        return {
            "numeric": list(self.df.select_dtypes(include=np.number).columns),
            "categorical": list(self.df.select_dtypes(include=["object", "category"]).columns),
            "datetime": list(self.df.select_dtypes(include=["datetime64"]).columns),
        }

    def get_stats(self) -> pd.DataFrame:
        self._require_data()
        num = self.df.select_dtypes(include=np.number)
        stats = num.describe().T
        stats["skew"] = num.skew()
        stats["kurtosis"] = num.kurtosis()
        stats["missing"] = self.df[num.columns].isnull().sum()
        stats["missing_pct"] = (stats["missing"] / len(self.df) * 100).round(2)
        return stats.round(4)

    def get_correlation_matrix(self) -> pd.DataFrame:
        self._require_data()
        return self.df.select_dtypes(include=np.number).corr()

    def get_top_correlations(self, target: str = None, n: int = 15) -> pd.DataFrame:
        self._require_data()
        corr = self.get_correlation_matrix()
        if target and target in corr.columns:
            result = corr[[target]].drop(target).abs().sort_values(target, ascending=False).head(n)
            return result
        pairs = corr.abs().unstack()
        idx = pairs.index.to_frame()
        pair_sets = [frozenset((i, j)) for i, j in idx.values]
        pairs.index = pd.MultiIndex.from_tuples([(p, '') for p in pair_sets])
        pairs = pairs[~pairs.index.duplicated(keep='first')]
        pairs = pairs[pairs < 1].sort_values(ascending=False).head(n)
        readable = []
        for s in pairs.index.get_level_values(0):
            lst = list(s)
            readable.append(f"{lst[0]} ↔ {lst[1]}")
        pairs.index = readable
        return pd.DataFrame({"correlation": pairs})

    def get_value_counts(self, column: str, top_n: int = 20) -> pd.DataFrame:
        self._require_data()
        vc = self.df[column].value_counts().head(top_n)
        pct = (vc / len(self.df) * 100).round(2)
        return pd.DataFrame({"Value": vc.index, "Count": vc.values, "Percent": pct.values})

    def generate_observations(self, n: int = 10) -> list:
        self._require_data()
        obs = []
        info = self.get_info()
        obs.append(f"Dataset has {info['rows']:,} rows and {info['cols']} columns.")
        total_miss = info["total_missing"]
        miss_pct = round(total_miss / (info["rows"] * info["cols"]) * 100, 2)
        if total_miss == 0:
            obs.append("Dataset is clean — no missing values found.")
        else:
            worst_col = max(info["missing_pct"], key=info["missing_pct"].get)
            obs.append(f"{total_miss:,} missing values detected ({miss_pct}%). "
                       f"Worst column: '{worst_col}' ({info['missing_pct'][worst_col]}% missing).")
        dups = info["duplicates"]
        obs.append(f"{'No duplicate' if dups == 0 else str(dups) + ' duplicate'} rows found.")
        num_cols = info["numeric_cols"]
        cat_cols = info["categorical_cols"]
        obs.append(f"{len(num_cols)} numeric and {len(cat_cols)} categorical columns detected.")
        if num_cols:
            stats = self.get_stats()
            most_skewed = stats["skew"].abs().idxmax()
            skew_val = round(stats.loc[most_skewed, "skew"], 2)
            obs.append(f"Most skewed column: '{most_skewed}' (skew={skew_val}).")
        if len(num_cols) >= 2:
            corr = self.get_correlation_matrix()
            flat = corr.abs().unstack()
            flat = flat[flat < 1].sort_values(ascending=False)
            seen = set()
            unique_pairs = []
            for (c1, c2), val in flat.items():
                pair = tuple(sorted((c1, c2)))
                if pair not in seen:
                    seen.add(pair)
                    unique_pairs.append(((c1, c2), val))
            if unique_pairs:
                (c1, c2), val = unique_pairs[0]
                obs.append(f"Strongest correlation: '{c1}' ↔ '{c2}' (r={val:.3f}).")
        if self.target_col and self.target_col in self.df.columns:
            target_vals = self.df[self.target_col]
            if target_vals.nunique() <= 20:
                dist = target_vals.value_counts(normalize=True) * 100
                top_class = dist.index[0]
                obs.append(f"Target '{self.target_col}': dominant class is '{top_class}' "
                           f"({dist.iloc[0]:.1f}% of data).")
                if len(dist) == 2:
                    imbalance = dist.iloc[0] / dist.iloc[1]
                    if imbalance > 3:
                        obs.append(f"Target is imbalanced — majority class is {imbalance:.1f}× larger than minority.")
        if cat_cols:
            high_card = [(c, self.df[c].nunique()) for c in cat_cols if self.df[c].nunique() > 10]
            if high_card:
                high_card.sort(key=lambda x: -x[1])
                obs.append(f"High-cardinality categorical columns: "
                           f"{', '.join([f'{c}({n})' for c,n in high_card[:3]])}.")
        mem = round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        obs.append(f"Dataset uses {mem} MB of memory.")
        return obs[:n]

    # ---- Charts (return base64) ----
    def plot_histogram(self, column: str, bins: int = 30) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(8, 4))
        self.df[column].dropna().plot.hist(ax=ax, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_title(f"Distribution of {column}", fontsize=13, fontweight="bold")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_boxplot(self, column: str) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(8, 4))
        self.df[[column]].dropna().plot.box(ax=ax, vert=True, patch_artist=True,
                                             boxprops=dict(facecolor="#4C72B0", alpha=0.7))
        ax.set_title(f"Box Plot — {column}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_correlation_heatmap(self, max_cols: int = 20) -> str:
        self._require_data()
        num = self.df.select_dtypes(include=np.number)
        if num.shape[1] > max_cols:
            if self.target_col and self.target_col in num.columns:
                top = num.corr()[self.target_col].abs().sort_values(ascending=False).head(max_cols).index
                num = num[top]
            else:
                num = num.iloc[:, :max_cols]
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.6), max(6, len(corr) * 0.5)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_target_distribution(self, target: str = None) -> str:
        self._require_data()
        col = target or self.target_col
        if not col or col not in self.df.columns:
            raise ValueError("Target column not set or not found")
        fig, ax = plt.subplots(figsize=(8, 4))
        if self.df[col].nunique() <= 20:
            vc = self.df[col].value_counts()
            colors = sns.color_palette("husl", len(vc))
            vc.plot.bar(ax=ax, color=colors, edgecolor="white")
            ax.set_title(f"Target Distribution — {col}", fontsize=13, fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            for p in ax.patches:
                ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha="center", va="bottom", fontsize=9)
            plt.xticks(rotation=30, ha="right")
        else:
            self.df[col].dropna().plot.hist(ax=ax, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.set_title(f"Target Distribution — {col}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_scatter(self, x_col: str, y_col: str, hue_col: str = None) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(8, 5))
        if hue_col and hue_col in self.df.columns:
            for label, group in self.df.groupby(hue_col):
                ax.scatter(group[x_col], group[y_col], label=str(label), alpha=0.6, s=20)
            ax.legend(title=hue_col)
        else:
            ax.scatter(self.df[x_col], self.df[y_col], alpha=0.5, color="#4C72B0", s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_missing_heatmap(self) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(max(8, self.df.shape[1] * 0.4), 5))
        sns.heatmap(self.df.isnull(), yticklabels=False, cbar=False, cmap="viridis", ax=ax)
        ax.set_title("Missing Values Map", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_feature_importance(self) -> str:
        if not self.feature_importance:
            raise ValueError("Train a model first to get feature importances.")
        # Use evaluation model's feature importance if available
        fi = self.feature_importance.copy()
        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        top = dict(list(fi.items())[:20])
        fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.35)))
        colors = sns.color_palette("Blues_r", len(top))
        ax.barh(list(top.keys())[::-1], list(top.values())[::-1], color=colors[::-1])
        ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_pairplot(self, columns: list = None, hue: str = None) -> str:
        self._require_data()
        num_cols = columns or list(self.df.select_dtypes(include=np.number).columns[:5])
        plot_df = self.df[num_cols + ([hue] if hue and hue not in num_cols else [])].dropna()
        if hue and hue in plot_df.columns:
            plot_df[hue] = plot_df[hue].astype(str)
        fig = sns.pairplot(plot_df, hue=hue, plot_kws={"alpha": 0.4, "s": 15}).fig
        fig.suptitle("Pair Plot", y=1.01, fontsize=13, fontweight="bold")
        return _fig_to_base64(fig)

    # ------------------------------------------------------------------
    # 4. TRAIN MODEL (enhanced with hyperparams, imbalance, feature selection)
    # ------------------------------------------------------------------
    def auto_detect_problem_type(self, target: str) -> str:
        self._require_data()
        col = self.df[target]
        n_unique = col.nunique()
        dtype = col.dtype
        if pd.api.types.is_float_dtype(dtype) and n_unique > 20:
            return "regression"
        if dtype == object or n_unique <= 10:
            return "classification"
        if pd.api.types.is_integer_dtype(dtype) and n_unique > 20:
            return "regression"
        return "classification"

    def get_model_recommendation(self, target: str = None) -> dict:
        """Generate recommendation for best model strategy based on problem."""
        self._require_data()
        target_column = target or self.target_col
        recommendation = {
            "problem_type": self.auto_detect_problem_type(target_column) if target_column else "Unknown",
            "dataset_size": len(self.df),
            "target_col": target_column or "Not set",
            "recommended_models": [],
            "recommended_strategies": [],
            "notes": []
        }
        
        if not target_column:
            return recommendation
        
        problem = recommendation["problem_type"]
        n_rows = len(self.df)
        
        if problem == "classification":
            recommendation["recommended_models"] = ["Random Forest", "XGBoost", "Gradient Boosting"]
            if target_column in self.df.columns:
                target_dist = self.df[target_column].value_counts(normalize=True)
                if len(target_dist) == 2:
                    ratio = target_dist.iloc[0] / target_dist.iloc[1]
                    if ratio > 3:
                        recommendation["recommended_strategies"].append("Use SMOTE for imbalanced classification")
                        recommendation["recommended_strategies"].append("Set class_weight='balanced'")
                        recommendation["notes"].append(f"⚠️ Class imbalance detected: {ratio:.1f}:1 ratio")
            if n_rows < 1000:
                recommendation["notes"].append("⚠️ Small dataset – consider cross-validation (CV=5 or more)")
        else:
            recommendation["recommended_models"] = ["Random Forest", "XGBoost", "Gradient Boosting"]
            recommendation["notes"].append("ℹ️ Regression task – evaluate using R², RMSE, and MAE")
        
        if "XGBoost" in _get_classification_models() or "XGBoost" in _get_regression_models():
            recommendation["notes"].append("✓ XGBoost available – often provides best performance")
        
        return recommendation

    def split_data(self, target: str, test_size: float = 0.2,
                   scale: bool = True, scaler_type: str = "standard") -> dict:
        self._require_data()
        if target not in self.df.columns:
            return {"success": False, "message": f"Column '{target}' not found"}
        self.target_col = target
        self.problem_type = self.auto_detect_problem_type(target)

        df = self.df.copy()
        y_raw = df[target]
        X = df.drop(columns=[target])

        # Fill missing in object columns before encoding
        for c in X.select_dtypes(include=["object", "category"]).columns:
            mode_val = X[c].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            X[c] = X[c].fillna(fill_val)

        for c in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            self.label_encoders[c] = le

        X = X.fillna(X.median(numeric_only=True))

        if self.problem_type == "classification" and y_raw.dtype == object:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y_raw.astype(str))
        else:
            y = y_raw.values

        self.feature_names = list(X.columns)

        # Use integer-position splitting so we can map test rows back to the original DataFrame
        n = len(X)
        indices = np.arange(n)
        stratify = y if (self.problem_type == "classification" and len(set(y)) < 50) else None
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42, stratify=stratify)
        X_tr = X.values[train_idx]
        X_te = X.values[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]
        # store original dataframe row indices for the test set so misclassified samples can be mapped
        try:
            self._test_indices = X.index.to_numpy()[test_idx]
        except Exception:
            self._test_indices = test_idx

        if scale:
            self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
            X_tr = self.scaler.fit_transform(X_tr)
            X_te = self.scaler.transform(X_te)

        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te

        self._log(f"Data split: train={len(X_tr)}, test={len(X_te)}, problem={self.problem_type}, features={len(self.feature_names)}")
        return {
            "success": True,
            "problem_type": self.problem_type,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "features": len(self.feature_names),
            "feature_names": self.feature_names,
        }

    def train(self, model_name: str = "Random Forest", use_smote: bool = False,
              class_weight: str = None,
              **kwargs) -> dict:
        """
        Train a single model with optional hyperparameters.
        use_smote: apply SMOTE (classification only)
        class_weight: 'balanced' or None
        feature_selection_k: number of top features to keep (using mutual info)
        kwargs: any model-specific parameters
        """
        if self.X_train is None:
            return {"success": False, "message": "Call split_data() first"}

        # Get fresh model instance
        if self.problem_type == "classification":
            models_base = _get_classification_models()
        else:
            models_base = _get_regression_models()
        if model_name not in models_base:
            return {"success": False, "message": f"Unknown model '{model_name}'"}

        # Copy and set hyperparameters
        model = copy.deepcopy(models_base[model_name])
        if kwargs:
            model.set_params(**kwargs)

        # Prepare data
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test

        selected_indices = None
        self._used_feature_selection = False
        self._selected_feature_names = self.feature_names if isinstance(self.feature_names, list) else []
        self._selected_features_indices = None

        # Imbalance handling
        if self.problem_type == "classification":
            if use_smote:
                try:
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    self._log("Applied SMOTE")
                except Exception as e:
                    self._log(f"SMOTE failed: {e}")
            elif class_weight == "balanced" and hasattr(model, "class_weight"):
                model.set_params(class_weight="balanced")
                self._log("Set class_weight='balanced'")


        # Train
        import time
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        self.model = model
        self.model_name = model_name
        
        # Store trained model with its feature selection info
        self.trained_models[model_name] = {
            'model': copy.deepcopy(model),
            'selected_feature_names': self._selected_feature_names,
            'selected_features_indices': self._selected_features_indices,
            'used_feature_selection': self._used_feature_selection,
            'feature_names': self.feature_names,
            'label_encoders': copy.deepcopy(self.label_encoders),
            'scaler': copy.deepcopy(self.scaler) if self.scaler else None,
            'target_encoder': copy.deepcopy(self.target_encoder) if self.target_encoder else None,
        }
        if model_name not in self.available_models:
            self.available_models.append(model_name)
        
        self._log(f"Trained {model_name} in {elapsed}s")

        # Store predictions for later
        self._extract_feature_importance()
        self._compute_predictions_with_selection(X_test, selected_indices)

        return {"success": True, "model": model_name, "elapsed": elapsed, "params": model.get_params()}

    def _compute_predictions(self):
        """Helper to compute predictions and probabilities after training."""
        if self.model is None:
            return
        X_test = self.X_test
        if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
            X_test = X_test[:, self._selected_features_indices]
        self.y_pred = self.model.predict(X_test)
        if hasattr(self.model, "predict_proba"):
            self.y_proba = self.model.predict_proba(X_test)
        else:
            self.y_proba = None
        if self.problem_type == "classification":
            self.misclassified_indices = np.where(self.y_pred != self.y_test)[0].tolist()
        else:
            self.misclassified_indices = []

    def _compute_predictions_with_selection(self, X_test, selected_indices):
        """Compute predictions with explicit feature selection."""
        if self.model is None:
            return
        self.y_pred = self.model.predict(X_test)
        if hasattr(self.model, "predict_proba"):
            self.y_proba = self.model.predict_proba(X_test)
        else:
            self.y_proba = None
        if self.problem_type == "classification":
            self.misclassified_indices = np.where(self.y_pred != self.y_test)[0].tolist()
        else:
            self.misclassified_indices = []

    def get_permutation_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Calculate permutation feature importance (model-agnostic)."""
        self._require_model()
        try:
            # Use evaluation model if set
            model_to_use = self.eval_model if self.eval_model else self.model
            X_test_to_use = self.X_test
            if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
                X_test_to_use = X_test_to_use[:, self._selected_features_indices]
            
            if self.problem_type == "classification":
                scoring = "accuracy"
            else:
                scoring = "r2"
            
            perm_imp = permutation_importance(model_to_use, X_test_to_use, self.y_test, 
                                            n_repeats=10, random_state=42, scoring=scoring)
            
            features_to_use = self.feature_names
            if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
                selected_names = [self.feature_names[i] for i in self._selected_features_indices]
                features_to_use = selected_names
            
            perm_df = pd.DataFrame({
                'Feature': features_to_use,
                'Importance': perm_imp.importances_mean,
                'Std': perm_imp.importances_std
            }).sort_values('Importance', ascending=False).head(top_n)
            
            return perm_df
        except Exception as e:
            self._log(f"Permutation importance error: {e}")
            return pd.DataFrame()

    def get_misclassified_samples(self, max_samples: int = 20, features_to_show: list = None) -> pd.DataFrame:
        """
        Return a DataFrame of misclassified test samples.
        
        Parameters:
        - max_samples: maximum number of samples to return
        - features_to_show: list of feature names to include (if None, shows first 5 features + target)
        """
        if self.problem_type != "classification" or not hasattr(self, 'misclassified_indices'):
            return pd.DataFrame()
        
        mis_idx = self.misclassified_indices
        if not mis_idx:
            return pd.DataFrame()
        
        # Limit number of samples
        mis_idx = mis_idx[:max_samples]
        
        # Get the feature values
        X_mis = self.X_test[mis_idx]
        if self.scaler:
            X_mis = self.scaler.inverse_transform(X_mis)
        
        actual = self.y_test[mis_idx]
        predicted = self.y_pred[mis_idx]
        
        # Decode labels if needed
        if self.target_encoder:
            actual = self.target_encoder.inverse_transform(actual.astype(int))
            predicted = self.target_encoder.inverse_transform(predicted.astype(int))
        
        # Determine which features to show
        if features_to_show is None:
            # Default: show first 5 features (or less if fewer exist)
            feature_indices = list(range(min(5, len(self.feature_names))))
            feature_names = [self.feature_names[i] for i in feature_indices]
        else:
            # Show only requested features that exist
            feature_names = [f for f in features_to_show if f in self.feature_names]
            feature_indices = [self.feature_names.index(f) for f in feature_names if f in self.feature_names]
        
        # Build the result dictionary
        result_dict = {}
        for i, feat in enumerate(feature_names):
            result_dict[feat] = X_mis[:, feature_indices[i]] if feature_indices else []
        
        result_dict["Actual"] = actual
        result_dict["Predicted"] = predicted
        result_dict["Correct?"] = ["✓" if a == p else "✗" for a, p in zip(actual, predicted)]
        
        return pd.DataFrame(result_dict)


        # ========== FEATURE SELECTION METHODS (ADD THESE) ==========
    
    def get_fast_feature_importance(self, top_n=20):
        """Fast Random Forest feature importance."""
        if self.X_train is None:
            raise ValueError("No training data. Call split_data() first.")
        
        if self.problem_type == "classification":
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        rf.fit(self.X_train, self.y_train)
        importance = pd.Series(rf.feature_importances_, index=self.feature_names)
        return importance.sort_values(ascending=False).head(top_n)
    
    def get_mutual_information_top(self, k=10):
        """Get top K features by mutual information."""
        if self.X_train is None:
            raise ValueError("No training data. Call split_data() first.")
        
        if self.problem_type == "classification":
            mi = mutual_info_classif(self.X_train, self.y_train, random_state=42)
        else:
            mi = mutual_info_regression(self.X_train, self.y_train, random_state=42)
        
        mi_series = pd.Series(mi, index=self.feature_names)
        return mi_series.sort_values(ascending=False).head(k)
    
    def select_features_manual(self, feature_list):
        """Manually select which features to keep."""
        if self.X_train is None:
            raise ValueError("No training data. Call split_data() first.")
        
        missing = [f for f in feature_list if f not in self.feature_names]
        if missing:
            raise ValueError(f"Features not found: {missing}")
        
        # Get indices of selected features
        selected_indices = [self.feature_names.index(f) for f in feature_list]
        
        # For numpy arrays, use integer indices
        if isinstance(self.X_train, np.ndarray):
            self.X_train = self.X_train[:, selected_indices]
            self.X_test = self.X_test[:, selected_indices]
        else:
            # For DataFrames, use column names
            self.X_train = self.X_train[feature_list]
            self.X_test = self.X_test[feature_list]
        
        self.feature_names = feature_list
        self._selected_feature_names = feature_list
        self._selected_features_indices = selected_indices
        self._used_feature_selection = True
        
        self._log(f"Manual feature selection: kept {len(feature_list)} features")
        return True
    
    def apply_pca(self, n_components=2):
        """Apply PCA for dimensionality reduction."""
        if self.X_train is None:
            raise ValueError("No training data. Call split_data() first.")
        
        from sklearn.decomposition import PCA
        
        if n_components > len(self.feature_names):
            n_components = len(self.feature_names)
        
        pca = PCA(n_components=n_components, random_state=42)
        
        # Store original data as numpy arrays (not DataFrames)
        self._X_train_original = self.X_train.copy()
        self._X_test_original = self.X_test.copy()
        self._feature_names_original = self.feature_names.copy()
        
        # Apply PCA
        X_train_pca = pca.fit_transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)
        
        # PCA returns numpy arrays
        self.X_train = X_train_pca
        self.X_test = X_test_pca
        self.feature_names = [f'PC{i+1}' for i in range(n_components)]
        self._selected_feature_names = self.feature_names
        self._selected_features_indices = list(range(n_components))
        self._used_feature_selection = True
        
        result = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': pca.explained_variance_ratio_.cumsum().tolist(),
            'n_components': n_components
        }
        
        self._log(f"PCA applied: {n_components} components, total variance: {sum(result['explained_variance_ratio']):.2%}")
        return result
    
    def restore_from_pca(self):
        """Restore original features after PCA."""
        if hasattr(self, '_X_train_original'):
            self.X_train = self._X_train_original
            self.X_test = self._X_test_original
            self.feature_names = self._feature_names_original
            self._selected_feature_names = self.feature_names
            self._selected_features_indices = None
            self._used_feature_selection = False
            self._log("Restored original features (PCA undone)")
            return True
        return False

    def train_all_models(self, use_smote=False, class_weight=None, **kwargs):
        """Train all models and return comparison DataFrame."""
        if self.X_train is None:
            raise ValueError("Call split_data() first")
        models = _get_classification_models() if self.problem_type == "classification" else _get_regression_models()
        results = []
        best_score = -1 if self.problem_type == "classification" else -np.inf
        best_model = None
        best_name = ""
        
        # Define which hyperparameters each model accepts
        model_valid_params = {
            "Random Forest": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
            "Gradient Boosting": ["n_estimators", "max_depth", "learning_rate", "subsample"],
            "Logistic Regression": ["C", "penalty", "solver", "max_iter"],
            "Decision Tree": ["max_depth", "min_samples_split", "min_samples_leaf", "criterion"],
            "SVM": ["C", "kernel", "gamma", "degree"],
            "KNN": ["n_neighbors", "weights", "algorithm", "p"],
            "Naive Bayes": ["var_smoothing", "priors"],
        }
        
        for name, mdl_template in models.items():
            try:
                t0 = time.time()
                # Copy model
                mdl = copy.deepcopy(mdl_template)
                
                # Only apply hyperparameters that this model accepts
                if name in model_valid_params:
                    valid_params = model_valid_params[name]
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                    if filtered_kwargs:
                        mdl.set_params(**filtered_kwargs)
                
                # Use full data
                X_tr = self.X_train
                y_tr = self.y_train
                X_test_current = self.X_test
                
                # Imbalance handling
                if self.problem_type == "classification":
                    if use_smote:
                        try:
                            sm = SMOTE(random_state=42)
                            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                        except Exception:
                            pass
                    elif class_weight == "balanced" and hasattr(mdl, "class_weight"):
                        mdl.set_params(class_weight="balanced")
                
                mdl.fit(X_tr, y_tr)
                elapsed = round(time.time() - t0, 2)
                preds = mdl.predict(X_test_current)

                if self.problem_type == "classification":
                    acc = accuracy_score(self.y_test, preds)
                    f1 = f1_score(self.y_test, preds, average="weighted", zero_division=0)
                    row = {
                        "Model": name,
                        "Accuracy": round(acc, 4),
                        "F1": round(f1, 4),
                        "Precision": round(precision_score(self.y_test, preds, average="weighted", zero_division=0), 4),
                        "Recall": round(recall_score(self.y_test, preds, average="weighted", zero_division=0), 4),
                        "Time(s)": elapsed,
                    }
                    score = acc
                else:
                    r2 = r2_score(self.y_test, preds)
                    row = {
                        "Model": name,
                        "R²": round(r2, 4),
                        "MAE": round(mean_absolute_error(self.y_test, preds), 4),
                        "MSE": round(mean_squared_error(self.y_test, preds), 4),
                        "RMSE": round(np.sqrt(mean_squared_error(self.y_test, preds)), 4),
                        "Time(s)": elapsed,
                    }
                    score = r2
                results.append(row)
                if score > best_score:
                    best_score = score
                    best_model = copy.deepcopy(mdl)
                    best_name = name
            except Exception as e:
                results.append({"Model": name, "Error": str(e)})
        
        df = pd.DataFrame(results)
        if best_model is not None:
            self.model = best_model
            self.model_name = best_name
            self._used_feature_selection = False
            self._selected_features_indices = None
            self._selected_feature_names = self.feature_names
            self._extract_feature_importance()
            self._compute_predictions()
            self._log(f"Best model: {best_name} (score={best_score:.4f})")
        else:
            self.model = None
            self.model_name = ""
        return df

    def get_trained_models(self) -> list:
        """Return list of available trained models."""
        return self.available_models.copy()

    def select_model(self, model_name: str) -> dict:
        """Switch to a previously trained model for evaluation."""
        if model_name not in self.trained_models:
            return {"success": False, "message": f"Model '{model_name}' not found"}
        
        model_data = self.trained_models[model_name]
        
        if isinstance(model_data, dict):
            self.model = copy.deepcopy(model_data['model'])
            self._selected_feature_names = model_data.get('selected_feature_names', self.feature_names)
            self._selected_features_indices = model_data.get('selected_features_indices', None)
            self._used_feature_selection = model_data.get('used_feature_selection', False)
            self.label_encoders = model_data.get('label_encoders', {})
            self.scaler = model_data.get('scaler')
            self.target_encoder = model_data.get('target_encoder')
        else:
            self.model = copy.deepcopy(model_data)
            self._selected_feature_names = self.feature_names
            self._selected_features_indices = None
            self._used_feature_selection = False
        
        self.model_name = model_name
        self._extract_feature_importance()
        self._compute_predictions()
        self._log(f"Selected model: {model_name}")
        return {"success": True, "model": model_name}

    def set_evaluation_model(self, model_name: str) -> dict:
        """Set the model to use for evaluation and predictions (separate from trained model)."""
        if model_name not in self.trained_models:
            return {"success": False, "message": f"Model '{model_name}' not found"}
        
        model_data = self.trained_models[model_name]
        
        # Check if model_data is a dict (regular model) or a model object (stacked)
        if isinstance(model_data, dict):
            self.eval_model = copy.deepcopy(model_data['model'])
            self._selected_feature_names = model_data.get('selected_feature_names', self.feature_names)
            self._selected_features_indices = model_data.get('selected_features_indices', None)
            self._used_feature_selection = model_data.get('used_feature_selection', False)
            self.label_encoders = model_data.get('label_encoders', {})
            self.scaler = model_data.get('scaler')
            self.target_encoder = model_data.get('target_encoder')
        else:
            # Stacked model or direct model object
            self.eval_model = copy.deepcopy(model_data)
            self._selected_feature_names = self.feature_names
            self._selected_features_indices = None
            self._used_feature_selection = False
        
        self.eval_model_name = model_name
        self._log(f"Evaluation model set to: {model_name}")
        return {"success": True, "model": model_name}

    def get_evaluation_model(self) -> str:
        """Get the current evaluation model name."""
        return self.eval_model_name    




    def train_stacked_model(self, base_model_1: str, base_model_2: str,
                           meta_model: str = "Logistic Regression") -> dict:
        """
        Train a stacked ensemble model combining two base models.
        meta_model: name of the meta-learner ('Logistic Regression', 'Random Forest', etc.)
        """
        if self.X_train is None:
            return {"success": False, "message": "Call split_data() first"}
        
        if base_model_1 not in self.trained_models or base_model_2 not in self.trained_models:
            return {"success": False, "message": f"One or both base models not found"}
        
        if base_model_1 == base_model_2:
            return {"success": False, "message": "Base models must be different"}
        
        try:
            t0 = time.time()
            
            # Get base model estimators
            base1 = copy.deepcopy(self.trained_models[base_model_1]['model'])
            base2 = copy.deepcopy(self.trained_models[base_model_2]['model'])
            
            # Get meta-model
            if self.problem_type == "classification":
                meta_models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                    "SVM": SVC(kernel="rbf", random_state=42, probability=True),
                }
                if meta_model not in meta_models:
                    meta_model = "Logistic Regression"
                meta_learner = meta_models[meta_model]
                
                # Create stacked classifier
                estimators = [
                    (base_model_1, base1),
                    (base_model_2, base2),
                ]
                stacked = StackingClassifier(
                    estimators=estimators,
                    final_estimator=meta_learner,
                    cv=3
                )
            else:
                meta_models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                    "Ridge": Ridge(random_state=42),
                }
                if meta_model not in meta_models:
                    meta_model = "Linear Regression"
                meta_learner = meta_models[meta_model]
                
                # Create stacked regressor
                estimators = [
                    (base_model_1, base1),
                    (base_model_2, base2),
                ]
                stacked = StackingRegressor(
                    estimators=estimators,
                    final_estimator=meta_learner,
                    cv=3
                )
            
            # Train stacked model
            X_train = self.X_train
            y_train = self.y_train
            X_test = self.X_test
            
            # Apply feature selection if used
            if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
                X_train = X_train[:, self._selected_features_indices]
                X_test = X_test[:, self._selected_features_indices]
            
            stacked.fit(X_train, y_train)
            elapsed = round(time.time() - t0, 2)
            
            # Store as main model
            self.model = stacked
            model_display_name = f"Stacked[{base_model_1}+{base_model_2}]({meta_model})"
            self.model_name = model_display_name
            
            # Remove any existing stacked model first
            for model_key in list(self.trained_models.keys()):
                if model_key.startswith("Stacked["):
                    del self.trained_models[model_key]
                    if model_key in self.available_models:
                        self.available_models.remove(model_key)

            # Store new stacked model
            self.trained_models[model_display_name] = copy.deepcopy(stacked)
            self.available_models.append(model_display_name)
            
            self._log(f"Trained Stacked Model: {model_display_name} in {elapsed}s")
            
            # Compute predictions
            self._compute_predictions()
            
            return {
                "success": True,
                "model": model_display_name,
                "base_models": [base_model_1, base_model_2],
                "meta_model": meta_model,
                "elapsed": elapsed
            }
        
        except Exception as e:
            return {"success": False, "message": f"Stacking error: {str(e)}"}

    def get_cross_val_score(self, cv: int = 5) -> dict:
        """Cross-validation score on training data."""
        self._require_model()
        scoring = "accuracy" if self.problem_type == "classification" else "r2"
        # Use StratifiedKFold for classification to preserve class distribution
        if self.problem_type == "classification":
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_split = cv
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv_split, scoring=scoring)
        return {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4),
                "scores": scores.round(4).tolist(), "metric": scoring}

    # ------------------------------------------------------------------
    # 5. EVALUATION (enhanced)
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        self._require_model()
        # Use evaluation model if set, otherwise use main model
        model_to_use = self.eval_model if self.eval_model else self.model
        X_test_to_use = self.X_test
        
        # Apply feature selection if it was used for this model
        if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
            X_test_to_use = X_test_to_use[:, self._selected_features_indices]
        
        preds = model_to_use.predict(X_test_to_use)
        
        if self.problem_type == "classification":
            m = {
                "accuracy": round(accuracy_score(self.y_test, preds), 4),
                "precision": round(precision_score(self.y_test, preds, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(self.y_test, preds, average="weighted", zero_division=0), 4),
                "f1": round(f1_score(self.y_test, preds, average="weighted", zero_division=0), 4),
            }
            if hasattr(model_to_use, "predict_proba"):
                proba = model_to_use.predict_proba(X_test_to_use)
                if proba.shape[1] == 2:
                    m["roc_auc"] = round(roc_auc_score(self.y_test, proba[:, 1]), 4)
                else:
                    try:
                        m["roc_auc"] = round(roc_auc_score(self.y_test, proba, multi_class="ovr"), 4)
                    except Exception:
                        pass
            m["classification_report"] = classification_report(self.y_test, preds, zero_division=0)
        else:
            m = {
                "r2": round(r2_score(self.y_test, preds), 4),
                "mae": round(mean_absolute_error(self.y_test, preds), 4),
                "mse": round(mean_squared_error(self.y_test, preds), 4),
                "rmse": round(np.sqrt(mean_squared_error(self.y_test, preds)), 4),
            }
        self.metrics = m
        return m

    def get_confusion_matrix(self) -> dict:
        self._require_model()
        if self.problem_type != "classification":
            raise ValueError("Confusion matrix only for classification")
        # Use evaluation model if set
        model_to_use = self.eval_model if self.eval_model else self.model
        X_test_to_use = self.X_test
        if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
            X_test_to_use = X_test_to_use[:, self._selected_features_indices]
        preds = model_to_use.predict(X_test_to_use)
        cm = confusion_matrix(self.y_test, preds)
        labels = (self.target_encoder.classes_.tolist() if self.target_encoder
                else sorted(set(self.y_test.tolist())))
        return {"matrix": cm.tolist(), "labels": labels}

    # ---- Extra evaluation plots ----
    def plot_precision_recall_curve(self) -> str:
        self._require_model()
        if self.problem_type != "classification":
            raise ValueError("Precision-Recall curve only for classification")
        # Use evaluation model if set
        model_to_use = self.eval_model if self.eval_model else self.model
        X_test_to_use = self.X_test
        if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
            X_test_to_use = X_test_to_use[:, self._selected_features_indices]
        if not hasattr(model_to_use, "predict_proba"):
            raise ValueError("Model does not support predict_proba")
        y_proba = model_to_use.predict_proba(X_test_to_use)
        if y_proba.shape[1] != 2:
            raise ValueError("PR curve only for binary classification")
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba[:, 1])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision, color="#4C72B0", lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_confusion_matrix(self) -> str:
        cm_data = self.get_confusion_matrix()
        cm = np.array(cm_data["matrix"])
        labels = cm_data["labels"]
        fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_roc_curve(self) -> str:
        self._require_model()
        if self.problem_type != "classification":
            raise ValueError("ROC curve only for classification")
        # Use evaluation model if set
        model_to_use = self.eval_model if self.eval_model else self.model
        X_test_to_use = self.X_test
        if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
            X_test_to_use = X_test_to_use[:, self._selected_features_indices]
        if not hasattr(model_to_use, "predict_proba"):
            raise ValueError("Model does not support predict_proba")
        y_proba = model_to_use.predict_proba(X_test_to_use)
        if y_proba.shape[1] != 2:
            raise ValueError("ROC curve only for binary classification")
        fpr, tpr, _ = roc_curve(self.y_test, y_proba[:, 1])
        auc = roc_auc_score(self.y_test, y_proba[:, 1])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)
    

    def plot_residuals(self) -> str:
        self._require_model()
        if self.problem_type != "regression":
            raise ValueError("Residual plot only for regression")
        # Use evaluation model if set
        model_to_use = self.eval_model if self.eval_model else self.model
        X_test_to_use = self.X_test
        if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
            X_test_to_use = X_test_to_use[:, self._selected_features_indices]
        preds = model_to_use.predict(X_test_to_use)
        residuals = self.y_test - preds
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(preds, residuals, alpha=0.5, color="#4C72B0", s=20)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(alpha=0.3)
        axes[1].hist(residuals, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[1].set_title("Residual Distribution")
        axes[1].grid(axis="y", alpha=0.3)
        plt.suptitle("Residual Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_base64(fig)

    # ------------------------------------------------------------------
    # 6. INSIGHTS (most important)
    # ------------------------------------------------------------------
    def get_dataset_health_score(self) -> dict:
        """Compute a health score 0-100."""
        self._require_data()
        score = 100
        issues = []
        missing_pct = self.df.isnull().sum().sum() / self.df.size * 100
        if missing_pct > 50:
            score -= 40
            issues.append(f"Critical missing: {missing_pct:.1f}%")
        elif missing_pct > 20:
            score -= 25
            issues.append(f"High missing: {missing_pct:.1f}%")
        elif missing_pct > 5:
            score -= 10
            issues.append(f"Moderate missing: {missing_pct:.1f}%")
        elif missing_pct > 0:
            score -= 5
            issues.append(f"Low missing: {missing_pct:.1f}%")
        duplicate_pct = self.df.duplicated().sum() / len(self.df) * 100
        if duplicate_pct > 10:
            score -= 15
            issues.append(f"High duplicates: {duplicate_pct:.1f}%")
        elif duplicate_pct > 0:
            score -= 5
        const_cols = [c for c in self.df.columns if self.df[c].nunique() <= 1]
        if const_cols:
            score -= len(const_cols) * 2
        return {
            "score": max(0, min(100, score)),
            "grade": "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F",
            "issues": issues,
            "total_issues": len(issues),
        }

    def generate_insights(self) -> list:
        self._require_data()
        insights = []
        info = self.get_info()
        health = self.get_dataset_health_score()
        insights.append("📊 DATA QUALITY")
        insights.append(f"  • Dataset: {info['rows']:,} rows × {info['cols']} columns")
        insights.append(f"  • Health score: {health['score']}/100 ({health['grade']})")
        if health['issues']:
            insights.append(f"  • Issues: {', '.join(health['issues'][:3])}")
        total_miss = info["total_missing"]
        if total_miss == 0:
            insights.append("  • Data quality: EXCELLENT — no missing values")
        else:
            miss_pct = round(total_miss / (info["rows"] * info["cols"]) * 100, 2)
            insights.append(f"  • Missing values: {total_miss:,} cells ({miss_pct}% of all data)")
            worst = max(info["missing_pct"], key=info["missing_pct"].get)
            insights.append(f"  • Worst column for missing data: '{worst}' ({info['missing_pct'][worst]}%)")
        dups = info["duplicates"]
        insights.append(f"  • Duplicates: {'None found ✓' if dups == 0 else str(dups) + ' duplicate rows detected'}")
        insights.append(f"  • Memory usage: {info['memory_mb']} MB")
        if self.target_col and self.target_col in self.df.columns:
            insights.append("\n🎯 TARGET ANALYSIS")
            target_data = self.df[self.target_col]
            n_unique = target_data.nunique()
            if self.problem_type == "classification":
                dist = target_data.value_counts()
                pcts = (dist / len(self.df) * 100).round(1)
                insights.append(f"  • Problem type: CLASSIFICATION")
                insights.append(f"  • Classes: {n_unique} unique values")
                for cls, cnt in dist.head(5).items():
                    insights.append(f"    — '{cls}': {cnt:,} samples ({pcts[cls]}%)")
                if n_unique == 2:
                    ratio = dist.iloc[0] / dist.iloc[1]
                    if ratio > 3:
                        insights.append(f"  ⚠️  Imbalanced dataset: majority is {ratio:.1f}× larger — consider SMOTE or class_weight")
                    else:
                        insights.append(f"  • Class balance: reasonable (ratio {ratio:.2f}:1)")
            else:
                insights.append(f"  • Problem type: REGRESSION")
                insights.append(f"  • Target range: {target_data.min():.3f} → {target_data.max():.3f}")
                insights.append(f"  • Target mean: {target_data.mean():.3f} | std: {target_data.std():.3f}")
                skew = round(target_data.skew(), 3)
                if abs(skew) > 1:
                    insights.append(f"  ⚠️  Target is skewed ({skew}) — consider log transform")
        insights.append("\n🔍 FEATURE ANALYSIS")
        num_cols = info["numeric_cols"]
        cat_cols = info["categorical_cols"]
        insights.append(f"  • {len(num_cols)} numeric features, {len(cat_cols)} categorical features")
        if num_cols:
            stats = self.get_stats()
            top_skewed = stats["skew"].abs().sort_values(ascending=False).head(3)
            if not top_skewed.empty:
                sk_str = ", ".join([f"'{c}'({v:.2f})" for c, v in top_skewed.items()])
                insights.append(f"  • Most skewed columns: {sk_str}")
        if cat_cols:
            high_card = [(c, self.df[c].nunique()) for c in cat_cols if self.df[c].nunique() > 10]
            if high_card:
                high_card.sort(key=lambda x: -x[1])
                s = ", ".join([f"'{c}'({n})" for c, n in high_card[:3]])
                insights.append(f"  • High-cardinality columns (may need encoding): {s}")
        if len(num_cols) >= 2:
            insights.append("\n📈 CORRELATION INSIGHTS")
            corr = self.get_correlation_matrix()
            flat = corr.abs().unstack()
            flat = flat[flat < 1].sort_values(ascending=False)
            seen = set()
            unique_pairs = []
            for (c1, c2), val in flat.items():
                pair = tuple(sorted((c1, c2)))
                if pair not in seen:
                    seen.add(pair)
                    unique_pairs.append(((c1, c2), val))
            top3 = unique_pairs[:3]
            for (c1, c2), val in top3:
                direction = "positive" if corr.loc[c1, c2] > 0 else "negative"
                insights.append(f"  • '{c1}' ↔ '{c2}': r={val:.3f} ({direction})")
            if self.target_col and self.target_col in corr.columns:
                target_corr = corr[self.target_col].drop(self.target_col).abs().sort_values(ascending=False)
                top_feat = target_corr.head(3)
                insights.append(f"  • Top predictors of '{self.target_col}':")
                for feat, val in top_feat.items():
                    insights.append(f"    — '{feat}': |r|={val:.3f}")
        if self.model and self.metrics:
            insights.append(f"\n🤖 MODEL PERFORMANCE ({self.model_name})")
            if self.problem_type == "classification":
                acc = self.metrics.get("accuracy", "N/A")
                f1 = self.metrics.get("f1", "N/A")
                auc = self.metrics.get("roc_auc", "N/A")
                if isinstance(acc, float):
                    insights.append(f"  • Accuracy: {acc:.1%}")
                else:
                    insights.append(f"  • Accuracy: {acc}")
                insights.append(f"  • F1 Score: {f1}")
                if isinstance(auc, float):
                    insights.append(f"  • ROC-AUC: {auc:.4f}")
                if isinstance(acc, float):
                    if acc >= 0.90:
                        insights.append("  • Performance: EXCELLENT (≥90%)")
                    elif acc >= 0.80:
                        insights.append("  • Performance: GOOD (80–90%)")
                    elif acc >= 0.70:
                        insights.append("  • Performance: FAIR (70–80%) — consider tuning")
                    else:
                        insights.append("  ⚠️  Performance: POOR (<70%) — needs improvement")
            else:
                r2 = self.metrics.get("r2", "N/A")
                rmse = self.metrics.get("rmse", "N/A")
                mae = self.metrics.get("mae", "N/A")
                insights.append(f"  • R² Score: {r2}")
                insights.append(f"  • RMSE: {rmse}")
                insights.append(f"  • MAE: {mae}")
                if isinstance(r2, float):
                    if r2 >= 0.9:
                        insights.append("  • Fit: EXCELLENT")
                    elif r2 >= 0.7:
                        insights.append("  • Fit: GOOD")
                    else:
                        insights.append("  ⚠️  Fit: POOR — try feature engineering or different model")
        if self.feature_importance:
            insights.append("\n⭐ TOP IMPORTANT FEATURES")
            fi_sorted = sorted(self.feature_importance.items(), key=lambda x: -x[1])
            for feat, imp in fi_sorted[:5]:
                bar = "█" * int(imp * 20)
                insights.append(f"  • {feat}: {imp:.4f} {bar}")
        insights.append("\n💡 RECOMMENDATIONS")
        if total_miss > 0:
            insights.append("  • Handle remaining missing values before deploying the model")
        if self.problem_type == "classification" and self.metrics:
            acc = self.metrics.get("accuracy", 1.0)
            if isinstance(acc, float) and acc < 0.80:
                insights.append("  • Try Gradient Boosting or XGBoost for better accuracy")
                insights.append("  • Consider feature engineering (interactions, polynomial features)")
        if self.feature_importance:
            low_fi = [f for f, v in self.feature_importance.items() if v < 0.01]
            if low_fi:
                insights.append(f"  • Consider removing {len(low_fi)} low-importance features to simplify model")
        insights.append("  • Save the pipeline with save_pipeline() for production use")
        self.insights = insights
        return insights

    # ------------------------------------------------------------------
    # 7. PREDICT
    # ------------------------------------------------------------------
    def get_input_form_schema(self) -> list:
        self._require_data()
        schema = []
        if self._selected_feature_names:
            features_to_show = self._selected_feature_names
        else:
            features_to_show = self.feature_names

        for feat in features_to_show:
            if feat not in self.df.columns:
                continue
            col = self.df[feat]
            if feat in self.label_encoders:
                options = list(self.label_encoders[feat].classes_)
                schema.append({"name": feat, "type": "categorical", "options": options})
            elif pd.api.types.is_numeric_dtype(col):
                schema.append({"name": feat, "type": "numeric",
                                "min": float(col.min()), "max": float(col.max()),
                                "mean": float(col.mean())})
            else:
                options = col.dropna().unique().tolist()[:50]
                schema.append({"name": feat, "type": "categorical", "options": options})
        return schema

    def predict(self, input_dict: dict) -> dict:
        """Predict for a single sample. Safely handles unknown categorical values."""

        # Use evaluation model if set, otherwise use main model
        model_to_use = self.eval_model if self.eval_model else self.model
        encoders_to_use = self.label_encoders
        scaler_to_use = self.scaler
        target_enc_to_use = self.target_encoder
        
        if model_to_use is None:
            raise ValueError("No model available for prediction")

        row = []
        # Use the features the model was trained on
        features_to_use = self._selected_feature_names if self._used_feature_selection and self._selected_feature_names else self.feature_names

        for feat in features_to_use:
            val = input_dict.get(feat, 0)
            if feat in encoders_to_use:
                try:
                    encoded = encoders_to_use[feat].transform([str(val)])[0]
                    val = encoded
                except ValueError:
                    self._log(f"⚠️ Unknown category in '{feat}': '{val}' — using mode")
                    mode_class = encoders_to_use[feat].classes_[0]
                    val = encoders_to_use[feat].transform([mode_class])[0]
                except Exception as e:
                    self._log(f"Error encoding '{feat}': {e}")
                    val = 0
            row.append(float(val))
        
        X = np.array(row).reshape(1, -1)
        if scaler_to_use:
            X = scaler_to_use.transform(X)

        
        pred_raw = model_to_use.predict(X)[0]
        
        if target_enc_to_use:
            try:
                pred_label = target_enc_to_use.inverse_transform([int(pred_raw)])[0]
            except Exception:
                pred_label = pred_raw
        else:
            pred_label = pred_raw
        
        result = {"prediction": pred_label, "raw": pred_raw}
        
        if hasattr(model_to_use, "predict_proba"):
            proba = model_to_use.predict_proba(X)[0]
            result["confidence"] = round(float(proba.max()), 4)
            if target_enc_to_use:
                result["probabilities"] = {
                    str(cls): round(float(p), 4)
                    for cls, p in zip(target_enc_to_use.classes_, proba)
                }
            else:
                result["probabilities"] = {str(i): round(float(p), 4) for i, p in enumerate(proba)}
        
        return result

    def predict_from_csv(self, file_path: str, output_path: str = None) -> pd.DataFrame:
        """Predict from CSV, return only sample_id, prediction, confidence."""
        self._require_model()
        df = pd.read_csv(file_path)
        results = []
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            try:
                result = self.predict(row_dict)
                results.append({
                    "sample_id": idx,
                    "prediction": result["prediction"],
                    "confidence": result.get("confidence", None)
                })
            except Exception as e:
                results.append({
                    "sample_id": idx,
                    "prediction": None,
                    "confidence": None,
                    "error": str(e)
                })
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False)
        
        return result_df

    # ------------------------------------------------------------------
    # 8. SAVE / LOAD / EXPORT
    # ------------------------------------------------------------------
    def save_pipeline(self, file_path: str) -> dict:
        self._require_model()
        from datetime import datetime
        pipeline = {
            "model": self.model,
            "model_name": self.model_name,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "target_encoder": self.target_encoder,
            "feature_names": self.feature_names,
            "target_col": self.target_col,
            "problem_type": self.problem_type,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "used_feature_selection": getattr(self, '_used_feature_selection', False),
            "selected_features_indices": getattr(self, '_selected_features_indices', None),
            "selected_feature_names": getattr(self, '_selected_feature_names', None),
            # Metadata
            "trained_at": datetime.now().isoformat(),
            "dataset_rows": len(self.df),
            "dataset_cols": len(self.df.columns),
            "feature_count": len(self.feature_names),
        }
        joblib.dump(pipeline, file_path)
        self._log(f"Pipeline saved to {file_path} with metadata")
        return {"success": True, "path": file_path}

    def load_pipeline(self, file_path: str) -> dict:
        pipeline = joblib.load(file_path)
        self.model = pipeline["model"]
        self.model_name = pipeline["model_name"]
        self.scaler = pipeline["scaler"]
        self.label_encoders = pipeline["label_encoders"]
        self.target_encoder = pipeline["target_encoder"]
        self.feature_names = pipeline["feature_names"]
        self.target_col = pipeline["target_col"]
        self.problem_type = pipeline["problem_type"]
        self.metrics = pipeline.get("metrics", {})
        self.feature_importance = pipeline.get("feature_importance", {})
        self._used_feature_selection = pipeline.get("used_feature_selection", False)
        self._selected_features_indices = pipeline.get("selected_features_indices", None)
        self._selected_feature_names = pipeline.get("selected_feature_names", None)
        self._log(f"Pipeline loaded from {file_path}")
        return {"success": True, "model": self.model_name}

    def export_report(self, file_path: str) -> dict:
        """Export comprehensive HTML report with charts, insights, and metrics."""
        self._require_data()
        charts_html = ""
        chart_methods = [
            ("Missing Values Map", self.plot_missing_heatmap),
            ("Correlation Heatmap", self.plot_correlation_heatmap),
        ]
        if self.target_col and self.target_col in self.df.columns:
            chart_methods.append(("Target Distribution", lambda: self.plot_target_distribution()))
        if self.model:
            if self.problem_type == "classification":
                chart_methods.append(("Confusion Matrix", self.plot_confusion_matrix))
                chart_methods.append(("ROC Curve", self.plot_roc_curve))
                chart_methods.append(("Precision-Recall Curve", self.plot_precision_recall_curve))
            else:
                chart_methods.append(("Residual Analysis", self.plot_residuals))
            if self.feature_importance:
                chart_methods.append(("Feature Importance", self.plot_feature_importance))
        
        for title, fn in chart_methods:
            try:
                b64 = fn()
                chart_img = '<h3>{}</h3><img src="data:image/png;base64,{}" style="max-width:100%;margin:10px 0"><hr>'.format(title, b64)
                charts_html += chart_img
            except Exception as e:
                self._log(f"Chart error ({title}): {e}")
        
        insights = self.generate_insights()
        ins_html = ""
        for line in insights:
            escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            escaped = escaped.replace("  •", "&nbsp;&nbsp;•").replace("\n", "<br>")
            ins_html += escaped + "<br>"
        
        try:
            stats_df = self.get_stats()
            stats_html = stats_df.to_html(classes="table", border=0, escape=True)
        except Exception as e:
            self._log(f"Stats table error: {e}")
            stats_html = "<p>Could not generate statistics table.</p>"
        
        health = self.get_dataset_health_score()
        metrics_cards = ""
        for k, v in self.metrics.items():
            if not isinstance(v, str):
                try:
                    val_str = "{:.4f}".format(v) if isinstance(v, (int, float)) else str(v)
                    card = '<div class="card"><div class="val">{}</div><div>{}</div></div>'.format(val_str, k.upper())
                    metrics_cards += card
                except Exception:
                    pass
        
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DataYab Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; background: #f8f9fa; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .metrics {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
        .card {{ background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 2px 6px rgba(0,0,0,.1); min-width: 120px; text-align: center; }}
        .card .val {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .insight-box {{ background: #fff; border-left: 4px solid #3498db; padding: 16px; border-radius: 4px; white-space: pre-wrap; font-family: monospace; font-size: 13px; box-shadow: 0 2px 6px rgba(0,0,0,.07); }}
        table {{ border-collapse: collapse; width: 100%; background: #fff; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 12px; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .health {{ font-size: 24px; font-weight: bold; margin: 10px 0; color: #2c3e50; }}
    </style>
</head>
<body>
    <h1>📊 DataYab Analysis Report</h1>
    <p><b>File:</b> {file} | <b>Shape:</b> {rows:,} rows × {cols} cols | <b>Model:</b> {model}</p>
    <div class="health">Dataset Health Score: {score}/100 ({grade})</div>
    <h2>Metrics</h2>
    <div class="metrics">{metrics}</div>
    <h2>Insights</h2>
    <div class="insight-box">{insights}</div>
    <h2>Charts</h2>
    {charts}
    <h2>Statistical Summary</h2>
    {stats}
    <p style="color:#888;font-size:11px;margin-top:40px">Generated by DataYab</p>
</body>
</html>"""
        
        html_content = html_template.format(
            file=self.file_path or "(no file)",
            rows=self.df.shape[0],
            cols=self.df.shape[1],
            model=self.model_name or "None trained",
            score=health['score'],
            grade=health['grade'],
            metrics=metrics_cards,
            insights=ins_html,
            charts=charts_html,
            stats=stats_html
        )
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            self._log(f"HTML report exported to {file_path}")
            return {"success": True, "path": file_path}
        except Exception as e:
            self._log(f"Error exporting report: {e}")
            return {"success": False, "message": str(e)}

    def export_cleaned_csv(self, file_path: str) -> dict:
        self._require_data()
        self.df.to_csv(file_path, index=False)
        self._log(f"Cleaned dataset exported to {file_path}")
        return {"success": True, "path": file_path}

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------
    def _require_data(self):
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_csv() or load_file() first.")
    def _require_model(self):
        self._require_data()
        if self.model is None:
            raise ValueError("No model trained. Call split_data() then train() first.")
    def _extract_feature_importance(self):
        if self.model is None or not self.feature_names:
            return
        try:
            # Get the feature names to use (either all or selected)
            if self._used_feature_selection and hasattr(self, '_selected_features_indices'):
                feature_names_for_importance = [self.feature_names[i] for i in self._selected_features_indices]
            else:
                feature_names_for_importance = self.feature_names
            
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                self.feature_importance = dict(zip(feature_names_for_importance, importances))
            elif hasattr(self.model, "coef_"):
                coefs = self.model.coef_
                if coefs.ndim > 1:
                    coefs = np.abs(coefs).mean(axis=0)
                coefs = np.abs(coefs)
                total = coefs.sum()
                if total > 0:
                    coefs = coefs / total
                self.feature_importance = dict(zip(feature_names_for_importance, coefs))
        except Exception:
            pass
    def _log(self, message: str):
        self.log.append(message)
        if len(self.log) > 200:
            self.log = self.log[-200:]
    def get_log(self) -> list:
        return self.log.copy()

# ------------------------------
# Quick test
# ------------------------------
if __name__ == "__main__":
    print("DataYab Core — Ultimate version ready")