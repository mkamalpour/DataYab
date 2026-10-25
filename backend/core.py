"""Dataset exploration, explicit cleaning, and local model orchestration."""

from .storage import atomic_target

import io
import base64

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer


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


# ------------------------------
# Core Class
# ------------------------------
from .modeling import ModelingMixin, model_catalog


class DataYabCore(ModelingMixin):
    def __init__(self):
        self.df = None
        self.original_df = None
        self.file_path = ""
        self.log = []
        self.insights = []
        self._reset_learning()

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------

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
            "categorical_cols": list(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime_cols": list(df.select_dtypes(include=["datetime64"]).columns),
        }

    # ------------------------------------------------------------------
    # 2. CLEAN DATA
    # ------------------------------------------------------------------
    def get_missing_report(self) -> pd.DataFrame:
        self._require_data()
        miss = self.df.isnull().sum()
        pct = (miss / len(self.df) * 100).round(2)
        report = pd.DataFrame(
            {"Column": miss.index, "Missing": miss.values, "Percent": pct.values}
        )
        return (
            report[report["Missing"] > 0]
            .sort_values("Missing", ascending=False)
            .reset_index(drop=True)
        )

    def handle_missing(self, strategy: str = "mean", columns: list = None) -> dict:
        self._require_data()
        cols = columns or [c for c in self.df.columns if c != self.target_col]
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
            numeric_cols = [
                c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])
            ]
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        elif strategy == "iterative":
            # Iterative imputation (MICE-like)
            from sklearn.impute import IterativeImputer

            numeric_cols = [
                c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])
            ]
            if numeric_cols:
                imputer = IterativeImputer(
                    random_state=42, max_iter=10, keep_empty_features=True
                )
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        else:
            return {"success": False, "message": f"Unknown strategy: {strategy}"}
        after = self.df.isnull().sum().sum()
        self._log(f"Missing values: {before} → {after} (strategy: {strategy})")
        return {
            "success": True,
            "message": f"Handled missing: {before - after} values filled/dropped",
            "remaining": after,
        }

    def remove_duplicates(self) -> dict:
        self._require_data()
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        self._log(f"Removed {removed} duplicate rows")
        return {
            "success": True,
            "removed": removed,
            "message": f"Removed {removed} duplicate rows",
        }

    def encode_categorical(self, columns: list = None, method: str = "label") -> dict:
        self._require_data()
        if method not in {'label', 'onehot'}:
            return {'success': False, 'message': f'Unknown categorical encoding method: {method}'}
        if self.target_col:
            cols = columns or [
                c
                for c in self.df.select_dtypes(include=["object", "category"]).columns
                if c != self.target_col
            ]
        else:
            cols = columns or list(
                self.df.select_dtypes(include=["object", "category"]).columns
            )
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
        return {
            "success": True,
            "encoded_columns": encoded,
            "message": f"Encoded: {encoded}",
        }

    def remove_constant_columns(self) -> dict:
        self._require_data()
        constant = [
            c
            for c in self.df.columns
            if c != self.target_col and self.df[c].nunique() <= 1
        ]
        self.df.drop(columns=constant, inplace=True)
        self._log(f"Removed {len(constant)} constant columns: {constant}")
        return {"success": True, "removed": constant, "message": f"Removed: {constant}"}

    def remove_high_cardinality(self, threshold: int = 50) -> dict:
        self._require_data()
        high = [
            c
            for c in self.df.select_dtypes(include="object").columns
            if self.df[c].nunique() > threshold and c != self.target_col
        ]
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
        self.df[column] = self.df[column].clip(
            lower=info["lower_bound"], upper=info["upper_bound"]
        )
        self._log(f"Capped outliers in '{column}'")
        return {
            "success": True,
            "message": f"Capped {info['count']} outliers in '{column}'",
        }

    def detect_all_outliers(self) -> pd.DataFrame:
        self._require_data()
        results = []
        for c in self.df.select_dtypes(include=np.number).columns:
            info = self.detect_outliers(c)
            if info["success"]:
                results.append(
                    {
                        "Column": c,
                        "Outliers": info["count"],
                        "Percent": info["pct"],
                        "Lower": info["lower_bound"],
                        "Upper": info["upper_bound"],
                    }
                )
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
            "contamination": contamination,
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
        return {
            "success": True,
            "message": f"Removed {info['count']} outliers",
            "removed": info["count"],
        }

    def fix_dtypes(self):
        self._require_data()
        converted = []
        for col in self.df.select_dtypes(include=["object", "string"]).columns:
            if col == self.target_col:
                continue
            try:
                values = pd.to_numeric(
                    self.df[col].replace(r"^\s*$", np.nan, regex=True), errors="raise"
                )
                self.df[col] = values.replace([np.inf, -np.inf], np.nan)
                converted.append(col)
            except (ValueError, TypeError):
                pass
        return {"success": True, "converted": converted}

    def auto_clean(self, target_col=None):
        self._require_data()
        if target_col and target_col not in self.df:
            raise ValueError("Target column not found.")
        if target_col:
            self.target_col = target_col
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        converted = self.fix_dtypes()["converted"]
        steps = [
            f"Removed {before - len(self.df)} duplicate rows",
            f"Converted numeric text columns: {converted}",
            "Missing values and outliers retained: training imputes features inside each fold; targets are never imputed.",
            "High-cardinality and constant columns retained for explicit review.",
        ]
        self._log(
            "Conservative auto-clean complete; no target imputation or automatic outlier clipping."
        )
        return {"success": True, "steps": steps, "shape": self.df.shape}

    # ------------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------------
    def get_column_types(self) -> dict:
        self._require_data()
        return {
            "numeric": list(self.df.select_dtypes(include=np.number).columns),
            "categorical": list(
                self.df.select_dtypes(include=["object", "category"]).columns
            ),
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
            result = (
                corr[[target]]
                .drop(target)
                .abs()
                .sort_values(target, ascending=False)
                .head(n)
            )
            return result
        pairs = corr.abs().unstack()
        idx = pairs.index.to_frame()
        pair_sets = [frozenset((i, j)) for i, j in idx.values]
        pairs.index = pd.MultiIndex.from_tuples([(p, "") for p in pair_sets])
        pairs = pairs[~pairs.index.duplicated(keep="first")]
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
        return pd.DataFrame(
            {"Value": vc.index, "Count": vc.values, "Percent": pct.values}
        )

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
            obs.append(
                f"{total_miss:,} missing values detected ({miss_pct}%). "
                f"Worst column: '{worst_col}' ({info['missing_pct'][worst_col]}% missing)."
            )
        dups = info["duplicates"]
        obs.append(
            f"{'No duplicate' if dups == 0 else str(dups) + ' duplicate'} rows found."
        )
        num_cols = info["numeric_cols"]
        cat_cols = info["categorical_cols"]
        obs.append(
            f"{len(num_cols)} numeric and {len(cat_cols)} categorical columns detected."
        )
        if num_cols:
            stats = self.get_stats()
            skew = stats['skew'].replace([np.inf, -np.inf], np.nan).dropna()
            if not skew.empty:
                most_skewed = skew.abs().idxmax()
                skew_val = round(skew.loc[most_skewed], 2)
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
            if 0 < target_vals.nunique() <= 20:
                dist = target_vals.value_counts(normalize=True) * 100
                top_class = dist.index[0]
                obs.append(
                    f"Target '{self.target_col}': dominant class is '{top_class}' "
                    f"({dist.iloc[0]:.1f}% of data)."
                )
                if len(dist) == 2:
                    imbalance = dist.iloc[0] / dist.iloc[1]
                    if imbalance > 3:
                        obs.append(
                            f"Target is imbalanced — majority class is {imbalance:.1f}× larger than minority."
                        )
        if cat_cols:
            high_card = [
                (c, self.df[c].nunique()) for c in cat_cols if self.df[c].nunique() > 10
            ]
            if high_card:
                high_card.sort(key=lambda x: -x[1])
                obs.append(
                    f"High-cardinality categorical columns: "
                    f"{', '.join([f'{c}({n})' for c, n in high_card[:3]])}."
                )
        mem = round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        obs.append(f"Dataset uses {mem} MB of memory.")
        return obs[:n]

    # ---- Charts (return base64) ----
    def plot_histogram(self, column: str, bins: int = 30) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(8, 4))
        self.df[column].dropna().plot.hist(
            ax=ax, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85
        )
        ax.set_title(f"Distribution of {column}", fontsize=13, fontweight="bold")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_boxplot(self, column: str) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(8, 4))
        self.df[[column]].dropna().plot.box(
            ax=ax,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#4C72B0", alpha=0.7),
        )
        ax.set_title(f"Box Plot — {column}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_correlation_heatmap(self, max_cols: int = 20) -> str:
        self._require_data()
        num = self.df.select_dtypes(include=np.number)
        if num.shape[1] > max_cols:
            if self.target_col and self.target_col in num.columns:
                top = (
                    num.corr()[self.target_col]
                    .abs()
                    .sort_values(ascending=False)
                    .head(max_cols)
                    .index
                )
                num = num[top]
            else:
                num = num.iloc[:, :max_cols]
        corr = num.corr()
        fig, ax = plt.subplots(
            figsize=(max(8, len(corr) * 0.6), max(6, len(corr) * 0.5))
        )
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 8},
        )
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
                ax.annotate(
                    f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plt.xticks(rotation=30, ha="right")
        else:
            self.df[col].dropna().plot.hist(
                ax=ax, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85
            )
            ax.set_title(f"Target Distribution — {col}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_scatter(self, x_col: str, y_col: str, hue_col: str = None) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(8, 5))
        if hue_col and hue_col in self.df.columns:
            for label, group in self.df.groupby(hue_col):
                ax.scatter(
                    group[x_col], group[y_col], label=str(label), alpha=0.6, s=20
                )
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
        sns.heatmap(
            self.df.isnull(), yticklabels=False, cbar=False, cmap="viridis", ax=ax
        )
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
        plot_df = self.df[
            num_cols + ([hue] if hue and hue not in num_cols else [])
        ].dropna()
        if hue and hue in plot_df.columns:
            plot_df[hue] = plot_df[hue].astype(str)
        fig = sns.pairplot(plot_df, hue=hue, plot_kws={"alpha": 0.4, "s": 15}).fig
        fig.suptitle("Pair Plot", y=1.01, fontsize=13, fontweight="bold")
        return _fig_to_base64(fig)

    # ------------------------------------------------------------------
    # 4. TRAIN MODEL (enhanced with hyperparams, imbalance, feature selection)
    # ------------------------------------------------------------------

    def get_model_recommendation(self, target: str = None) -> dict:
        """Generate recommendation for best model strategy based on problem."""
        self._require_data()
        target_column = target or self.target_col
        recommendation = {
            "problem_type": self.auto_detect_problem_type(target_column)
            if target_column
            else "Unknown",
            "dataset_size": len(self.df),
            "target_col": target_column or "Not set",
            "recommended_models": [],
            "recommended_strategies": [],
            "notes": [],
        }

        if not target_column:
            return recommendation

        problem = recommendation["problem_type"]
        n_rows = len(self.df)

        if problem == "classification":
            recommendation["recommended_models"] = [
                "Random Forest",
                "XGBoost",
                "Gradient Boosting",
            ]
            if target_column in self.df.columns:
                target_dist = self.df[target_column].value_counts(normalize=True)
                if len(target_dist) == 2:
                    ratio = target_dist.iloc[0] / target_dist.iloc[1]
                    if ratio > 3:
                        recommendation["recommended_strategies"].append(
                            "Use SMOTE for imbalanced classification"
                        )
                        recommendation["recommended_strategies"].append(
                            "Set class_weight='balanced'"
                        )
                        recommendation["notes"].append(
                            f"⚠️ Class imbalance detected: {ratio:.1f}:1 ratio"
                        )
            if n_rows < 1000:
                recommendation["notes"].append(
                    "⚠️ Small dataset – consider cross-validation (CV=5 or more)"
                )
        else:
            recommendation["recommended_models"] = [
                "Random Forest",
                "XGBoost",
                "Gradient Boosting",
            ]
            recommendation["notes"].append(
                "ℹ️ Regression task – evaluate using R², RMSE, and MAE"
            )

        if "XGBoost" in model_catalog("classification"):
            recommendation["notes"].append(
                "✓ XGBoost available – often provides best performance"
            )

        return recommendation

        # ========== FEATURE SELECTION METHODS (ADD THESE) ==========

    # ------------------------------------------------------------------
    # 5. EVALUATION (enhanced)
    # ------------------------------------------------------------------

    # ---- Extra evaluation plots ----

    def plot_confusion_matrix(self) -> str:
        cm_data = self.get_confusion_matrix()
        cm = np.array(cm_data["matrix"])
        labels = cm_data["labels"]
        fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
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
            "grade": "A"
            if score >= 90
            else "B"
            if score >= 75
            else "C"
            if score >= 60
            else "D"
            if score >= 40
            else "F",
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
        if health["issues"]:
            insights.append(f"  • Issues: {', '.join(health['issues'][:3])}")
        total_miss = info["total_missing"]
        if total_miss == 0:
            insights.append("  • Data quality: EXCELLENT — no missing values")
        else:
            miss_pct = round(total_miss / (info["rows"] * info["cols"]) * 100, 2)
            insights.append(
                f"  • Missing values: {total_miss:,} cells ({miss_pct}% of all data)"
            )
            worst = max(info["missing_pct"], key=info["missing_pct"].get)
            insights.append(
                f"  • Worst column for missing data: '{worst}' ({info['missing_pct'][worst]}%)"
            )
        dups = info["duplicates"]
        insights.append(
            f"  • Duplicates: {'None found ✓' if dups == 0 else str(dups) + ' duplicate rows detected'}"
        )
        insights.append(f"  • Memory usage: {info['memory_mb']} MB")
        if self.target_col and self.target_col in self.df.columns:
            insights.append("\n🎯 TARGET ANALYSIS")
            target_data = self.df[self.target_col]
            n_unique = target_data.nunique()
            if self.problem_type == "classification":
                dist = target_data.value_counts()
                pcts = (dist / len(self.df) * 100).round(1)
                insights.append("  • Problem type: CLASSIFICATION")
                insights.append(f"  • Classes: {n_unique} unique values")
                for cls, cnt in dist.head(5).items():
                    insights.append(f"    — '{cls}': {cnt:,} samples ({pcts[cls]}%)")
                if n_unique == 2:
                    ratio = dist.iloc[0] / dist.iloc[1]
                    if ratio > 3:
                        insights.append(
                            f"  ⚠️  Imbalanced dataset: majority is {ratio:.1f}× larger — consider SMOTE or class_weight"
                        )
                    else:
                        insights.append(
                            f"  • Class balance: reasonable (ratio {ratio:.2f}:1)"
                        )
            else:
                insights.append("  • Problem type: REGRESSION")
                insights.append(
                    f"  • Target range: {target_data.min():.3f} → {target_data.max():.3f}"
                )
                insights.append(
                    f"  • Target mean: {target_data.mean():.3f} | std: {target_data.std():.3f}"
                )
                skew = round(target_data.skew(), 3)
                if abs(skew) > 1:
                    insights.append(
                        f"  ⚠️  Target is skewed ({skew}) — consider log transform"
                    )
        insights.append("\n🔍 FEATURE ANALYSIS")
        num_cols = info["numeric_cols"]
        cat_cols = info["categorical_cols"]
        insights.append(
            f"  • {len(num_cols)} numeric features, {len(cat_cols)} categorical features"
        )
        if num_cols:
            stats = self.get_stats()
            top_skewed = stats["skew"].abs().sort_values(ascending=False).head(3)
            if not top_skewed.empty:
                sk_str = ", ".join([f"'{c}'({v:.2f})" for c, v in top_skewed.items()])
                insights.append(f"  • Most skewed columns: {sk_str}")
        if cat_cols:
            high_card = [
                (c, self.df[c].nunique()) for c in cat_cols if self.df[c].nunique() > 10
            ]
            if high_card:
                high_card.sort(key=lambda x: -x[1])
                s = ", ".join([f"'{c}'({n})" for c, n in high_card[:3]])
                insights.append(
                    f"  • High-cardinality columns (may need encoding): {s}"
                )
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
                target_corr = (
                    corr[self.target_col]
                    .drop(self.target_col)
                    .abs()
                    .sort_values(ascending=False)
                )
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
                        insights.append(
                            "  • Performance: FAIR (70–80%) — consider tuning"
                        )
                    else:
                        insights.append(
                            "  ⚠️  Performance: POOR (<70%) — needs improvement"
                        )
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
                        insights.append(
                            "  ⚠️  Fit: POOR — try feature engineering or different model"
                        )
        if self.feature_importance:
            insights.append("\n⭐ TOP IMPORTANT FEATURES")
            fi_sorted = sorted(self.feature_importance.items(), key=lambda x: -x[1])
            for feat, imp in fi_sorted[:5]:
                bar = "█" * int(imp * 20)
                insights.append(f"  • {feat}: {imp:.4f} {bar}")
        insights.append("\n💡 RECOMMENDATIONS")
        if total_miss > 0:
            insights.append(
                "  • Handle remaining missing values before deploying the model"
            )
        if self.problem_type == "classification" and self.metrics:
            acc = self.metrics.get("accuracy", 1.0)
            if isinstance(acc, float) and acc < 0.80:
                insights.append(
                    "  • Try Gradient Boosting or XGBoost for better accuracy"
                )
                insights.append(
                    "  • Consider feature engineering (interactions, polynomial features)"
                )
        if self.feature_importance:
            low_fi = [f for f, v in self.feature_importance.items() if v < 0.01]
            if low_fi:
                insights.append(
                    f"  • Consider removing {len(low_fi)} low-importance features to simplify model"
                )
        insights.append("  • Save the pipeline with save_pipeline() for production use")
        self.insights = insights
        return insights

    # ------------------------------------------------------------------
    # 7. PREDICT
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 8. SAVE / LOAD / EXPORT
    # ------------------------------------------------------------------

    def export_cleaned_csv(self, file_path: str) -> dict:
        self._require_data()
        with atomic_target(file_path) as temporary:
            self.df.to_csv(temporary, index=False)
        self._log(f"Cleaned dataset exported to {file_path}")
        return {"success": True, "path": file_path}

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------
    def _require_data(self):
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_csv() or load_file() first.")

    def _require_model(self):
        if self.model is None:
            raise ValueError("Train or import a model first.")

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

# Every successful dataset edit invalidates dependent splits/models; failures roll back.
from functools import wraps


def _dataset_edit(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._require_data()
        if getattr(self, "_editing_dataset", False):
            return method(self, *args, **kwargs)
        original = self.df.copy(deep=True)
        original_target = self.target_col
        original_encoders = dict(self.label_encoders)
        original_log = list(self.log)
        self._editing_dataset = True
        try:
            result = method(self, *args, **kwargs)
            if isinstance(result, dict) and result.get("success") is False:
                self.df = original
                self.target_col = original_target
                self.label_encoders = original_encoders
                self.log = original_log
                return result
            if self.df.empty:
                raise ValueError(
                    "This operation would leave an empty dataset. No changes applied."
                )
        except Exception:
            self.df = original
            self.target_col = original_target
            self.label_encoders = original_encoders
            self.log = original_log
            raise
        finally:
            self._editing_dataset = False
        target = self.target_col
        self._reset_learning()
        self.target_col = target if target in self.df else ""
        self.insights = []
        return result

    return guarded


for _name in [
    "handle_missing",
    "remove_duplicates",
    "encode_categorical",
    "remove_constant_columns",
    "remove_high_cardinality",
    "cap_outliers",
    "remove_outliers_isolation_forest",
    "fix_dtypes",
    "auto_clean",
]:
    setattr(DataYabCore, _name, _dataset_edit(getattr(DataYabCore, _name)))
