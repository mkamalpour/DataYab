"""Visualization module for charts and plots."""

from typing import Optional, Dict, Any, List
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


class Visualizer:
    """Handles all visualization and chart generation."""

    def __init__(self, data_loader, feature_engineer, model_trainer):
        self.dl = data_loader
        self.fe = feature_engineer
        self.mt = model_trainer

    def _require_data(self):
        if self.dl.df is None:
            raise ValueError("No dataset loaded.")

    def _require_model(self):
        if self.mt.model is None:
            raise ValueError("Train or import a model first.")

    def plot_histogram(self, column: str, bins: int = 30) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(8, 4))
        self.dl.df[column].dropna().plot.hist(
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
        self.dl.df[[column]].dropna().plot.box(
            ax=ax, vert=True, patch_artist=True,
            boxprops=dict(facecolor="#4C72B0", alpha=0.7),
        )
        ax.set_title(f"Box Plot — {column}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_correlation_heatmap(self, max_cols: int = 20) -> str:
        self._require_data()
        num = self.dl.df.select_dtypes(include=np.number)
        if num.shape[1] > max_cols:
            if self.fe.target_col and self.fe.target_col in num.columns:
                top = (
                    num.corr()[self.fe.target_col]
                    .abs().sort_values(ascending=False).head(max_cols).index
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
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8},
        )
        ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_target_distribution(self, target: str = None) -> str:
        self._require_data()
        col = target or self.fe.target_col
        if not col or col not in self.dl.df.columns:
            raise ValueError("Target column not set or not found")
        fig, ax = plt.subplots(figsize=(8, 4))
        if self.dl.df[col].nunique() <= 20:
            vc = self.dl.df[col].value_counts()
            colors = sns.color_palette("husl", len(vc))
            vc.plot.bar(ax=ax, color=colors, edgecolor="white")
            ax.set_title(f"Target Distribution — {col}", fontsize=13, fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9,
                )
            plt.xticks(rotation=30, ha="right")
        else:
            self.dl.df[col].dropna().plot.hist(
                ax=ax, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85
            )
            ax.set_title(f"Target Distribution — {col}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_scatter(self, x_col: str, y_col: str, hue_col: str = None) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(8, 5))
        if hue_col and hue_col in self.dl.df.columns:
            for label, group in self.dl.df.groupby(hue_col):
                ax.scatter(
                    group[x_col], group[y_col], label=str(label), alpha=0.6, s=20
                )
            ax.legend(title=hue_col)
        else:
            ax.scatter(self.dl.df[x_col], self.dl.df[y_col], alpha=0.5, color="#4C72B0", s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_missing_heatmap(self) -> str:
        self._require_data()
        fig, ax = plt.subplots(figsize=(max(8, self.dl.df.shape[1] * 0.4), 5))
        sns.heatmap(
            self.dl.df.isnull(), yticklabels=False, cbar=False, cmap="viridis", ax=ax
        )
        ax.set_title("Missing Values Map", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_feature_importance(self) -> str:
        if not self.mt.feature_importance:
            raise ValueError("Train a model first to get feature importances.")
        fi = dict(sorted(self.mt.feature_importance.items(), key=lambda x: x[1], reverse=True))
        top = dict(list(fi.items())[:20])
        fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.35)))
        colors = sns.color_palette("Blues_r", len(top))
        ax.barh(list(top.keys())[::-1], list(top.values())[::-1], color=colors[::-1])
        ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_pairplot(self, columns: List[str] = None, hue: str = None) -> str:
        self._require_data()
        num_cols = columns or list(self.dl.df.select_dtypes(include=np.number).columns[:5])
        plot_df = self.dl.df[num_cols + ([hue] if hue and hue not in num_cols else [])].dropna()
        if hue and hue in plot_df.columns:
            plot_df[hue] = plot_df[hue].astype(str)
        fig = sns.pairplot(plot_df, hue=hue, plot_kws={"alpha": 0.4, "s": 15}).fig
        fig.suptitle("Pair Plot", y=1.01, fontsize=13, fontweight="bold")
        return _fig_to_base64(fig)

    def plot_confusion_matrix(self) -> str:
        self._require_model()
        if self.fe.problem_type != "classification" or self.fe._raw_test is None:
            raise ValueError("A classification holdout is required.")
        self.mt._compute_predictions()
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            self.fe.y_test, self.mt.y_pred,
            labels=np.arange(len(self.fe.target_encoder.classes_))
        )
        labels = self.fe.target_encoder.classes_.tolist()
        fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_roc_curve(self) -> str:
        self._require_model()
        if (
            self.fe.problem_type != "classification"
            or self.mt.y_proba is None
            or self.mt.y_proba.shape[1] != 2
        ):
            raise ValueError("This curve requires binary classification with probability estimates.")
        fpr, tpr, _ = roc_curve(self.fe.y_test, self.mt.y_proba[:, 1])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            title="Holdout ROC",
        )
        fig.tight_layout()
        return _fig_to_base64(fig)

    def plot_precision_recall_curve(self) -> str:
        self._require_model()
        if (
            self.fe.problem_type != "classification"
            or self.mt.y_proba is None
            or self.mt.y_proba.shape[1] != 2
        ):
            raise ValueError("This curve requires binary classification with probability estimates.")
        precision, recall, _ = precision_recall_curve(self.fe.y_test, self.mt.y_proba[:, 1])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(recall, precision)
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title="Holdout precision-recall",
        )
        fig.tight_layout()
        return _fig_to_base64(fig)

    def plot_residuals(self) -> str:
        self._require_model()
        if self.fe.problem_type != "regression":
            raise ValueError("Residuals require regression.")
        self.mt._compute_predictions()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(self.mt.y_pred, self.fe.y_test - self.mt.y_pred, alpha=0.6)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set(xlabel="Prediction", ylabel="Residual", title="Holdout residuals")
        fig.tight_layout()
        return _fig_to_base64(fig)

    def plot_learning_curve(self, train_sizes: np.ndarray = None) -> str:
        """Plot learning curve for the current model."""
        from sklearn.model_selection import learning_curve
        self._require_model()
        if self.fe._raw_train is None:
            raise ValueError("Need training data for learning curve.")

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, test_scores = learning_curve(
            self.mt.model, self.fe._raw_train, self.fe.y_train,
            train_sizes=train_sizes, cv=5, n_jobs=1, scoring="accuracy" if self.fe.problem_type == "classification" else "r2"
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes_abs, train_mean, 'o-', color="blue", label="Training score")
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        ax.plot(train_sizes_abs, test_mean, 'o-', color="red", label="Cross-validation score")
        ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.set_title("Learning Curve", fontsize=13, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_validation_curve(self, param_name: str, param_range: np.ndarray) -> str:
        """Plot validation curve for a hyperparameter."""
        from sklearn.model_selection import validation_curve
        self._require_model()
        if self.fe._raw_train is None:
            raise ValueError("Need training data for validation curve.")

        train_scores, test_scores = validation_curve(
            self.mt.model, self.fe._raw_train, self.fe.y_train,
            param_name=param_name, param_range=param_range,
            cv=5, n_jobs=1, scoring="accuracy" if self.fe.problem_type == "classification" else "r2"
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(param_range, train_mean, 'o-', color="blue", label="Training score")
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        ax.plot(param_range, test_mean, 'o-', color="red", label="Cross-validation score")
        ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Score")
        ax.set_title(f"Validation Curve: {param_name}", fontsize=13, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return _fig_to_base64(fig)

    def plot_partial_dependence(self, features: List[str], kind: str = "average") -> str:
        """Plot partial dependence plots."""
        from sklearn.inspection import PartialDependenceDisplay
        self._require_model()
        if self.fe._raw_train is None:
            raise ValueError("Need training data for partial dependence.")

        # Map feature names to indices
        feature_indices = [self.fe.feature_names.index(f) for f in features if f in self.fe.feature_names]
        if not feature_indices:
            raise ValueError("No valid features found for partial dependence.")

        fig, ax = plt.subplots(figsize=(10, 6))
        display = PartialDependenceDisplay.from_estimator(
            self.mt.model, self.fe._raw_train, feature_indices,
            feature_names=self.fe.feature_names, kind=kind, ax=ax, n_jobs=1
        )
        ax.set_title("Partial Dependence Plots", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _fig_to_base64(fig)

    def get_stats(self) -> pd.DataFrame:
        self._require_data()
        return self.dl.get_stats()

    def get_correlation_matrix(self) -> pd.DataFrame:
        self._require_data()
        return self.dl.df.select_dtypes(include=np.number).corr()

    def get_top_correlations(self, target: str = None, n: int = 15) -> pd.DataFrame:
        self._require_data()
        corr = self.get_correlation_matrix()
        if target and target in corr.columns:
            return (
                corr[[target]].drop(target).abs()
                .sort_values(target, ascending=False).head(n)
            )
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
        vc = self.dl.df[column].value_counts().head(top_n)
        pct = (vc / len(self.dl.df) * 100).round(2)
        return pd.DataFrame({"Value": vc.index, "Count": vc.values, "Percent": pct.values})
