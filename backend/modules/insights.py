"""Insights generation module."""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd


class InsightsGenerator:
    """Generates comprehensive insights from data and models."""

    def __init__(self, data_loader, feature_engineer, model_trainer):
        self.dl = data_loader
        self.fe = feature_engineer
        self.mt = model_trainer

    def _require_data(self):
        if self.dl.df is None:
            raise ValueError("No dataset loaded.")

    def generate_observations(self, n: int = 10) -> List[str]:
        self._require_data()
        obs = []
        info = self.dl.get_info()
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
            stats = self.dl.get_stats()
            skew = stats['skew'].replace([np.inf, -np.inf], np.nan).dropna()
            if not skew.empty:
                most_skewed = skew.abs().idxmax()
                skew_val = round(skew.loc[most_skewed], 2)
                obs.append(f"Most skewed column: '{most_skewed}' (skew={skew_val}).")
        if len(num_cols) >= 2:
            corr = self.dl.get_correlation_matrix()
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
        if self.fe.target_col and self.fe.target_col in self.dl.df.columns:
            target_vals = self.dl.df[self.fe.target_col]
            if 0 < target_vals.nunique() <= 20:
                dist = target_vals.value_counts(normalize=True) * 100
                top_class = dist.index[0]
                obs.append(
                    f"Target '{self.fe.target_col}': dominant class is '{top_class}' "
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
                (c, self.dl.df[c].nunique()) for c in cat_cols if self.dl.df[c].nunique() > 10
            ]
            if high_card:
                high_card.sort(key=lambda x: -x[1])
                obs.append(
                    f"High-cardinality categorical columns: "
                    f"{', '.join([f'{c}({n})' for c, n in high_card[:3]])}."
                )
        mem = round(self.dl.df.memory_usage(deep=True).sum() / 1024**2, 2)
        obs.append(f"Dataset uses {mem} MB of memory.")
        return obs[:n]

    def get_dataset_health_score(self) -> Dict:
        self._require_data()
        score = 100
        issues = []
        missing_pct = self.dl.df.isnull().sum().sum() / self.dl.df.size * 100
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
        duplicate_pct = self.dl.df.duplicated().sum() / len(self.dl.df) * 100
        if duplicate_pct > 10:
            score -= 15
            issues.append(f"High duplicates: {duplicate_pct:.1f}%")
        elif duplicate_pct > 0:
            score -= 5
        const_cols = [c for c in self.dl.df.columns if self.dl.df[c].nunique() <= 1]
        if const_cols:
            score -= len(const_cols) * 2
        return {
            "score": max(0, min(100, score)),
            "grade": "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F",
            "issues": issues,
            "total_issues": len(issues),
        }

    def generate_insights(self) -> List[str]:
        self._require_data()
        insights = []
        info = self.dl.get_info()
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

        if self.fe.target_col and self.fe.target_col in self.dl.df.columns:
            insights.append("\n🎯 TARGET ANALYSIS")
            target_data = self.dl.df[self.fe.target_col]
            n_unique = target_data.nunique()
            problem_type = self.fe.problem_type or self._auto_detect_problem_type(self.fe.target_col)
            if problem_type == "classification":
                dist = target_data.value_counts()
                pcts = (dist / len(self.dl.df) * 100).round(1)
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
            stats = self.dl.get_stats()
            top_skewed = stats["skew"].abs().sort_values(ascending=False).head(3)
            if not top_skewed.empty:
                sk_str = ", ".join([f"'{c}'({v:.2f})" for c, v in top_skewed.items()])
                insights.append(f"  • Most skewed columns: {sk_str}")
        if cat_cols:
            high_card = [
                (c, self.dl.df[c].nunique()) for c in cat_cols if self.dl.df[c].nunique() > 10
            ]
            if high_card:
                high_card.sort(key=lambda x: -x[1])
                s = ", ".join([f"'{c}'({n})" for c, n in high_card[:3]])
                insights.append(
                    f"  • High-cardinality columns (may need encoding): {s}"
                )

        if len(num_cols) >= 2:
            insights.append("\n📈 CORRELATION INSIGHTS")
            corr = self.dl.get_correlation_matrix()
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
            if self.fe.target_col and self.fe.target_col in corr.columns:
                target_corr = (
                    corr[self.fe.target_col].drop(self.fe.target_col).abs().sort_values(ascending=False)
                )
                top_feat = target_corr.head(3)
                insights.append(f"  • Top predictors of '{self.fe.target_col}':")
                for feat, val in top_feat.items():
                    insights.append(f"    — '{feat}': |r|={val:.3f}")

        if self.mt.model and self.mt.metrics:
            insights.append(f"\n🤖 MODEL PERFORMANCE ({self.mt.model_name})")
            if self.fe.problem_type == "classification":
                acc = self.mt.metrics.get("accuracy", "N/A")
                f1 = self.mt.metrics.get("f1", "N/A")
                auc = self.mt.metrics.get("roc_auc", "N/A")
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
                r2 = self.mt.metrics.get("r2", "N/A")
                rmse = self.mt.metrics.get("rmse", "N/A")
                mae = self.mt.metrics.get("mae", "N/A")
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

        if self.mt.feature_importance:
            insights.append("\n⭐ TOP IMPORTANT FEATURES")
            fi_sorted = sorted(self.mt.feature_importance.items(), key=lambda x: -x[1])
            for feat, imp in fi_sorted[:5]:
                bar = "█" * int(imp * 20)
                insights.append(f"  • {feat}: {imp:.4f} {bar}")

        insights.append("\n💡 RECOMMENDATIONS")
        if total_miss > 0:
            insights.append("  • Handle remaining missing values before deploying the model")
        if self.fe.problem_type == "classification" and self.mt.metrics:
            acc = self.mt.metrics.get("accuracy", 1.0)
            if isinstance(acc, float) and acc < 0.80:
                insights.append("  • Try Gradient Boosting or XGBoost for better accuracy")
                insights.append("  • Consider feature engineering (interactions, polynomial features)")
        if self.mt.feature_importance:
            low_fi = [f for f, v in self.mt.feature_importance.items() if v < 0.01]
            if low_fi:
                insights.append(
                    f"  • Consider removing {len(low_fi)} low-importance features to simplify model"
                )
        insights.append("  • Save the pipeline with save_pipeline() for production use")
        return insights

    def _auto_detect_problem_type(self, target: str) -> str:
        col = self.dl.df[target].dropna()
        if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            return "classification"
        if col.nunique() <= 20 and np.equal(col, np.floor(col)).all():
            return "classification"
        return "regression"

    def generate_model_card(self) -> Dict:
        """Generate a comprehensive model card for documentation."""
        if not self.mt.model:
            raise ValueError("No trained model available.")

        card = {
            "model_name": self.mt.model_name,
            "problem_type": self.fe.problem_type,
            "target_column": self.fe.target_col,
            "training_data_shape": self.fe._raw_train.shape if self.fe._raw_train is not None else None,
            "test_data_shape": self.fe._raw_test.shape if self.fe._raw_test is not None else None,
            "features_used": self.fe._selected_input_names,
            "feature_count": len(self.fe._selected_input_names),
            "extraction_method": self.fe._extraction_method,
            "extraction_components": self.fe._extraction_components,
            "metrics": self.mt.metrics,
            "feature_importance": self.mt.feature_importance,
            "cross_validation": self.mt.get_cross_val_score(cv=5) if self.mt.model else None,
            "preprocessing": {
                "scale": self.fe._scale,
                "scaler_type": self.fe._scaler_type,
                "missing_strategy": "median (numeric) / constant (categorical)",
                "encoding": "ordinal (handle_unknown=use_encoded_value)",
            },
            "warnings": [],
        }

        # Add warnings
        if self.fe._selection_data_driven:
            card["warnings"].append(
                "Feature selection was data-driven on training set; CV is conditional on these choices."
            )
        if self.fe.problem_type == "classification" and self.mt.metrics:
            acc = self.mt.metrics.get("accuracy", 1.0)
            if isinstance(acc, float) and acc < 0.7:
                card["warnings"].append("Model accuracy below 70% - consider retraining with different features/models.")

        return card
