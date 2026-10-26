import pandas as pd
import numpy as np
import json
import time
from pathlib import Path


def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(value) for value in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    if pd.isna(obj):
        return None
    return obj


def ultimate_profile(file_path):
    """Generate comprehensive dataset profile."""
    path = Path(file_path)
    ext = path.suffix.lower()

    print(f"📂 Loading: {path.name}...")
    start_time = time.time()

    # Check file size warning
    file_size_mb = path.stat().st_size / 1024**2
    if file_size_mb > 100:
        print(f"⚠️ Large file ({file_size_mb:.1f} MB) - profiling may take time...")

    # Load based on file extension
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        raise

    # Check for empty dataset
    if df.empty:
        raise ValueError("Dataset is empty (0 rows)")

    print(
        f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns in {time.time() - start_time:.2f}s"
    )

    profile = {
        "dataset_identity": {
            "name": path.stem,
            "format": ext,
            "full_rows": len(df),
            "full_columns": len(df.columns),
            "full_cells": len(df) * len(df.columns),
            "file_size_mb": round(file_size_mb, 2),
        },
        "quality_metrics": {
            "total_missing_cells": int(df.isna().sum().sum()),
            "missing_percent": round(
                100 * df.isna().sum().sum() / (len(df) * len(df.columns)), 2
            ),
            "total_duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percent": round(100 * df.duplicated().sum() / len(df), 2),
            "complete_rows": int((~df.isna().any(axis=1)).sum()),
            "complete_rows_percent": round(
                100 * (~df.isna().any(axis=1)).sum() / len(df), 2
            ),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        },
        "columns": {},
        "relationships": {"potential_primary_keys": [], "highly_correlated_pairs": []},
    }

    print("🔍 Analyzing columns...")

    for idx, col in enumerate(df.columns):
        if idx % 10 == 0:
            print(f"   Progress: {idx}/{len(df.columns)} columns...")

        col_data = df[col]
        non_null = col_data.dropna()

        col_info = {
            "basic": {
                "dtype": str(col_data.dtype),
                "nulls": int(col_data.isna().sum()),
                "null_pct": round(100 * col_data.isna().sum() / len(df), 2),
                "unique_count": int(col_data.nunique()),
                "unique_pct": round(100 * col_data.nunique() / len(df), 2),
                "cardinality": "high"
                if col_data.nunique() > 100
                else "medium"
                if col_data.nunique() > 20
                else "low",
            },
            "uniqueness": {
                "is_perfect_id": col_data.nunique() == len(df)
                and col_data.isna().sum() == 0,
                "is_completely_constant": col_data.nunique() == 1,
                "has_duplicates": col_data.duplicated().sum() > 0,
            },
        }

        # Sample values (critical for understanding content)
        if len(non_null) > 0:
            samples = non_null.head(5).tolist()
            samples = [
                str(x) if not isinstance(x, (int, float, str, bool)) else x
                for x in samples
            ]
            col_info["sample_values"] = samples

        # Top values for low cardinality columns
        if col_data.nunique() <= 30 and len(non_null) > 0:
            top = col_data.value_counts().head(5).to_dict()
            col_info["top_values"] = {str(k): v for k, v in top.items()}

        # Numeric column statistics
        if (pd.api.types.is_numeric_dtype(col_data)
                and not pd.api.types.is_bool_dtype(col_data) and len(non_null) > 0):
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1

            col_info["numeric"] = {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "median": float(non_null.median()),
                "std": float(non_null.std()),
                "q1": float(q1),
                "q3": float(q3),
                "skewness": float(non_null.skew()),
                "negatives": int((non_null < 0).sum()),
                "zeros": int((non_null == 0).sum()),
                "outliers_low": int((non_null < (q1 - 1.5 * iqr)).sum()),
                "outliers_high": int((non_null > (q3 + 1.5 * iqr)).sum()),
            }

        # Datetime column statistics
        elif pd.api.types.is_datetime64_any_dtype(col_data) and len(non_null) > 0:
            col_info["datetime"] = {
                "min": non_null.min().isoformat(),
                "max": non_null.max().isoformat(),
                "range_days": (non_null.max() - non_null.min()).days
                if len(non_null) > 1
                else 0,
            }

        # Text column statistics
        elif len(non_null) > 0:
            non_null_str = non_null.astype(str)
            col_info["text"] = {
                "min_length": int(non_null_str.str.len().min()),
                "max_length": int(non_null_str.str.len().max()),
                "mean_length": round(non_null_str.str.len().mean(), 2),
            }

        profile["columns"][col] = col_info

        # Detect primary key candidates
        if col_info["uniqueness"]["is_perfect_id"]:
            profile["relationships"]["potential_primary_keys"].append(col)

    # Correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        if len(numeric_cols) > 50:
            print(
                f"⚠️ Computing correlations for {len(numeric_cols)} numeric columns - may take a moment..."
            )

        print("📊 Computing correlations...")
        corr_matrix = df[numeric_cols].corr().abs()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > 0.8:
                    profile["relationships"]["highly_correlated_pairs"].append(
                        {
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "correlation": round(corr_val, 3),
                        }
                    )

    # Add summary statistics
    profile["summary"] = {
        "numeric_columns_count": len(numeric_cols),
        "categorical_columns_count": len(
            df.select_dtypes(include=["object", "category"]).columns
        ),
        "datetime_columns_count": len(df.select_dtypes(include=["datetime64"]).columns),
        "high_correlation_pairs_count": len(
            profile["relationships"]["highly_correlated_pairs"]
        ),
        "primary_key_candidates_count": len(
            profile["relationships"]["potential_primary_keys"]
        ),
        "profile_time_seconds": round(time.time() - start_time, 2),
    }

    print(f"✅ Profiling complete in {profile['summary']['profile_time_seconds']}s")
    return convert_to_serializable(profile)


def print_quick_summary(profile):
    """Print a human-readable quick summary."""
    print("\n" + "=" * 50)
    print("📊 QUICK SUMMARY")
    print("=" * 50)
    print(f"   Name: {profile['dataset_identity']['name']}")
    print(f"   Rows: {profile['dataset_identity']['full_rows']:,}")
    print(f"   Columns: {profile['dataset_identity']['full_columns']}")
    print(f"   File size: {profile['dataset_identity']['file_size_mb']} MB")
    print(f"   Missing: {profile['quality_metrics']['missing_percent']}%")
    print(f"   Duplicates: {profile['quality_metrics']['duplicate_percent']}%")
    print(f"   Complete rows: {profile['quality_metrics']['complete_rows_percent']}%")
    print(f"   Numeric columns: {profile['summary']['numeric_columns_count']}")
    print(f"   Categorical columns: {profile['summary']['categorical_columns_count']}")
    print(
        f"   Primary key candidates: {profile['summary']['primary_key_candidates_count']}"
    )
    print(f"   High correlations: {profile['summary']['high_correlation_pairs_count']}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a detailed JSON dataset profile."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new filename.")
    profile = ultimate_profile(args.dataset)
    args.output.write_text(
        json.dumps(profile, indent=2, default=convert_to_serializable, allow_nan=False), encoding="utf-8"
    )
    print_quick_summary(profile)
