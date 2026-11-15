"""Data loading and basic exploration module."""

import io
import base64
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

from ..storage import atomic_target


def _array_to_frame(value: np.ndarray) -> pd.DataFrame:
    """Turn a NumPy array into a tabular DataFrame."""
    if isinstance(value, np.ndarray) and value.dtype.names:
        return pd.DataFrame.from_records(value)
    array = np.asarray(value)
    if array.ndim == 1:
        return pd.DataFrame({"value": array})
    if array.ndim == 2:
        return pd.DataFrame(
            array, columns=[f"column_{i}" for i in range(1, array.shape[1] + 1)]
        )
    raise ValueError("Array files must contain a 1D or 2D array.")


def _arff_frame(file_path: Path) -> pd.DataFrame:
    """Read a Weka ARFF file."""
    data, _ = arff.loadarff(file_path)
    frame = pd.DataFrame(data)
    for column in frame.columns:
        values = frame[column].map(
            lambda v: v.decode("utf-8", "replace") if isinstance(v, bytes) else v
        ).replace("?", np.nan)
        numeric = pd.to_numeric(values, errors="coerce")
        frame[column] = numeric if values.notna().sum() == numeric.notna().sum() else values
    return frame


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _arff_escape(text: Any) -> str:
    value = str(text).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{value}'"


def _arff_token(value: Any) -> str:
    text = str(value)
    unsafe = not text or any(ch in text for ch in " ,{}%'\"\\\t\n")
    return _arff_escape(text) if unsafe else text


def _arff_cell(value: Any) -> str:
    if pd.isna(value):
        return "?"
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value)) if np.isfinite(value) else "?"
    return _arff_token(value)


def _arff_nominal(values: pd.Series) -> str:
    tokens = []
    for value in values.dropna().unique():
        token = _arff_token(value)
        if token not in tokens:
            tokens.append(token)
    return "{" + ",".join(tokens) + "}" if tokens else "{'Unknown'}"


class DataLoader:
    """Handles all data loading operations."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.file_path: str = ""
        self.log: list[str] = []

    def _log(self, message: str):
        self.log.append(message)
        if len(self.log) > 200:
            self.log = self.log[-200:]

    def _set_dataset(self, df: pd.DataFrame, path: str = "") -> dict:
        """Validate and set the dataset."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Dataset must be a non-empty DataFrame.")
        if len(df.columns) > 2000:
            raise ValueError("Maximum 2000 columns allowed.")
        if len(df) > 1_000_000:
            raise ValueError("Maximum 1,000,000 rows allowed.")

        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        if not df.columns.is_unique:
            raise ValueError("Column names must be unique.")

        for col in df.select_dtypes(include="object"):
            if df[col].map(lambda v: isinstance(v, (list, dict, set, tuple, np.ndarray))).any():
                raise ValueError(f"Column '{col}' contains Nested values. Flatten first.")

        if not df.columns.is_unique:
            raise ValueError("Column names must be unique.")
        if any(not c.strip() for c in df.columns):
            raise ValueError("Column names cannot be blank.")

        self.df = df.reset_index(drop=True)
        self.original_df = self.df.copy(deep=True)
        self.file_path = str(path)
        self._log(f"Loaded {len(df):,} rows / {len(df.columns)} columns")
        callback = getattr(self, "on_dataset_changed", None)
        if callback is not None:
            callback()

        return {
            "success": True,
            "message": f"Loaded {len(df):,} rows / {len(df.columns)} columns",
            "preview": self.df.head(10),
            "info": self.get_info(),
        }

    def load_csv(self, file_path: str, sep: Optional[str] = None) -> dict:
        import csv
        path = Path(file_path)
        try:
            for encoding in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    delimiter = sep
                    if delimiter is None:
                        with open(path, encoding=encoding, newline="") as stream:
                            sample = stream.read(65536)
                        try:
                            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                        except csv.Error:
                            delimiter = ","
                    return self._set_dataset(
                        pd.read_csv(path, sep=delimiter, encoding=encoding), str(path)
                    )
                except UnicodeDecodeError:
                    continue
            raise ValueError("Cannot decode this CSV.")
        except Exception as e:
            return {"success": False, "message": f"CSV load failed: {e}"}

    def load_excel(self, file_path: str, sheet_name: Any = 0) -> dict:
        try:
            return self._set_dataset(
                pd.read_excel(file_path, sheet_name=sheet_name), file_path
            )
        except Exception as e:
            return {"success": False, "message": f"Excel load failed: {e}"}

    def load_json(self, file_path: str) -> dict:
        try:
            import json
            with open(file_path, encoding="utf-8-sig") as stream:
                value = json.load(stream)
            if isinstance(value, dict) and len(value) == 1 and isinstance(next(iter(value.values())), list):
                value = next(iter(value.values()))
            df = pd.json_normalize(value) if isinstance(value, list) else pd.DataFrame(value)
            return self._set_dataset(df, file_path)
        except Exception as e:
            return {"success": False, "message": f"JSON load failed: {e}"}

    def load_parquet(self, file_path: str) -> dict:
        try:
            return self._set_dataset(pd.read_parquet(file_path), file_path)
        except Exception as e:
            return {"success": False, "message": f"Parquet load failed: {e}"}

    def load_arr(self, file_path: str) -> dict:
        try:
            with open(file_path, "rb") as stream:
                magic = stream.read(8)
            path = Path(file_path)
            if magic.startswith(b"\x93NUMPY"):
                return self._set_dataset(_array_to_frame(np.load(path, allow_pickle=False)), str(path))
            if magic.startswith(b"ARROW1") or magic.startswith(b"FEA1"):
                import pyarrow as pa
                if magic.startswith(b"FEA1"):
                    return self._set_dataset(pd.read_feather(path), str(path))
                try:
                    with pa.memory_map(str(path), "r") as source:
                        table = pa.ipc.open_file(source).read_all()
                except Exception:
                    return self._set_dataset(pd.read_feather(path), str(path))
                return self._set_dataset(table.to_pandas(), str(path))
            result = self.load_csv(file_path)
            if not result["success"]:
                detail = result.get("message", "").split(": ", 1)[-1]
                return {"success": False, "message": f"Array load failed: {detail}"}
            return result
        except Exception as e:
            return {"success": False, "message": f"Array load failed: {e}"}

    def load_arff(self, file_path: str) -> dict:
        try:
            return self._set_dataset(_arff_frame(Path(file_path)), file_path)
        except Exception as e:
            return {"success": False, "message": f"ARFF load failed: {e}"}

    def load_file(self, file_path: str) -> dict:
        loader_map = {
            ".csv": self.load_csv,
            ".xlsx": self.load_excel,
            ".xls": self.load_excel,
            ".json": self.load_json,
            ".parquet": self.load_parquet,
            ".arr": self.load_arr,
            ".arff": self.load_arff,
        }
        loader = loader_map.get(Path(file_path).suffix.lower())
        if loader:
            return loader(file_path)
        return {
            "success": False,
            "message": "Supported: CSV, XLSX, XLS, JSON, Parquet, ARR, ARFF."
        }

    def reset_to_original(self) -> dict:
        if self.original_df is None:
            raise ValueError("No original dataset available.")
        self._set_dataset(self.original_df.copy(deep=True), self.file_path)
        return {"success": True, "message": "Restored original data."}

    def get_info(self) -> dict:
        if self.df is None:
            raise ValueError("No dataset loaded.")
        df = self.df
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        missing = df.isnull().sum()
        return {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "missing_count": missing.to_dict(),
            "missing_pct": (missing / len(df) * 100).round(2).to_dict(),
            "total_missing": int(missing.sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(memory_mb, 3),
            "numeric_cols": list(df.select_dtypes(include=np.number).columns),
            "categorical_cols": list(df.select_dtypes(include=["object", "category"]).columns),
            "datetime_cols": list(df.select_dtypes(include=["datetime64"]).columns),
        }

    def get_preview(self, rows: int = 10) -> dict:
        if self.df is None:
            raise ValueError("No dataset loaded.")
        frame = self.df.head(rows)
        return {
            "success": True,
            "columns": list(frame.columns),
            "data": frame.values.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        }

    def get_stats(self) -> pd.DataFrame:
        """Get descriptive statistics for numeric columns."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        stats = self.df.describe(include="all").transpose()
        stats["skew"] = np.nan
        stats["kurtosis"] = np.nan
        stats["missing"] = self.df.isna().sum()
        stats["missing_pct"] = (stats["missing"] / len(self.df) * 100).round(2)
        # Add skew for numeric columns
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            skew_vals = self.df[numeric_cols].skew()
            stats.loc[skew_vals.index, 'skew'] = skew_vals.values
            stats.loc[numeric_cols, 'kurtosis'] = self.df[numeric_cols].kurtosis().values
        return stats

    def get_column_types(self) -> dict:
        if self.df is None:
            raise ValueError("No dataset loaded.")
        return {
            "numeric": list(self.df.select_dtypes(include=np.number).columns),
            "categorical": list(self.df.select_dtypes(include=["object", "category"]).columns),
            "datetime": list(self.df.select_dtypes(include=["datetime64"]).columns),
        }

    def get_correlation_matrix(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No dataset loaded.")
        return self.df.select_dtypes(include=np.number).corr()

    def export_cleaned_csv(self, file_path: str) -> dict:
        if self.df is None:
            raise ValueError("No dataset loaded.")
        with atomic_target(file_path) as temporary:
            self.df.to_csv(temporary, index=False)
        self._log(f"Cleaned dataset exported to {file_path}")
        return {"success": True, "path": file_path}

    def export_cleaned_arff(self, file_path: str, relation: str = "datayab_dataset") -> dict:
        if self.df is None:
            raise ValueError("No dataset loaded.")
        with atomic_target(file_path) as temporary:
            lines = [f"@relation {_arff_token(relation)}", ""]
            for column in self.df.columns:
                numeric = pd.api.types.is_numeric_dtype(self.df[column]) and not pd.api.types.is_bool_dtype(self.df[column])
                kind = "numeric" if numeric else _arff_nominal(self.df[column])
                lines.append(f"@attribute {_arff_token(column)} {kind}")
            lines.extend(["", "@data"])
            for row in self.df.itertuples(index=False, name=None):
                lines.append(",".join(_arff_cell(value) for value in row))
            temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._log(f"Cleaned ARFF exported to {file_path}")
        return {"success": True, "path": file_path}

    def export_cleaned_data(self, file_path: str) -> dict:
        if Path(file_path).suffix.lower() == ".arff":
            return self.export_cleaned_arff(file_path)
        return self.export_cleaned_csv(file_path)

    def get_log(self) -> list[str]:
        return self.log.copy()
