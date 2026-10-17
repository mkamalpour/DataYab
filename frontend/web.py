"""Local, serialized DataYab web workspace with bounded temporary uploads."""

from pathlib import Path
from functools import wraps
import threading
import tempfile
import math
import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from backend.core import DataYabCore

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 50 * 1024 * 1024
core = DataYabCore()
state_lock = threading.RLock()
app = FastAPI(title="DataYab", version="2.0.0")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.middleware("http")
async def local_request_guard(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin not in {
        f"http://{request.headers.get('host')}",
        f"https://{request.headers.get('host')}",
    }:
        return JSONResponse(
            {"detail": "Cross-origin access is not allowed."}, status_code=403
        )
    if request.headers.get("sec-fetch-site") == "cross-site":
        return JSONResponse(
            {"detail": "Cross-site access is not allowed."}, status_code=403
        )
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_UPLOAD + 1024 * 1024:
                return JSONResponse(
                    {"detail": "Upload limit is 50 MB."}, status_code=413
                )
        except ValueError:
            return JSONResponse({"detail": "Invalid content length."}, status_code=400)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


def convert_to_native(value):
    if isinstance(value, dict):
        return {str(k): convert_to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [convert_to_native(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)):
        return value.isoformat()
    return value


def workspace(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with state_lock:
            try:
                result = function(*args, **kwargs)
                if isinstance(result, dict) and result.get("success") is False:
                    raise HTTPException(400, result.get("message", "Operation failed."))
                return convert_to_native(result)
            except HTTPException:
                raise
            except (ValueError, TypeError, KeyError, IndexError) as error:
                raise HTTPException(400, str(error)) from error

    return guarded


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = Field(min_length=1)
    model: str = "Random Forest"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    problem_type: Literal["auto", "classification", "regression"] = "auto"
    scale: bool = True
    scaler_type: Literal["standard", "minmax"] = "standard"
    use_smote: bool = False
    class_weight: Literal["balanced", "none"] | None = None
    n_estimators: int | None = Field(default=None, ge=1, le=2000)
    max_depth: int | None = Field(default=None, ge=1, le=100)
    learning_rate: float | None = Field(default=None, gt=0, le=10)
    C: float | None = Field(default=None, gt=0, le=100000)
    n_neighbors: int | None = Field(default=None, ge=1, le=1000)


def prepare(request):
    return core.split_data(
        request.target,
        test_size=request.test_size,
        scale=request.scale,
        scaler_type=request.scaler_type,
        problem_type=request.problem_type,
    )


@app.get("/")
def home():
    return FileResponse(ROOT / "templates/index.html", media_type="text/html")


@app.get("/api/status")
@workspace
def status():
    if core.df is None:
        return {
            "loaded": False,
            "message": "No dataset loaded",
            "model": core.model_name,
        }
    info = core.get_info()
    return {
        "loaded": True,
        "rows": info["rows"],
        "cols": info["cols"],
        "columns": info["columns"],
        "target": core.target_col,
        "model": core.model_name,
        "problem_type": core.problem_type,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls", ".json", ".parquet"):
        await file.close()
        raise HTTPException(400, "Supported files: CSV, Excel, JSON, Parquet.")
    try:
        with tempfile.TemporaryDirectory(prefix="datayab-upload-") as directory:
            path = Path(directory) / ("dataset" + suffix)
            size = 0
            with path.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "Upload limit is 50 MB.")
                    stream.write(chunk)
            if not size:
                raise HTTPException(400, "The uploaded file is empty.")

            def install():
                with state_lock:
                    result = core.load_file(path)
                    if not result["success"]:
                        raise HTTPException(400, result["message"])
                    core.file_path = Path(
                        (file.filename or "dataset").replace("\\", "/")
                    ).name
                    return {
                        "success": True,
                        "message": result["message"],
                        "shape": f"{len(core.df):,} x {len(core.df.columns)}",
                        "columns": list(core.df.columns),
                    }

            return await run_in_threadpool(install)
    finally:
        await file.close()


@app.get("/api/preview")
@workspace
def preview(rows: int = Query(10, ge=1, le=200)):
    core._require_data()
    frame = core.df.head(rows)
    return {
        "success": True,
        "columns": list(frame.columns),
        "data": frame.values.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
    }


@app.get("/api/columns")
@workspace
def columns():
    core._require_data()
    info = core.get_info()
    return {
        "success": True,
        "columns": info["columns"],
        "numeric": info["numeric_cols"],
        "categorical": info["categorical_cols"],
    }


@app.post("/api/reset")
@workspace
def reset():
    return core.reset_to_original()


@app.post("/api/clean")
@workspace
def clean(operation: str, strategy: str | None = None, target: str | None = None):
    core._require_data()
    if target:
        if target not in core.df:
            raise ValueError("Target column does not exist.")
        core.target_col = target
    operations = {
        "auto_clean": lambda: core.auto_clean(target_col=target),
        "remove_missing": lambda: core.handle_missing(strategy or "median"),
        "remove_duplicates": core.remove_duplicates,
        "remove_outliers": lambda: core.remove_outliers_isolation_forest(0.1),
        "encode_categorical": core.encode_categorical,
        "remove_constant": core.remove_constant_columns,
        "fix_dtypes": core.fix_dtypes,
    }
    if operation not in operations:
        raise ValueError("Unknown cleaning operation.")
    return operations[operation]()


@app.post("/api/split")
@workspace
def split(request: TrainRequest):
    return prepare(request)


@app.post("/api/train")
@workspace
def train(request: TrainRequest):
    prepare(request)
    names = {
        "Random Forest": ["n_estimators", "max_depth"],
        "Gradient Boosting": ["n_estimators", "max_depth", "learning_rate"],
        "Decision Tree": ["max_depth"],
        "Logistic Regression": ["C"],
        "SVM": ["C"],
        "SVR": ["C"],
        "KNN": ["n_neighbors"],
        "XGBoost": ["n_estimators", "max_depth", "learning_rate"],
    }.get(request.model, [])
    params = {
        name: getattr(request, name)
        for name in names
        if getattr(request, name) is not None
    }
    core.train(
        request.model,
        use_smote=request.use_smote,
        class_weight=request.class_weight,
        **params,
    )
    return {
        "success": True,
        "model": core.model_name,
        "problem_type": core.problem_type,
        "metrics": core.get_metrics(),
    }


@app.post("/api/train/compare")
@workspace
def train_compare(request: TrainRequest):
    prepare(request)
    frame = core.train_all_models(
        use_smote=request.use_smote, class_weight=request.class_weight
    )
    metric = "Accuracy" if core.problem_type == "classification" else "R\u00b2"
    leaderboard = [
        {"model": row["Model"], "score": float(row[metric])}
        for _, row in frame.iterrows()
    ]
    return {
        "success": True,
        "leaderboard": leaderboard,
        "best_model": core.model_name,
        "evaluation": "training_cross_validation",
    }


@app.get("/api/models")
@workspace
def models():
    return {
        "success": True,
        "models": core.get_trained_models(),
        "current_model": core.model_name,
    }


@app.post("/api/model/select")
@workspace
def select_model(request: dict = Body(...)):
    return core.select_model(request.get("model_name", ""))


@app.post("/api/model/stack")
@workspace
def stack(request: dict = Body(...)):
    return core.train_stacked_model(
        request.get("base_model_1"),
        request.get("base_model_2"),
        request.get("meta_model", "Logistic Regression"),
    )


@app.post("/api/predict")
@workspace
def predict(inputs: dict = Body(...)):
    return {"success": True, "result": core.predict(inputs)}


@app.get("/api/input-schema")
@workspace
def input_schema():
    return {"success": True, "schema": core.get_input_form_schema()}


@app.get("/api/evaluate")
@workspace
def evaluate():
    return {"success": True, "metrics": core.get_metrics()}


@app.get("/api/confusion-matrix")
@workspace
def confusion():
    return {"success": True, **core.get_confusion_matrix()}


@app.get("/api/feature-importance")
@workspace
def feature_importance():
    core._require_model()
    return {"success": True, "importance": core.feature_importance}


@app.get("/api/feature-importance-rf")
@workspace
def rf_importance():
    return {"success": True, "importance": core.get_fast_feature_importance().to_dict()}


@app.get("/api/mutual-info")
@workspace
def mutual_info():
    return {
        "success": True,
        "importance": core.get_mutual_information_top(20).to_dict(),
    }


@app.post("/api/feature/select")
@workspace
def feature_select(request: dict = Body(...)):
    features = request.get("features")
    if not isinstance(features, list):
        raise ValueError("Provide a list of feature names.")
    core.select_features_manual(features)
    return {"success": True, "count": len(features)}


@app.post("/api/feature/pca")
@workspace
def pca(components: int = Query(2, ge=1)):
    result = core.apply_pca(components)
    return {"success": True, "total_variance": sum(result["explained_variance_ratio"])}


@app.post("/api/feature/restore")
@workspace
def restore():
    return {"success": core.restore_from_pca()}


@app.post("/api/feature/keep_top")
@workspace
def keep_top(k: int = Query(10, ge=1, le=2000)):
    features = list(core.get_fast_feature_importance(k).index)
    core.select_features_manual(features)
    return {"success": True, "features": features}


@app.get("/api/feature/status")
@workspace
def feature_status():
    return {
        "success": True,
        "total": len(core._all_input_names),
        "selected": len(core.feature_names),
    }


@app.get("/api/cross-validation")
@workspace
def cross_validation():
    result = core.get_cross_val_score()
    return {"success": True, "cv": result["folds"], **result}


@app.get("/api/permutation-importance")
@workspace
def permutation():
    return {
        "success": True,
        "importance": core.get_permutation_importance().to_dict(orient="records"),
    }


@app.get("/api/misclassified")
@workspace
def misclassified():
    return {
        "success": True,
        "samples": core.get_misclassified_samples(30).to_dict(orient="records"),
    }


@app.get("/api/feature-importance-plot")
@workspace
def importance_plot():
    return {"success": True, "image": core.plot_feature_importance()}


@app.post("/api/chart/{chart_type}")
@workspace
def chart(chart_type: str, column: str | None = Query(None)):
    core._require_data()
    if column and column not in core.df:
        raise ValueError("Column not found.")
    if chart_type == "histogram":
        if not column:
            numeric = list(core.df.select_dtypes(include=np.number).columns)
            if not numeric:
                raise ValueError("Select a numeric column.")
            column = numeric[0]
        if not pd.api.types.is_numeric_dtype(core.df[column]):
            raise ValueError("Histogram requires a numeric column.")
        image = core.plot_histogram(column)
    elif chart_type == "target":
        image = core.plot_target_distribution(column or core.target_col)
    else:
        functions = {
            "correlation": core.plot_correlation_heatmap,
            "missing": core.plot_missing_heatmap,
            "confusion": core.plot_confusion_matrix,
            "roc": core.plot_roc_curve,
            "pr": core.plot_precision_recall_curve,
            "residuals": core.plot_residuals,
        }
        if chart_type not in functions:
            raise ValueError("Unknown chart type.")
        image = functions[chart_type]()
    return {"success": True, "image": image}


@app.get("/api/insights")
@workspace
def insights():
    return {"success": True, "insights": core.generate_insights()}


@app.get("/api/health-score")
@workspace
def health():
    return {"success": True, **core.get_dataset_health_score()}


@app.get("/api/export/report")
@workspace
def export_report():
    with tempfile.TemporaryDirectory(prefix="datayab-report-") as directory:
        path = Path(directory) / "report.html"
        core.export_report(path)
        return Response(
            path.read_bytes(),
            media_type="text/html",
            headers={
                "Content-Disposition": 'attachment; filename="DataYab_Report.html"'
            },
        )


def run_web_app(host="127.0.0.1", port=8001):
    import uvicorn

    print(f"DataYab local workspace: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_web_app()
