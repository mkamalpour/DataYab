"""
web.py — FastAPI web interface for DataYab
Modern, fast, async backend for HTML front-end
"""

import os
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.core import DataYabCore

# ================================
# FastAPI Setup
# ================================
app = FastAPI(title="DataYab", version="1.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folders
UPLOAD_FOLDER = Path(__file__).parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Global core instance
core = DataYabCore()

# Helper function
def convert_to_native(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# ================================
# Routes
# ================================
@app.get("/")
async def home():
    """Serve HTML front-end."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>HTML template not found</h1>", status_code=404)
    return FileResponse(str(html_path), media_type="text/html")

@app.get("/api/status")
async def status():
    """Get current status."""
    if core.df is None:
        return {"loaded": False, "message": "No dataset loaded"}
    
    info = core.get_info()
    return {
        "loaded": True,
        "rows": info['rows'],
        "cols": info['cols'],
        "columns": info['columns'][:20],
        "target": core.target_col,
        "model": core.model_name,
        "problem_type": core.problem_type
    }

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload and load dataset."""
    try:
        filepath = UPLOAD_FOLDER / file.filename
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        result = core.load_file(str(filepath))
        if result["success"]:
            info = core.get_info()
            return {
                "success": True,
                "message": result["message"],
                "shape": f"{info['rows']:,} × {info['cols']}",
                "columns": info['columns']
            }
        else:
            return {"success": False, "message": result["message"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/preview")
async def preview(rows: int = 10):
    """Get data preview."""
    if core.df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    df = core.df.head(rows)
    return {
        "success": True,
        "columns": list(df.columns),
        "data": df.values.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

@app.post("/api/clean")
async def clean(operation: str, strategy: Optional[str] = None, target: Optional[str] = None):
    """Apply cleaning operations."""
    if core.df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        if operation == "auto_clean":
            result = core.auto_clean(target_col=target)
            return {"success": True, "steps": result["steps"]}
        elif operation == "remove_missing":
            result = core.handle_missing(strategy or "median")
            return {"success": True, "message": result["message"]}
        elif operation == "remove_duplicates":
            result = core.remove_duplicates()
            return {"success": True, "removed": int(result.get("removed", 0))}
        elif operation == "remove_outliers":
            result = core.remove_outliers_isolation_forest(0.1)
            return {"success": True, "removed": int(result.get("removed", 0))}
        elif operation == "encode_categorical":
            result = core.encode_categorical()
            return {"success": True, "encoded": result.get("encoded_columns", [])}
        elif operation == "remove_constant":
            result = core.remove_constant_columns()
            return {"success": True, "removed": result.get("removed", [])}
        elif operation == "fix_dtypes":
            result = core.fix_dtypes()
            return {"success": True, "converted": result.get("converted", [])}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train(request: Dict[str, Any] = Body(...)):
    """Train a model - accepts JSON body."""
    if core.df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        target = request.get("target")
        model = request.get("model", "Random Forest")
        test_size = request.get("test_size", 0.2)
        use_smote = request.get("use_smote", False)
        
        if not target:
            raise HTTPException(status_code=400, detail="Target column required")
        
        # Split data
        core.split_data(target, test_size=test_size)
        
        # Build hyperparameters based on model type (filter properly)
        hyper = {}
        
        # n_estimators - only for RF and Gradient Boosting
        if "n_estimators" in request and model in ["Random Forest", "Gradient Boosting"]:
            hyper["n_estimators"] = request["n_estimators"]
        
        # max_depth - for tree-based models
        if "max_depth" in request and model in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            hyper["max_depth"] = request["max_depth"]
        
        # learning_rate - only for Gradient Boosting
        if "learning_rate" in request and model == "Gradient Boosting":
            hyper["learning_rate"] = request["learning_rate"]
        
        # C - for Logistic Regression and SVM
        if "C" in request and model in ["Logistic Regression", "SVM"]:
            hyper["C"] = request["C"]
        
        # n_neighbors - for KNN
        if "n_neighbors" in request and model == "KNN":
            hyper["n_neighbors"] = request["n_neighbors"]
        
        # Train the model
        core.train(model_name=model, use_smote=use_smote, **hyper)
        metrics = core.get_metrics()
        metrics = convert_to_native(metrics)
        
        return {
            "success": True,
            "model": model,
            "problem_type": core.problem_type,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(inputs: Dict[str, Any] = Body(...)):
    """Make a prediction."""
    if core.model is None:
        raise HTTPException(status_code=400, detail="No model trained")
    
    try:
        result = core.predict(inputs)
        result = convert_to_native(result)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights")
async def insights():
    """Generate insights."""
    if core.df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        insights_list = core.generate_insights()
        return {"success": True, "insights": insights_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chart/{chart_type}")
async def chart(chart_type: str, column: Optional[str] = Query(None)):
    """Generate a chart."""
    if core.df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        if chart_type == "histogram":
            if not column:
                numeric_cols = core.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    column = numeric_cols[0]
                else:
                    column = core.df.columns[0]
            if not pd.api.types.is_numeric_dtype(core.df[column]):
                raise HTTPException(400, f"Cannot plot histogram on non-numeric column: '{column}'. Please select a numeric column.")
            img_b64 = core.plot_histogram(column)
        elif chart_type == "correlation":
            img_b64 = core.plot_correlation_heatmap()
        elif chart_type == "missing":
            img_b64 = core.plot_missing_heatmap()
        elif chart_type == "target":
            if not column:
                column = core.target_col
            if not column:
                raise HTTPException(400, "No target column set. Select a target column first.")
            img_b64 = core.plot_target_distribution(column)
        elif chart_type == "confusion" and core.problem_type == "classification":
            img_b64 = core.plot_confusion_matrix()
        elif chart_type == "roc" and core.problem_type == "classification":
            img_b64 = core.plot_roc_curve()
        elif chart_type == "pr" and core.problem_type == "classification":
            img_b64 = core.plot_precision_recall_curve()
        elif chart_type == "residuals" and core.problem_type == "regression":
            img_b64 = core.plot_residuals()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown chart: {chart_type}")
        
        return {"success": True, "image": img_b64}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/columns")
async def columns():
    """Get list of columns."""
    if core.df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    info = core.get_info()
    return {
        "success": True,
        "columns": list(core.df.columns),
        "numeric": info['numeric_cols'],
        "categorical": info['categorical_cols']
    }

@app.get("/api/export/report")
async def export_report():
    """Export HTML report."""
    if core.df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        report_path = UPLOAD_FOLDER / "report.html"
        result = core.export_report(str(report_path))
        if result["success"]:
            return FileResponse(str(report_path), media_type="text/html", filename="DataYab_Report.html")
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models():
    """Get list of trained models."""
    models = core.get_trained_models()
    return {
        "success": True,
        "models": models,
        "current_model": core.model_name
    }

@app.post("/api/model/select")
async def select_model(request: Dict[str, Any] = Body(...)):
    """Switch to a trained model for evaluation."""
    model_name = request.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name required")
    result = core.set_evaluation_model(model_name)
    if result.get("success"):
        return {"success": True, "model": model_name}
    else:
        raise HTTPException(status_code=400, detail=result.get("message"))

@app.post("/api/model/stack")
async def train_stacked(request: Dict[str, Any] = Body(...)):
    """Train a stacked ensemble model."""
    try:
        base_model_1 = request.get("base_model_1")
        base_model_2 = request.get("base_model_2")
        meta_model = request.get("meta_model", "Logistic Regression")
        
        if not base_model_1 or not base_model_2:
            raise HTTPException(400, "Both base models required")
        
        result = core.train_stacked_model(base_model_1, base_model_2, meta_model)
        if result.get("success"):
            return {
                "success": True,
                "model": result.get("model"),
                "base_models": result.get("base_models"),
                "meta_model": result.get("meta_model"),
                "elapsed": result.get("elapsed")
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("message"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/input-schema")
async def get_input_schema():
    """Get input form schema (only selected features)."""
    if core.model is None:
        raise HTTPException(status_code=400, detail="No model trained. Train a model first.")
    
    try:
        schema = core.get_input_form_schema()
        return {"success": True, "schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluate")
async def evaluate():
    """Get evaluation metrics for current model."""
    if core.model is None:
        raise HTTPException(status_code=400, detail="No model trained")
    try:
        metrics = core.get_metrics()
        metrics = convert_to_native(metrics)
        return {"success": True, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/confusion-matrix")
async def confusion_matrix():
    """Get confusion matrix data."""
    if core.model is None:
        raise HTTPException(status_code=400, detail="No model trained")
    if core.problem_type != "classification":
        raise HTTPException(status_code=400, detail="Confusion matrix only for classification")
    try:
        cm_data = core.get_confusion_matrix()
        return {"success": True, "matrix": cm_data["matrix"], "labels": cm_data["labels"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feature-importance")
async def feature_importance():
    """Get feature importance."""
    if core.model is None:
        raise HTTPException(status_code=400, detail="No model trained")
    try:
        return {"success": True, "importance": convert_to_native(core.feature_importance)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== FEATURE SELECTION ENDPOINTS ====================

@app.get("/api/feature-importance-rf")
async def feature_importance_rf():
    """Get Random Forest feature importance."""
    if core.X_train is None:
        raise HTTPException(400, "Split data first (train a model)")
    try:
        importance = core.get_fast_feature_importance(top_n=20)
        return {"success": True, "importance": convert_to_native(importance.to_dict())}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/mutual-info")
async def mutual_info():
    """Get mutual information scores."""
    if core.X_train is None:
        raise HTTPException(400, "Split data first (train a model)")
    try:
        mi = core.get_mutual_information_top(k=20)
        return {"success": True, "importance": convert_to_native(mi.to_dict())}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/feature/select")
async def feature_select(request: Dict[str, Any] = Body(...)):
    """Apply manual feature selection."""
    features = request.get("features", [])
    if not features:
        raise HTTPException(400, "No features provided")
    if core.X_train is None:
        raise HTTPException(400, "No training data. Split data first.")
    try:
        core.select_features_manual(features)
        return {"success": True, "count": len(features)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/feature/pca")
async def feature_pca(components: int = 2):
    """Apply PCA."""
    if core.X_train is None:
        raise HTTPException(400, "No training data. Split data first.")
    if len(core.feature_names) < 2:
        raise HTTPException(400, "Need at least 2 features for PCA")
    try:
        result = core.apply_pca(components)
        return {"success": True, "total_variance": sum(result['explained_variance_ratio'])}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/feature/restore")
async def feature_restore():
    """Restore from PCA."""
    try:
        result = core.restore_from_pca()
        return {"success": result}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/feature/keep_top")
async def feature_keep_top(k: int = 10):
    """Keep top K features."""
    if core.X_train is None:
        raise HTTPException(400, "No training data. Split data first.")
    try:
        importance = core.get_fast_feature_importance(top_n=k)
        features = list(importance.index)
        core.select_features_manual(features)
        return {"success": True, "features": features[:10]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/feature/status")
async def feature_status():
    """Get feature selection status."""
    total = len(core.feature_names) if core.feature_names else 0
    selected = len(core._selected_feature_names) if core._selected_feature_names else total
    return {"success": True, "total": total, "selected": selected}

@app.post("/api/train/compare")
async def train_compare(request: Dict[str, Any] = Body(...)):
    """Compare all models - without passing hyperparameters."""
    target = request.get("target")
    if not target:
        raise HTTPException(400, "Target required")
    if core.df is None:
        raise HTTPException(400, "No data loaded")
    try:
        core.split_data(target, test_size=0.2)
        # IMPORTANT: Don't pass any hyperparameters here
        df = core.train_all_models()
        
        score_col = "Accuracy" if core.problem_type == "classification" else "R²"
        leaderboard = []
        for _, row in df.iterrows():
            if score_col in row:
                leaderboard.append({
                    "model": row["Model"],
                    "score": float(row[score_col])
                })
        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        return {
            "success": True,
            "leaderboard": leaderboard[:10],
            "best_model": leaderboard[0]["model"] if leaderboard else None
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/permutation-importance")
async def permutation_importance():
    """Get permutation importance."""
    if core.model is None:
        raise HTTPException(400, "No model trained")
    try:
        df = core.get_permutation_importance(top_n=15)
        importance = df.to_dict(orient="records")
        return {"success": True, "importance": convert_to_native(importance)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/cross-validation")
async def cross_validation():
    """Get cross-validation scores."""
    if core.model is None:
        raise HTTPException(400, "No model trained")
    try:
        cv = core.get_cross_val_score(cv=5)
        return {"success": True, "cv": 5, "mean": cv["mean"], "std": cv["std"], "scores": cv["scores"]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/misclassified")
async def misclassified():
    """Get misclassified samples."""
    if core.model is None:
        raise HTTPException(400, "No model trained")
    if core.problem_type != "classification":
        raise HTTPException(400, "Only for classification")
    try:
        df = core.get_misclassified_samples(max_samples=30)
        samples = df.to_dict(orient="records")
        return {"success": True, "samples": convert_to_native(samples)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/feature-importance-plot")
async def feature_importance_plot():
    """Get feature importance plot."""
    if core.model is None:
        raise HTTPException(400, "No model trained")
    try:
        img_b64 = core.plot_feature_importance()
        return {"success": True, "image": img_b64}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/health-score")
async def health_score():
    """Get dataset health score."""
    if core.df is None:
        raise HTTPException(400, "No data loaded")
    try:
        health = core.get_dataset_health_score()
        return {"success": True, "score": health["score"], "grade": health["grade"], "issues": health["issues"]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/chart/pr")
async def chart_pr():
    """Generate Precision-Recall curve."""
    if core.model is None:
        raise HTTPException(400, "No model trained")
    if core.problem_type != "classification":
        raise HTTPException(400, "Only for classification")
    try:
        img_b64 = core.plot_precision_recall_curve()
        return {"success": True, "image": img_b64}
    except Exception as e:
        raise HTTPException(500, str(e))


def run_web_app(host: str = "127.0.0.1", port: int = 8001):
    """Run FastAPI app."""
    print(f"\n🚀 DataYab Web Interface (FastAPI)")
    print(f"{'='*50}")
    print(f"📍 Open your browser: http://{host}:{port}")
    print(f"{'='*50}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_web_app()