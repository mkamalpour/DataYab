# 📦 DataYab — Complete Frozen Memory Document

## 🎯 What This Document Is

This is the **complete, frozen reference** for the DataYab project. Every file, every function, every endpoint, every feature is documented here. **You do not need to send code files anymore.** Just reference this document and I will understand the entire project.

---

## 📊 Project Overview

**DataYab** is a **local, private dataset analyzer** that runs entirely on your machine. No data ever leaves your computer.

### Three Interfaces
| Interface | File | Purpose |
|-----------|------|---------|
| **Tkinter GUI** | `frontend/tinker.py` | Full-featured desktop app (primary) |
| **Web Interface** | `frontend/web.py` | Simple browser-based demo |
| **CLI Mode** | Built into `app.py` | Quick terminal testing |

### Core Capabilities
- Load CSV, Excel, JSON, Parquet
- Clean data (11 missing strategies, outliers, duplicates)
- Explore with charts (histograms, boxplots, correlation, scatter)
- Train classification & regression models (10+ models)
- Feature selection (RF importance, mutual info, manual, PCA)
- Stacked ensembles with meta-learner choice
- Evaluate (metrics, confusion matrix, ROC, PR, residuals, CV, permutation importance)
- Predict (manual form, batch CSV)
- Export HTML reports, save/load pipelines
- Data profiling (standalone utility)

---

## 📁 Complete File Structure

```
DataYab/
│
├── app.py                          # Main launcher
│
├── backend/
│   └── core.py                     # ML engine (2000+ lines)
│
├── frontend/
│   ├── tinker.py                   # Tkinter GUI (2000+ lines)
│   ├── web.py                      # FastAPI server (600+ lines)
│   ├── templates/
│   │   └── index.html              # Web UI (Bootstrap 5)
│   └── static/
│       └── app.js                  # JavaScript client
│
├── data/
│   ├── data_profile.py             # Standalone profiler
│   ├── dataset_Telco-Customer-Churn.csv  # Sample dataset
│   └── result.json                 # Profiler output
│
├── uploads/                        # Web uploads folder
│
└── requirements.txt                # Dependencies
```

---

## 🔧 backend/core.py — DataYabCore Class

### Initialization Attributes
```python
self.df                    # Current DataFrame
self.original_df           # Original copy
self.target_col            # Target column name
self.problem_type          # 'classification' or 'regression'
self.X_train, self.X_test  # Training/test features
self.y_train, self.y_test  # Training/test targets
self.feature_names         # List of feature names
self.model                 # Trained model
self.model_name            # Name of current model
self.scaler                # StandardScaler or MinMaxScaler
self.label_encoders        # Dict of LabelEncoders for categoricals
self.target_encoder        # LabelEncoder for target
self.trained_models        # Dict of trained models with metadata
self.available_models      # List of trained model names
self.metrics               # Evaluation metrics
self.feature_importance    # Feature → importance score dict
self.insights              # List of insight strings
self.log                   # Operation history
self.y_pred, self.y_proba  # Predictions and probabilities
self.misclassified_indices # Indices of wrong predictions

# Feature selection tracking
self._used_feature_selection = False
self._selected_features_indices = None
self._selected_feature_names = None

# Model switching
self.eval_model = None
self.eval_model_name = None
```

### Data Loading Methods
| Method | Description |
|--------|-------------|
| `load_csv(file_path, sep=None)` | Auto-detects encoding and separator |
| `load_excel(file_path, sheet_name=0)` | Loads Excel file |
| `load_json(file_path)` | Handles nested JSON structures |
| `load_parquet(file_path)` | Loads Parquet file |
| `load_file(file_path)` | Auto-detects format from extension |
| `get_info()` | Returns dataset info dict |
| `reset_to_original()` | Restores original dataset |

### Data Cleaning Methods
| Method | Description |
|--------|-------------|
| `get_missing_report()` | Returns DataFrame of missing values |
| `handle_missing(strategy, columns)` | 11 strategies: drop_rows, mean, median, mode, ffill, bfill, zero, unknown, knn, iterative, drop_cols |
| `remove_duplicates()` | Removes duplicate rows |
| `encode_categorical(columns, method)` | Label or One-Hot encoding |
| `remove_constant_columns()` | Removes single-value columns |
| `remove_high_cardinality(threshold=50)` | Removes high cardinality categoricals |
| `detect_outliers(column)` | IQR outlier detection |
| `cap_outliers(column)` | Caps outliers at IQR bounds |
| `detect_outliers_isolation_forest(contamination=0.1)` | Isolation Forest outlier detection |
| `remove_outliers_isolation_forest(contamination=0.1)` | Removes Isolation Forest outliers |
| `fix_dtypes()` | Converts object columns to numeric/datetime |
| `auto_clean(target_col=None)` | One-click complete cleaning |

### EDA/Chart Methods (all return base64 PNG)
| Method | Description |
|--------|-------------|
| `plot_histogram(column, bins=30)` | Distribution histogram |
| `plot_boxplot(column)` | Box plot for outliers |
| `plot_correlation_heatmap(max_cols=20)` | Correlation matrix heatmap |
| `plot_target_distribution(target=None)` | Target variable distribution |
| `plot_scatter(x_col, y_col, hue_col=None)` | Scatter plot |
| `plot_missing_heatmap()` | Missing values visualization |
| `plot_feature_importance()` | Feature importance bar chart |
| `plot_pairplot(columns=None, hue=None)` | Pairplot of numeric columns |
| `get_column_types()` | Returns dict of column types |
| `get_stats()` | DataFrame with statistics |
| `get_correlation_matrix()` | Correlation DataFrame |
| `generate_observations(n=10)` | List of natural language observations |

### Training Methods
| Method | Description |
|--------|-------------|
| `auto_detect_problem_type(target)` | Returns 'classification' or 'regression' |
| `get_model_recommendation(target=None)` | Returns dict with recommended models/strategies |
| `split_data(target, test_size=0.2, scale=True, scaler_type='standard')` | Splits data, encodes, scales |
| `train(model_name, use_smote=False, class_weight=None, **kwargs)` | Trains single model, stores with metadata |
| `train_all_models(use_smote=False, class_weight=None, **kwargs)` | Trains all models, returns comparison DataFrame |
| `get_trained_models()` | Returns list of available trained models |
| `select_model(model_name)` | Switches to previously trained model |
| `set_evaluation_model(model_name)` | Sets model for Evaluation tab (separate from main) |
| `get_evaluation_model()` | Returns current evaluation model name |
| `train_stacked_model(base_model_1, base_model_2, meta_model='Logistic Regression')` | Trains stacking ensemble |
| `get_cross_val_score(cv=5)` | Cross-validation scores |

### Feature Selection Methods
| Method | Description |
|--------|-------------|
| `get_fast_feature_importance(top_n=20)` | Fast RF feature importance |
| `get_mutual_information_top(k=10)` | Top K features by mutual information |
| `select_features_manual(feature_list)` | Manually select features (handles numpy arrays) |
| `apply_pca(n_components=2)` | PCA dimensionality reduction |
| `restore_from_pca()` | Restores original features after PCA |

### Evaluation Methods
| Method | Description |
|--------|-------------|
| `get_metrics()` | Returns metrics dict (acc, prec, recall, f1, roc_auc or r2, mae, mse, rmse) |
| `get_confusion_matrix()` | Returns dict with matrix and labels |
| `plot_confusion_matrix()` | Base64 PNG |
| `plot_roc_curve()` | Base64 PNG |
| `plot_precision_recall_curve()` | Base64 PNG |
| `plot_residuals()` | Base64 PNG (regression only) |
| `get_permutation_importance(top_n=15)` | Model-agnostic importance DataFrame |
| `get_misclassified_samples(max_samples=20, features_to_show=None)` | DataFrame of misclassified samples |

### Prediction Methods
| Method | Description |
|--------|-------------|
| `get_input_form_schema()` | Returns list of features with type, options, min/max |
| `predict(input_dict)` | Returns prediction, confidence, probabilities |
| `predict_from_csv(file_path, output_path=None)` | Batch predictions, returns DataFrame with sample_id, prediction, confidence |

### Export & Persistence Methods
| Method | Description |
|--------|-------------|
| `save_pipeline(file_path)` | Saves model, scaler, encoders, metadata to .pkl |
| `load_pipeline(file_path)` | Loads pipeline with all metadata |
| `export_report(file_path)` | Generates HTML report with charts and insights |
| `export_cleaned_csv(file_path)` | Saves cleaned DataFrame |
| `get_dataset_health_score()` | Returns score (0-100) and grade (A-F) |
| `generate_insights()` | Returns structured insight list |

### Available Models

**Classification:**
- Random Forest
- Logistic Regression
- Decision Tree
- SVM (with probability=True)
- KNN
- Gradient Boosting
- Naive Bayes
- XGBoost (if installed)

**Regression:**
- Random Forest
- Linear Regression
- Ridge
- Decision Tree
- SVR
- KNN
- Gradient Boosting
- XGBoost (if installed)

### Model Valid Parameters (for hyperparameter filtering)
```python
model_valid_params = {
    "Random Forest": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
    "Gradient Boosting": ["n_estimators", "max_depth", "learning_rate", "subsample"],
    "Logistic Regression": ["C", "penalty", "solver", "max_iter"],
    "Decision Tree": ["max_depth", "min_samples_split", "min_samples_leaf", "criterion"],
    "SVM": ["C", "kernel", "gamma", "degree"],
    "KNN": ["n_neighbors", "weights", "algorithm", "p"],
    "Naive Bayes": ["var_smoothing", "priors"],
}
```

---

## 🖥️ frontend/tinker.py — Tkinter GUI

### Main Application Class
```python
class DataYabApp:
    def __init__(self):
        # 1500x980 window, dark mode support
        # Creates 7 tabs, status bar, log panel
```

### Tab Classes
| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `LoadTab` | File upload, preview | `load_csv/excel/json/parquet()`, `refresh_preview()` |
| `CleanTab` | Data cleaning | `handle_missing()`, `remove_duplicates()`, `encode_cat()`, `auto_clean()` |
| `EDATab` | Charts | `generate_chart()`, `generate_obs()` |
| `TrainTab` | Model training | `train_model()`, `compare_all()`, `show_rf_importance()`, `show_mutual_info()`, `apply_manual_selection()`, `apply_pca()`, `restore_from_pca()`, `show_stacking_dialog()` |
| `EvaluateTab` | Model evaluation | `show_metrics()`, `show_confusion()`, `show_roc()`, `show_pr()`, `show_perm_importance()`, `show_cv()`, `show_misclassified()`, `load_selected_model()` |
| `InsightsTab` | Insights | `generate_insights()`, `show_health()` |
| `PredictTab` | Predictions | `build_form()`, `predict_manual()`, `batch_predict()` |

### Key UI Features
- **Dark mode toggle** - Changes all colors dynamically
- **Threaded operations** - UI never freezes during training
- **Log panel** - Timestamped messages
- **Progress bar** - Shows when processing
- **Hyperparameter sliders** - n_estimators, max_depth, learning_rate, C, n_neighbors
- **Feature selection UI** - RF importance, mutual info, manual entry, PCA controls
- **Stacking dialog** - Choose base models and meta-learner

### Running
```bash
python frontend/tinker.py
# or
python app.py --tkinter
```

---

## 🌐 frontend/web.py — FastAPI Web Interface

### Server Configuration
- **Port:** 8001
- **CORS:** Allow all origins
- **Static files:** `/static` folder
- **Uploads:** `/uploads` folder

### Complete API Endpoints (32 total)

| # | Endpoint | Method | Description |
|---|----------|--------|-------------|
| 1 | `/` | GET | Serves HTML interface |
| 2 | `/api/status` | GET | Dataset and model status |
| 3 | `/api/upload` | POST | Upload dataset file |
| 4 | `/api/preview` | GET | Data preview (10 rows) |
| 5 | `/api/clean` | POST | Apply cleaning operations |
| 6 | `/api/train` | POST | Train a model (JSON body) |
| 7 | `/api/predict` | POST | Make prediction (JSON body) |
| 8 | `/api/insights` | GET | Generate insights |
| 9 | `/api/chart/{chart_type}` | POST | Generate chart |
| 10 | `/api/columns` | GET | List all columns |
| 11 | `/api/export/report` | GET | Download HTML report |
| 12 | `/api/models` | GET | List trained models |
| 13 | `/api/model/select` | POST | Switch evaluation model |
| 14 | `/api/model/stack` | POST | Train stacked ensemble |
| 15 | `/api/input-schema` | GET | Input form schema |
| 16 | `/api/evaluate` | GET | Evaluation metrics |
| 17 | `/api/confusion-matrix` | GET | Confusion matrix data |
| 18 | `/api/feature-importance` | GET | Feature importance |
| 19 | `/api/feature-importance-rf` | GET | RF feature importance |
| 20 | `/api/mutual-info` | GET | Mutual information scores |
| 21 | `/api/feature/select` | POST | Manual feature selection |
| 22 | `/api/feature/pca` | POST | Apply PCA |
| 23 | `/api/feature/restore` | POST | Restore from PCA |
| 24 | `/api/feature/keep_top` | POST | Keep top K features |
| 25 | `/api/feature/status` | GET | Feature selection status |
| 26 | `/api/train/compare` | POST | Compare all models |
| 27 | `/api/permutation-importance` | GET | Permutation importance |
| 28 | `/api/cross-validation` | GET | Cross-validation scores |
| 29 | `/api/misclassified` | GET | Misclassified samples |
| 30 | `/api/feature-importance-plot` | GET | Feature importance plot |
| 31 | `/api/health-score` | GET | Dataset health score |
| 32 | `/api/chart/pr` | POST | Precision-Recall curve |

### Helper Function
```python
def convert_to_native(obj):
    """Converts numpy types to Python native types for JSON serialization"""
    # Handles int64, float64, ndarray, dict, list, bool_
```

### Running
```bash
python frontend/web.py
# or
python app.py --web
# Open http://127.0.0.1:8001
```

---

## 📄 frontend/templates/index.html — Web UI

### Sections (8 total)
1. **Upload Data** - Drag-drop area, file input
2. **Explore** - Data preview table, chart generator
3. **Clean Data** - 6 cleaning buttons
4. **Feature Selection** - RF importance, mutual info, manual, PCA
5. **Train Model** - Target selector, model selector, hyperparameters
6. **Evaluate** - Model selector, metrics, 8 evaluation buttons
7. **Predict** - Dynamic form, predict button
8. **Insights** - Generate insights, health score, export

### Modals
- **Stacking Dialog** - Base model 1, base model 2, meta-learner selector
- **Misclassified Samples** - Table with scrollbars

### Styling
- Bootstrap 5
- Custom CSS with CSS variables
- Responsive design
- Health score color coding (A-F grades)

---

## 📜 frontend/static/app.js — JavaScript Client

### Key Functions (40+)

**Upload & Data:**
- `uploadFile()` - Uploads dataset
- `loadPreview()` - Shows data preview
- `loadColumns()` - Populates column dropdowns

**Cleaning:**
- `cleanData(operation)` - Calls clean endpoint

**Feature Selection:**
- `getRFImportance()` - Calls `/api/feature-importance-rf`
- `getMutualInfo()` - Calls `/api/mutual-info`
- `applyManualSelection()` - Calls `/api/feature/select`
- `applyPCA()` - Calls `/api/feature/pca`
- `restoreFromPCA()` - Calls `/api/feature/restore`
- `keepTopFeatures()` - Calls `/api/feature/keep_top`
- `loadFeatureStatus()` - Calls `/api/feature/status`

**Training:**
- `trainModel()` - Sends JSON to `/api/train`
- `compareAllModels()` - Sends JSON to `/api/train/compare`
- `showStackingDialog()` - Shows modal, loads models
- `trainStackedModel()` - Sends JSON to `/api/model/stack`

**Evaluation:**
- `loadModels()` - Gets trained models
- `loadSelectedModel()` - Switches evaluation model
- `loadMetrics()` - Gets metrics
- `generateChart(type)` - Calls chart endpoint
- `getPermutationImportance()` - Calls `/api/permutation-importance`
- `getCrossValidation()` - Calls `/api/cross-validation`
- `showMisclassified()` - Shows misclassified samples modal
- `getFeatureImportance()` - Gets feature importance plot

**Predictions:**
- `loadPredictForm()` - Builds dynamic form from schema
- `makePrediction()` - Sends JSON to `/api/predict`

**Insights:**
- `generateInsights()` - Gets insights
- `showHealthScore()` - Shows health score
- `exportReport()` - Downloads HTML report

**Navigation:**
- Section switching with data-section attributes
- Auto-loads data when switching tabs

---

## 📊 data/data_profile.py — Standalone Profiler

### Purpose
Analyzes any dataset without loading into DataYab. Outputs JSON with complete statistics.

### Supported Formats
- CSV
- Excel (.xlsx, .xls)
- JSON
- Parquet

### Output Structure
```json
{
  "dataset_identity": {
    "name", "format", "full_rows", "full_columns", "full_cells", "file_size_mb"
  },
  "quality_metrics": {
    "total_missing_cells", "missing_percent", "total_duplicate_rows",
    "duplicate_percent", "complete_rows", "complete_rows_percent", "memory_usage_mb"
  },
  "columns": {
    "column_name": {
      "basic": {"dtype", "nulls", "null_pct", "unique_count", "unique_pct", "cardinality"},
      "uniqueness": {"is_perfect_id", "is_completely_constant", "has_duplicates"},
      "sample_values": [...],
      "top_values": {...},  // for low cardinality
      "numeric": {...},      // min, max, mean, median, std, skewness, outliers
      "datetime": {...},     // min, max, range_days
      "text": {...}          // min_length, max_length, mean_length
    }
  },
  "relationships": {
    "potential_primary_keys": [...],
    "highly_correlated_pairs": [{"col1", "col2", "correlation"}]
  },
  "summary": {...}
}
```

### Key Functions
- `ultimate_profile(file_path)` - Generates complete profile
- `convert_to_serializable(obj)` - Handles numpy/pandas types
- `print_quick_summary(profile)` - Prints human-readable summary

### Running
```bash
python data/data_profile.py
# Prompts for file path or auto-selects first dataset in folder
# Outputs result.json in same folder
```

---

## 🚀 app.py — Main Launcher

### Interactive Menu
```
========================================
   📊 DataYab — Ultimate Dataset Analyzer
========================================

[1] 🖥️  Tkinter GUI (Desktop App)
[2] 🌐 Web Interface (FastAPI + HTML)
[3] 📟 CLI Mode (Terminal)
[4] 📖 Help
[0] ❌ Exit
```

### Command-Line Arguments
| Argument | Action |
|----------|--------|
| `--tkinter` or `-t` | Launch Tkinter GUI directly |
| `--web` or `-w` | Launch web interface directly |
| `--cli` or `-c` | Launch CLI mode directly |
| `--help` or `-h` | Show help |

### CLI Mode Workflow
1. Prompts for dataset path
2. Auto-detects file type
3. Loads dataset
4. Prompts for target column (optional)
5. If target provided: auto-clean, split, train Random Forest, show metrics
6. Waits for Enter to exit

### Functions
- `launch_tkinter()` - Imports and runs DataYabApp
- `launch_web()` - Calls run_web_app() on port 8001
- `launch_cli()` - Interactive terminal workflow
- `show_help()` - Prints usage information

---

## 📦 requirements.txt

```
# Core
pandas
numpy
scikit-learn
matplotlib
seaborn
pillow
joblib
openpyxl

# Web
fastapi
uvicorn

# Optional (auto-detected)
xgboost
imbalanced-learn
```

---

## 🔄 Execution Flow

```
User runs: python app.py
    │
    ├── Menu choice 1 → launch_tkinter() → DataYabApp() → Tkinter GUI
    │       │
    │       └── 7 tabs → User interacts → Calls core.py methods
    │
    ├── Menu choice 2 → launch_web() → run_web_app() → FastAPI on :8001
    │       │
    │       └── Browser → index.html + app.js → REST API → core.py
    │
    └── Menu choice 3 → launch_cli() → Terminal workflow → core.py
```

**All frontends use the same `backend/core.py` (DataYabCore class)**

---

## 🎯 Important Notes for AI Understanding

1. **All operations run locally** - No external API calls, no data leaves the machine

2. **Model switching** - Uses `eval_model` separate from main `model`. Evaluation tab and Predict tab use the evaluation model when set.

3. **Feature selection** - `_selected_feature_names` tracks selected features. Prediction forms use this to show only selected features.

4. **Batch predictions** - Returns only `sample_id`, `prediction`, `confidence` (not the whole CSV)

5. **Stacked models** - Only one stacked model exists; training a new one replaces the old.

6. **Pipeline saves** - Include all metadata: model, scaler, encoders, feature names, selected features, training timestamp.

7. **Web interface port** - Runs on **8001** (not 8000)

8. **Misclassified samples** - Limited to 30 rows and 5 features for readability

9. **Hyperparameter filtering** - Web endpoint filters hyperparameters by model type to prevent errors (e.g., no `learning_rate` for Random Forest)

10. **Type conversion** - `convert_to_native()` converts numpy types for JSON serialization

11. **Tkinter is primary** - Web interface is a demo with basic workflow only

12. **Data profiler is standalone** - Does not require DataYab core to run

---

## 🏆 Summary

DataYab is a **complete, production-ready dataset analyzer** with:

- ✅ 3 interfaces (Tkinter, Web, CLI)
- ✅ Classification & regression support
- ✅ 10+ ML models
- ✅ Feature selection (RF, mutual info, manual, PCA)
- ✅ Stacked ensembles with meta-learner choice
- ✅ Comprehensive evaluation (10+ metrics/plots)
- ✅ HTML report export
- ✅ Save/load pipelines
- ✅ Standalone data profiler
- ✅ 100% local and private

**Total lines of code:** ~7,500+

**Primary interface:** Tkinter desktop app
**Demo interface:** Web browser on port 8001
**Quick testing:** CLI mode

---

*This document is the frozen memory of DataYab. Reference it instead of sending code files.* 🚀