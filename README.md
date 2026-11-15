# DataYab

**[Explore the interactive project presentation](docs/overview.html)** — open locally in your browser; works offline.

Open a table, find data-quality problems, and compare prediction models without writing a notebook. DataYab runs locally as a **Streamlit web app** or **Tkinter desktop app**, or from the command line.

## Try it first

Open a CSV or Excel table and inspect the column types and missing values before training. Choose the target column and task, then compare held-out scores. Keep an untouched copy of your input and export results to a separate file.

<details>
<summary>Using the existing managed Windows workspace</summary>

Assigned environment: **base**. From this repository folder in PowerShell:

```powershell
python app.py --streamlit
```

The launcher uses your active Python interpreter. Activate a compatible
environment with the listed dependencies before launching.

</details>

## Quick start

Use **Python 3.13** (tested with 3.13.5) from an existing compatible environment. Requirements pin the versions validated together. Standard Windows Python includes Tkinter; Linux may require `python3-tk`.

### Streamlit Web UI (recommended)
```powershell
conda activate base  # Or use your existing compatible environment.
# Install dependencies if they are not already available in your environment.
python -m pip install -r requirements.txt
python -m pip install -r requirements-streamlit.txt
python app.py --streamlit
```

### Tkinter Desktop App
```powershell
python app.py --tkinter
# or just
python app.py
```

### CLI Mode
```powershell
python app.py --cli --file "path/to/dataset.csv"
python app.py --cli --file "path/to/dataset.csv" --target outcome --task classification
python app.py --cli --file "path/to/dataset.arff" --target Result --task classification
python tools/profile_data.py "path/to/dataset.csv" --output "path/to/profile.json"

# Write a deduplicated copy plus a Markdown quality report (source is untouched).
python tools/clean_dataset.py "path/to/Training Dataset.arff" --target Result --csv
```

## Workflow

1. Load CSV, Excel, JSON records, Parquet, `.arr` array files (NumPy arrays, Arrow/Feather tables, or delimited text), or Weka `.arff` datasets. Tables support up to 1,000,000 rows and 2,000 columns; flatten nested arrays/objects first.
2. Inspect the dataset quality report (duplicates, missing values, constant columns, class balance, strongest correlations). Automatic cleaning reports duplicates and converts numeric text but keeps every row; removing duplicates or outliers is an explicit action. The cleaned working table can be exported as CSV or ARFF.
3. Choose a target and train/test split. The UI and CLI support an explicit classification/regression override; automatic detection uses a low-cardinality integer heuristic.
4. **Feature selection** (choose which original columns feed the model) and **feature extraction** (build components from them) are separate stages. Selection offers Random Forest importance, mutual information, ANOVA F-test, correlation with the target, recursive feature elimination, a constant-column filter, and a manual column list. Extraction offers PCA, Truncated SVD, and LDA (classification only). Selection resets extraction; extraction leaves the selected columns intact. Data and feature changes clear outdated models.
5. Evaluate holdout metrics and charts, then predict from raw inputs. Desktop mode also supports batch CSV prediction and model import/export.

## Advanced Analytics (Streamlit)

The Streamlit UI includes dedicated tabs for:
- **Time Series**: Decomposition, ARIMA forecasting, stationarity testing (ADF)
- **Clustering**: K-Means, DBSCAN, Hierarchical, Gaussian Mixture Models
- **Anomaly Detection**: Isolation Forest, Local Outlier Factor, One-Class SVM, Z-Score, IQR

## Modeling details

Missing target rows are excluded. Imputation, scaling, categorical encoding, PCA, and sampling are fitted inside each training pipeline and cross-validation fold. Unknown categories are supported through ordinal encoding; assess whether its imposed ordering suits your data.

Comparison ranks training cross-validation scores, separate from reserved holdout evaluation. Repeated exploration can still overfit choices: validate final decisions on fresh data. CV following manual feature exploration is conditional on those choices. SMOTE supports numeric inputs; use supported class weights for mixed categorical datasets. XGBoost is optional and outside the tested base installation.

Exported packages contain preprocessing and raw-input schemas, enabling prediction without the original dataset. Imports require matching dependency versions; retrain old incomplete packages. Only open trusted model files: joblib/pickle can execute code.

## Local workspace and data

The application is intended for one local user. Large datasets and model comparisons may take time and memory.

Existing datasets and reports remain local, excluded from Git alongside model packages, environments, and caches. Files previously published remain in older Git history.

## Structure

- `app.py`: launcher and CLI
- `backend/`: modular exploration, cleaning, and model pipelines
  - `backend/core.py`: unified interface using modular components
  - `backend/modules/`: 11 focused modules (data_loader, data_cleaner, feature_engineering, modeling, prediction, visualization, insights, pipeline, validation, advanced_analytics)
- `frontend/desktop.py`: Tkinter desktop application
- `frontend/streamlit/`: Streamlit web application
- `tools/profile_data.py`: detailed JSON dataset profiler
- `tools/clean_dataset.py`: deduplicated ARFF/CSV copy plus Markdown quality report
- `tests/`: synthetic-data regression tests

## Validation

```powershell
# Install dependencies if they are not already available in your environment.
python -m pip install pytest==9.1.1 httpx==0.28.1
python -m pytest tests -q
python -m pip check
```

Regression tests cover training-only preprocessing, model export/import, regression, stacking, state invalidation, uploads, request guards, report escaping, ARFF/ARR loading, selection and extraction methods, dataset quality reporting, and ARFF/CSV export.

Run the regression suite in your active environment to verify your installation.

Prediction probabilities are uncalibrated model outputs, not guarantees of correctness. The evaluation model selector has an explicit accessible label.

No license file is included.

Core cleaned-data, batch-prediction, model and HTML-report exports are staged before replacing the destination. Failed writes preserve the previous export; locked destinations return an error rather than a false success.

Failed model selection and cleaning restore the previous working state. Nested array cells are rejected before replacing a dataset, and summaries tolerate entirely missing columns.

See the [engineering audit](docs/ENGINEERING_AUDIT.md) for the dated test baseline, project assessment, environment assignment and remaining work.

## Additional workflow improvements

Classification metrics now include balanced accuracy, macro F1 and the training-majority baseline; regression includes a training-mean baseline and RMSE skill.

Train a model and inspect Metrics in the web/desktop interface or CLI. Positive regression RMSE skill beats the baseline.
