# DataYab

A local dataset workbench for exploring tables, reviewing data quality, comparing machine-learning models, and making predictions. Choose the browser dashboard, desktop application, or command line.

## Quick start

Use **Python 3.13** (tested with 3.13.5) and a clean virtual environment. Requirements pin the versions validated together. Standard Windows Python includes Tkinter; Linux may require `python3-tk`.

```powershell
conda activate base  # Or activate another existing compatible environment.
# Optional dependency setup; requires explicit permission and connectivity.
python -m pip install -r requirements.txt
python app.py --web
```

Open http://127.0.0.1:8001. Run `python app.py --tkinter` for desktop mode (also the default). Web assets are local: no CDN or API key required.

```powershell
python app.py --cli --file "path/to/dataset.csv"
python app.py --cli --file "path/to/dataset.csv" --target outcome --task classification
python tools/profile_data.py "path/to/dataset.csv" --output "path/to/profile.json"
```

## Workflow

1. Load CSV, Excel, JSON records, or Parquet. Web uploads are temporary and limited to 50 MB. Tables support up to 1,000,000 rows and 2,000 columns; flatten nested arrays/objects first.
2. Inspect missing values, duplicates, distributions, correlations, and observations. Automatic cleaning removes duplicates and converts numeric text; other cleaning operations are explicit.
3. Choose a target and train/test split. The web UI and CLI support an explicit classification/regression override; automatic detection uses a low-cardinality integer heuristic.
4. Train or compare models, explore feature rankings, select original columns, or apply PCA. Data and feature changes clear outdated models.
5. Evaluate holdout metrics and charts, then predict from raw inputs. Desktop mode also supports batch CSV prediction and model import/export.

## Modeling details

Missing target rows are excluded. Imputation, scaling, categorical encoding, PCA, and sampling are fitted inside each training pipeline and cross-validation fold. Unknown categories are supported through ordinal encoding; assess whether its imposed ordering suits your data.

Comparison ranks training cross-validation scores, separate from reserved holdout evaluation. Repeated exploration can still overfit choices: validate final decisions on fresh data. CV following manual feature exploration is conditional on those choices. SMOTE supports numeric inputs; use supported class weights for mixed categorical datasets. XGBoost is optional and outside the tested base installation.

Exported packages contain preprocessing and raw-input schemas, enabling prediction without the original dataset. Imports require matching dependency versions; retrain old incomplete packages. Only open trusted model files: joblib/pickle can execute code.

## Local workspace and data

The loopback web server has **one shared workspace** across tabs and serializes operations. It is intended for one local user, not public multi-user hosting. Large datasets and model comparisons may take time and memory.

Existing datasets and reports remain local, excluded from Git alongside model packages, environments, and caches. Files previously published remain in older Git history.

## Structure

- `app.py`: launcher and CLI
- `backend/`: exploration, cleaning, and model pipelines
- `frontend/`: desktop UI, web API, templates, and local assets
- `tools/profile_data.py`: detailed JSON dataset profiler
- `tests/`: synthetic-data regression tests

## Validation

```powershell
# Optional dependency setup; requires explicit permission and connectivity.
python -m pip install pytest==9.1.1 httpx==0.28.1
python -m pytest tests -q
python -m pip check
```

Regression tests cover training-only preprocessing, model/PCA persistence, regression, stacking, state invalidation, uploads, request guards, and report escaping. Browser checks cover upload-to-prediction workflows and responsive layouts.

Current lightweight check (Windows, existing Conda base / Python 3.13.5): **13 passed**. No dependency installs, model downloads, live publishing, or heavy datasets were used.
