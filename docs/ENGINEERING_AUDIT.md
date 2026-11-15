# Engineering audit — 2026-11-16

Purpose: Local table profiling, preprocessing and model comparison.

Verification update (2026-11-16): 67 backend tests passed in the installed
Windows environment. Added integration coverage for dataset replacement,
feature/model invalidation, failed feature operations, grid search, conditional
CV disclosure, holdout row mapping, categorical-only statistics, nine advanced
analytics algorithms and ARIMA forecasting. Global warning suppression was
removed; dependency warnings remain visible. This count supersedes older
baselines below and does not guarantee arbitrary datasets or GUI interactions.

Classification: **A** (single application/tool).
Final structure: backend/, frontend/, tools/, tests/. Existing capabilities and source material were retained.

| Assessment | Result |
| --- | --- |
| Idea strength | 8/10 |
| Implementation strength | 7/10 |
| Core strength | Synthetic workflow tests and explicit train/validation boundaries. |
| Main limitation | Large or unusual real datasets need separate memory and schema validation. |
| Readiness | Local prototype/tool with the verified baseline below; no production certification |

Scores are subjective engineering estimates, not measured product or model quality.

## Delivered

- **Modular backend**: Refactored into 11 focused modules under `backend/modules/` (data_loader, data_cleaner, feature_engineering, modeling, prediction, visualization, insights, pipeline, validation, advanced_analytics) with `backend/core.py` as unified interface.
- **Streamlit UI**: Modern, memory-efficient web interface with 12 tabs covering full ML workflow plus advanced analytics (Time Series, Clustering, Anomaly Detection).
- **Tkinter Desktop**: Retained for offline desktop use.
- **Removed legacy**: FastAPI web server (`frontend/web.py`), legacy HTML/JS templates, and hardcoded-path launcher (`run.cmd`).

Assigned environment: **base**. Run `python app.py --streamlit` or `python app.py --tkinter`.
The launcher installs nothing and preserves its child exit status. On this workstation
it selects the assigned interpreter; standalone clones fall back to active Python.

## Verification

- **Automated tests:** 55 passed; 0 reported skips.
- **Startup check:** `python -c "from backend.core import DataYabCore; DataYabCore()"` passes in the assigned environment.
- **Tested runtime:** Windows; Conda base; Python 3.13.5.
- **Data handling:** tests used temporary/synthetic fixtures. No live transactions, provider generation, deployment, new model downloads or package installs were performed.
- **Repository check:** origin/main fetched before publication; no pre-existing local changes were present. A tracked-file scan found no matches for the high-confidence credential patterns checked. That scan is not a guarantee that all personal data or arbitrary secret formats are absent. Ignored local data was not staged.
- **Limits:** optional external workflows, representative real-data quality and all possible platform/version combinations remain unverified.

Test command from the repository root:

```powershell
conda activate base
python -m pip install pytest==9.1.1 httpx==0.28.1
python -m pytest tests -q
```

## Next three upgrades

1. Reusable column-mapping presets.
2. Exported experiment provenance.
3. Representative large-file benchmarks.

Expected benefit: clearer decisions, more reproducible workflows and stronger evidence for quality. These are proposed upgrades, not implemented features or measured speedups.

External assets: Synthetic tables suffice for the baseline; real schemas are needed for import validation.

No root license file is set. No license was selected or changed.

## Additional implementation pass

Classification metrics now include balanced accuracy, macro F1 and the training-majority baseline; regression includes a training-mean baseline and RMSE skill.

Use: Train a model and inspect Metrics in the Streamlit/desktop interface or CLI. Positive regression RMSE skill beats the baseline.

Verification: the complete offline suite passed in the assigned **base** environment after this change (55 tests). Regression fixtures remain temporary/synthetic. Optional real-model/data quality and external integrations remain unverified. No packages, models or additional environments were installed.

The earlier numeric scores are subjective historical assessments; new features do not automatically establish a higher score or production readiness.
