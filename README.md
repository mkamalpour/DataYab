# 📊 DataYab

**DataYab** is a local dataset analyzer that loads, cleans, explores, trains models, and makes predictions — all on your computer. No data ever leaves your machine.

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
python app.py

# Then choose:
# [1] Desktop App (full features)
# [2] Web Browser (simple demo)
# [3] Terminal (quick testing)
```

---

## ✨ What It Does

| Step | What You Do |
|------|-------------|
| 1. **Load** | Upload CSV, Excel, JSON, or Parquet |
| 2. **Clean** | Click "Auto Clean" — handles missing values, duplicates, outliers |
| 3. **Explore** | Generate charts and auto-observations |
| 4. **Train** | Select target column → click "Train Model" |
| 5. **Evaluate** | View metrics, confusion matrix, ROC curve |
| 6. **Predict** | Enter values manually or upload a CSV batch |
| 7. **Export** | Save HTML reports or trained models |

### Advanced Features
- **Feature Selection** — RF importance, mutual info, manual selection, PCA
- **Model Comparison** — Train all models at once, see leaderboard
- **Stacked Ensembles** — Combine two models with meta-learner choice
- **Cross-Validation** — 5-fold stratified validation
- **Permutation Importance** — Model-agnostic feature importance
- **Health Score** — 0-100 score with A-F grade
- **Batch Predictions** — Upload CSV, get predictions with confidence scores

---

## 📁 Files

| File | What It Does |
|------|--------------|
| `app.py` | Launcher (menu to choose interface) |
| `backend/core.py` | ML engine |
| `frontend/tinker.py` | Desktop GUI (7 tabs, dark mode) |
| `frontend/web.py` | Web interface (port 8001) |
| `data/data_profile.py` | Standalone data profiler |

---

## 🖥️ Three Interfaces

| Interface | Command | Best For |
|-----------|---------|----------|
| Desktop | `python app.py --tkinter` | Full features |
| Web | `python app.py --web` | Quick demo |
| CLI | `python app.py --cli` | Terminal testing |

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pillow joblib openpyxl fastapi uvicorn
```

Optional: `xgboost`, `imbalanced-learn`

---

## 🎯 Example (5 minutes)

```bash
python app.py --tkinter
```

1. **Load** → `data/dataset_Telco-Customer-Churn.csv`
2. **Clean** → Click "Auto Clean"
3. **Train** → Select target `Churn` → Click "Split & Train"
4. **Insights** → Click "Generate Insights"
5. **Predict** → Enter values → See prediction

---

## 📄 Outputs

| Output | How |
|--------|-----|
| HTML Report | Insights tab → "Export HTML Report" |
| Cleaned CSV | File menu → "Export Cleaned CSV" |
| Trained Model | File menu → "Save Pipeline" |

---

## ✅ Requirements

- Python 3.8+
- 4GB RAM
- 500MB storage

---

**Your data stays local. Always.** 🔒