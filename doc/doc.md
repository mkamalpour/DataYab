# 📊 DataYab — Complete Project Documentation

## What is DataYab?

**DataYab** is a **local, private dataset analyzer** that runs entirely on your computer. You load a dataset (CSV, Excel, JSON, Parquet), and DataYab helps you clean it, explore it, train machine learning models, evaluate them, and make predictions — all without sending your data anywhere.

**Think of it as your personal data science assistant that works offline.**

---

## Why Use DataYab?

| Problem | DataYab Solution |
|---------|------------------|
| Your data is sensitive/private | ✅ Runs 100% locally, no internet needed |
| You don't know which model to use | ✅ Compares 10+ models automatically |
| You need to clean messy data | ✅ Auto-clean with one click |
| You want to understand your data | ✅ Generates charts and insights automatically |
| You need to make predictions | ✅ Simple form or batch CSV upload |

---

## Quick Start (5 Minutes)

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run
```bash
python app.py
```

### 3. Choose Interface
- `[1]` Desktop App (Tkinter) — Full features
- `[2]` Web Browser — Simple interface
- `[3]` Terminal — Quick testing

### 4. Load a Dataset
- Use the sample file: `data/dataset_Telco-Customer-Churn.csv`
- Or upload your own CSV, Excel, JSON, or Parquet file

### 5. Click "Auto Clean" — One click cleans your data

### 6. Select a Target Column (what you want to predict)

### 7. Click "Train Model" — DataYab builds a model

### 8. See Results — Accuracy, charts, insights, predictions

---

## What DataYab Can Do

### 📂 Load Data
| Format | Support |
|--------|---------|
| CSV | ✅ Auto-detects encoding and separators |
| Excel | ✅ (.xlsx, .xls) |
| JSON | ✅ Handles nested structures |
| Parquet | ✅ |

### 🔧 Clean Data
| Operation | What it does |
|-----------|---------------|
| Auto Clean | One-click complete cleaning |
| Remove Missing | Fill or drop missing values (11 strategies) |
| Remove Duplicates | Delete duplicate rows |
| Remove Outliers | Detect and remove outliers (Isolation Forest) |
| Encode Categorical | Convert text to numbers |
| Fix Data Types | Auto-convert columns to correct types |

### 📊 Explore Data
| Chart | Purpose |
|-------|---------|
| Histogram | See distribution of a column |
| Box Plot | Detect outliers |
| Correlation Heatmap | See relationships between columns |
| Scatter Plot | Compare two columns |
| Missing Values Map | Visualize missing data |

### 🤖 Train Models
| Model Type | Classification | Regression |
|------------|---------------|------------|
| Random Forest | ✅ | ✅ |
| XGBoost | ✅ | ✅ |
| Gradient Boosting | ✅ | ✅ |
| Logistic Regression | ✅ | ❌ |
| Linear Regression | ❌ | ✅ |
| SVM | ✅ | ✅ |
| KNN | ✅ | ✅ |
| Decision Tree | ✅ | ✅ |
| Naive Bayes | ✅ | ❌ |
| Ridge | ❌ | ✅ |

### 🎯 Feature Selection
| Method | What it does |
|--------|---------------|
| RF Importance | Shows which features matter most |
| Mutual Information | Finds non-linear relationships |
| Manual Selection | Choose features by name |
| PCA | Reduce dimensions while preserving information |

### 📈 Evaluate Models
| Metric/Plot | Purpose |
|-------------|---------|
| Accuracy, Precision, Recall, F1 | Classification performance |
| R², MAE, MSE, RMSE | Regression performance |
| Confusion Matrix | See correct/incorrect predictions |
| ROC Curve | Binary classification performance |
| Precision-Recall Curve | Imbalanced data evaluation |
| Cross-Validation | 5-fold validation score |
| Permutation Importance | Model-agnostic feature importance |
| Misclassified Samples | See which predictions were wrong |

### 🔮 Predict
- **Manual Input:** Dynamic form with dropdowns for categories
- **Batch Predictions:** Upload CSV, download results with predictions and confidence scores

### 💡 Insights
- **Health Score:** 0-100 grade (A-F) for data quality
- **Natural Language Insights:** Readable observations about your data
- **Recommendations:** What to do next

### 📄 Export
| Format | What's included |
|--------|-----------------|
| HTML Report | All charts, metrics, insights in one file |
| CSV | Cleaned dataset |
| Pipeline (.pkl) | Save trained model + all preprocessing steps |

---

## Three Interfaces

### 🖥️ Desktop App (Tkinter) — Full Features
```
┌─────────────────────────────────────────────────────────────┐
│  [Load] [Clean] [EDA] [Train] [Evaluate] [Insights] [Predict] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • 7 professional tabs                                      │
│  • Dark mode toggle                                         │
│  • Log panel with timestamps                                │
│  • Progress indicators                                      │
│  • ALL features available                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🌐 Web Interface (Browser) — Simple Demo
```
http://127.0.0.1:8001
```
- Drag-and-drop file upload
- One-click operations
- Mobile-friendly design
- Basic ML workflow

### 📟 CLI Mode (Terminal) — Quick Testing
```bash
python app.py --cli
```
Simple prompt-based workflow for quick testing.

---

## Project Structure

```
DataYab/
│
├── app.py                      # Main launcher (menu)
│
├── backend/
│   └── core.py                 # ML engine (everything important)
│
├── frontend/
│   ├── tinker.py               # Desktop GUI
│   ├── web.py                  # Web server
│   ├── templates/
│   │   └── index.html          # Web page
│   └── static/
│       └── app.js              # Web JavaScript
│
├── data/
│   ├── data_profile.py         # Data profiling tool
│   └── dataset_*.csv           # Sample dataset
│
└── requirements.txt            # Python packages
```

---

## Example Workflow (Customer Churn)

### Step 1: Load Data
```
File: customer_churn.csv
Rows: 7,043
Columns: 21
```

### Step 2: Auto Clean
```
✓ Fixed data types
✓ Removed constant columns
✓ Removed duplicates
✓ Filled missing values
✓ Capped outliers
```

### Step 3: Explore
```
Correlation Heatmap shows:
• tenure ↔ total_charges: r=0.83
• monthly_charges ↔ total_charges: r=0.65
```

### Step 4: Train Model
```
Target: Churn (Yes/No)
Model: Random Forest
Test Size: 20%
SMOTE: Enabled (dataset is imbalanced)
```

### Step 5: Evaluate
```
Accuracy: 0.82
Precision: 0.79
Recall: 0.81
F1 Score: 0.80
ROC-AUC: 0.87
```

### Step 6: Predict New Customer
```
Input:
• tenure: 12 months
• monthly_charges: $85
• contract_type: Month-to-month

Output:
Prediction: Churn ⚠️
Confidence: 87%
```

---

## Who Is This For?

| User | Why they'd use DataYab |
|------|------------------------|
| **Students** | Learn ML without cloud costs |
| **Analysts** | Quick data exploration and modeling |
| **Researchers** | Private data stays private |
| **Small Business** | Predict churn, sales, etc. without expensive tools |
| **Anyone** | Understand their data without coding |

---

## Comparison to Other Tools

| Feature | DataYab | Python (manual) | Jupyter | Cloud AutoML |
|---------|---------|-----------------|---------|--------------|
| No coding required | ✅ | ❌ | ❌ | ✅ |
| Local/Private | ✅ | ✅ | ✅ | ❌ |
| GUI Interface | ✅ | ❌ | ⚠️ | ✅ |
| Free | ✅ | ✅ | ✅ | ❌ |
| One-click cleaning | ✅ | ❌ | ❌ | ⚠️ |
| Model comparison | ✅ | ❌ | ❌ | ✅ |
| Export HTML reports | ✅ | ❌ | ⚠️ | ✅ |

---

## System Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.8+ |
| RAM | 4GB |
| Storage | 500MB |
| OS | Windows, macOS, Linux |
| Internet | Only for installation |

---

## Installation

```bash
# Clone or download the project
cd DataYab

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

---

## Current Status

| Component | Status |
|-----------|--------|
| Backend ML Engine | ✅ Production Ready |
| Desktop GUI | ✅ Production Ready |
| Web Interface | ✅ Stable Demo |
| CLI Mode | ✅ Stable |
| Data Profiler | ✅ Production Ready |

---

## What Makes DataYab Special

1. **100% Local** — Your data never leaves your computer
2. **Zero Cost** — No cloud fees, no subscriptions
3. **Complete Workflow** — From raw CSV to predictions in one app
4. **Three Interfaces** — Desktop, web, or terminal
5. **Professional Reports** — HTML export with charts and insights
6. **Production Ready** — Save/load pipelines for deployment

---

## Support

- **Documentation:** This document
- **Examples:** `data/dataset_Telco-Customer-Churn.csv`
- **Run:** `python app.py` then choose interface

---

**DataYab — Your Private Data Science Assistant** 🚀