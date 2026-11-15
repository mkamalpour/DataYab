"""
DataYab Streamlit Frontend
A modern, memory-efficient UI for the DataYab ML workbench.
Run with: streamlit run frontend/streamlit/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import base64
from pathlib import Path
import sys
import hashlib
import tempfile
from contextlib import contextmanager

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core import DataYabCore


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="DataYab | ML Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "DataYab - Local ML Workbench for Tabular Data"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 0.5rem 0;
    }
    .error-card {
        background: #fee2e2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: #10b981 !important;
        color: white !important;
    }
    div[data-testid="stExpander"] details {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
    }
    .stDownloadButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SESSION STATE ====================
if 'core' not in st.session_state:
    st.session_state.core = DataYabCore()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'split_done' not in st.session_state:
    st.session_state.split_done = False


# ==================== HELPER FUNCTIONS ====================
def get_core() -> DataYabCore:
    return st.session_state.core


def show_status(message: str, type: str = "info"):
    """Show status message in sidebar."""
    if type == "success":
        st.sidebar.success(message)
    elif type == "warning":
        st.sidebar.warning(message)
    elif type == "error":
        st.sidebar.error(message)
    else:
        st.sidebar.info(message)


def plotly_histogram(df: pd.DataFrame, column: str, bins: int = 30):
    """Create plotly histogram."""
    fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}")
    fig.update_layout(template="plotly_white", height=400)
    return fig


def plotly_boxplot(df: pd.DataFrame, column: str):
    """Create plotly boxplot."""
    fig = px.box(df, y=column, title=f"Box Plot — {column}")
    fig.update_layout(template="plotly_white", height=400)
    return fig


def plotly_correlation_heatmap(df: pd.DataFrame, max_cols: int = 20):
    """Create plotly correlation heatmap."""
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return go.Figure().update_layout(title="No numeric columns available")
    if num.shape[1] > max_cols:
        num = num.iloc[:, :max_cols]
    corr = num.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Correlation Matrix"
    )
    fig.update_layout(template="plotly_white", height=max(500, len(corr) * 25))
    return fig


def plotly_missing_heatmap(df: pd.DataFrame):
    """Create plotly missing values heatmap."""
    fig = px.imshow(
        df.isnull().astype(int),
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Missing Values Map"
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


def plotly_target_distribution(df: pd.DataFrame, target: str):
    """Create plotly target distribution."""
    if df[target].nunique() <= 20:
        vc = df[target].value_counts().reset_index()
        vc.columns = [target, 'count']
        fig = px.bar(vc, x=target, y='count', title=f"Target Distribution — {target}", text='count')
        fig.update_traces(textposition='outside')
    else:
        fig = px.histogram(df, x=target, nbins=40, title=f"Target Distribution — {target}")
    fig.update_layout(template="plotly_white", height=400)
    return fig


def plotly_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None):
    """Create plotly scatter plot."""
    if hue_col and hue_col in df.columns:
        fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, title=f"{x_col} vs {y_col}")
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
    fig.update_layout(template="plotly_white", height=500)
    return fig


def plotly_confusion_matrix(cm: list, labels: list):
    """Create plotly confusion matrix."""
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    fig.update_layout(template="plotly_white", height=400)
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return fig


def plotly_roc_curve(fpr: np.ndarray, tpr: np.ndarray):
    """Create plotly ROC curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=400
    )
    return fig


def plotly_precision_recall(precision: np.ndarray, recall: np.ndarray):
    """Create plotly precision-recall curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR'))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=400
    )
    return fig


def plotly_residuals(y_pred: np.ndarray, y_test: np.ndarray):
    """Create plotly residuals plot."""
    fig = px.scatter(x=y_pred, y=y_test - y_pred, title="Residuals Plot", labels={'x': 'Prediction', 'y': 'Residual'})
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_white", height=400)
    return fig


def plotly_feature_importance(importance: dict):
    """Create plotly feature importance."""
    if not importance:
        return None
    fi = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    top = dict(list(fi.items())[:20])
    fig = px.bar(
        x=list(top.values())[::-1],
        y=list(top.keys())[::-1],
        orientation='h',
        title="Feature Importance",
        labels={'x': 'Importance Score', 'y': 'Feature'}
    )
    fig.update_layout(template="plotly_white", height=max(400, len(top) * 25))
    return fig


def plotly_learning_curve(train_sizes, train_scores, test_scores):
    """Create plotly learning curve."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean, mode='lines+markers', name='Training',
        line=dict(color='blue'), error_y=dict(type='data', array=train_std)
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean, mode='lines+markers', name='Validation',
        line=dict(color='red'), error_y=dict(type='data', array=test_std)
    ))
    fig.update_layout(
        title="Learning Curve",
        xaxis_title="Training Examples",
        yaxis_title="Score",
        template="plotly_white",
        height=400
    )
    return fig


def df_to_csv_download(df: pd.DataFrame, filename: str):
    """Create CSV download button."""
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


@contextmanager
def temporary_file(data, suffix):
    with tempfile.TemporaryDirectory(prefix="datayab-") as directory:
        path = Path(directory) / ("input" + suffix)
        path.write_bytes(data)
        yield str(path)


def prepare_export(core, kind, key):
    extension = '.joblib' if kind == 'pipeline' else '.html'
    if st.button('Prepare ' + kind, key=key):
        try:
            with tempfile.TemporaryDirectory(prefix='datayab-export-') as directory:
                path = Path(directory) / ('datayab' + extension)
                result = (core.save_pipeline if kind == 'pipeline' else core.export_report)(str(path))
                if result.get('success'):
                    st.session_state[key + '_data'] = path.read_bytes()
                else:
                    st.error(result.get('message', 'Export failed'))
        except Exception as error:
            st.error(str(error))
    if key + '_data' in st.session_state:
        st.download_button('Download ' + kind, st.session_state[key + '_data'],
                           'datayab' + extension, key=key + '_download')


# ==================== MAIN APP ====================
def main():
    core = get_core()
    st.session_state.data_loaded = core.df is not None
    st.session_state.split_done = core.feature_engineer._raw_train is not None
    st.session_state.model_trained = core.model is not None

    current_table = id(core.df)
    if st.session_state.get('_analysis_table') != current_table:
        for key in list(st.session_state):
            if key.endswith('_data') or key in ('ts_result', 'cluster_result', 'anomaly_result'):
                del st.session_state[key]
        st.session_state._analysis_table = current_table

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0;">📊 DataYab</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Local ML Workbench for Tabular Data — Explore, Clean, Train, Predict</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar - Status
    with st.sidebar:
        st.header("📋 Status")
        if core.data_loader.df is not None:
            info = core.get_info()
            st.metric("Rows", f"{info['rows']:,}")
            st.metric("Columns", info['cols'])
            st.metric("Memory", f"{info['memory_mb']} MB")
            if core.target_col:
                st.metric("Target", core.target_col)
            if core.problem_type:
                st.metric("Problem Type", core.problem_type.upper())
            if core.model_name:
                st.metric("Active Model", core.model_name)
        else:
            st.info("No dataset loaded")

        st.divider()

        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Reset to Original Data", use_container_width=True):
            if core.data_loader.df is not None:
                core.reset_to_original()
                for key in list(st.session_state):
                    if key.endswith("_data") or key in ("ts_result", "cluster_result", "anomaly_result"):
                        del st.session_state[key]
                st.session_state.data_loaded = True
                st.session_state.split_done = False
                st.session_state.model_trained = False
                st.rerun()

        if core.model is not None:
            prepare_export(core, 'pipeline', 'sidebar_pipeline')
        if core.df is not None:
            prepare_export(core, 'report', 'sidebar_report')

    # Main tabs
    tabs = st.tabs([
        "📁 Upload Data",
        "📊 Explore",
        "🔧 Clean Data",
        "🎯 Features",
        "🤖 Train Model",
        "📈 Evaluate",
        "🔮 Predict",
        "💡 Insights",
        "✅ Validation",
        "📈 Time Series",
        "🎯 Clustering",
        "⚠️ Anomaly Detection"
    ])

    # ==================== TAB 1: UPLOAD DATA ====================
    with tabs[0]:
        st.header("📁 Upload Dataset")

        uploaded_file = st.file_uploader(
            "Choose a dataset file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'arr', 'arff'],
            help="Supported: CSV, Excel, JSON, Parquet, ARR, ARFF. Max 50MB."
        )

        upload_token = hashlib.sha256(uploaded_file.getvalue()).hexdigest() if uploaded_file is not None else None
        if uploaded_file is not None and upload_token != st.session_state.get('loaded_upload_token'):
            with st.spinner("Loading dataset..."):
                try:
                    with temporary_file(uploaded_file.getvalue(), Path(uploaded_file.name).suffix) as tmp_path:
                        result = core.load_file(tmp_path)
                except Exception as error:
                    st.error(str(error))
                    return

                if result["success"]:
                    core.file_path = uploaded_file.name
                    st.session_state.loaded_upload_token = upload_token
                    for export_key in list(st.session_state):
                        if export_key.endswith('_data') or export_key in ('ts_result', 'cluster_result', 'anomaly_result'):
                            del st.session_state[export_key]
                    st.session_state.data_loaded = True
                    st.session_state.split_done = False
                    st.session_state.model_trained = False
                    st.success(result["message"])

                    # Show preview
                    st.subheader("Data Preview")
                    st.dataframe(result["preview"], use_container_width=True)

                    # Show info
                    info = result["info"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Rows", f"{info['rows']:,}")
                    col2.metric("Columns", info['cols'])
                    col3.metric("Missing", f"{info['total_missing']:,}")
                    col4.metric("Duplicates", f"{info['duplicates']:,}")

                    with st.expander("Column Details"):
                        col_info = pd.DataFrame({
                            'Column': info['columns'],
                            'Type': [info['dtypes'][c] for c in info['columns']],
                            'Missing': [info['missing_count'][c] for c in info['columns']],
                            'Missing %': [info['missing_pct'][c] for c in info['columns']]
                        })
                        st.dataframe(col_info, use_container_width=True)
                else:
                    st.error(result["message"])

    # ==================== TAB 2: EXPLORE ====================
    with tabs[1]:
        if not st.session_state.data_loaded:
            st.warning("Please upload a dataset first.")
        else:
            st.header("📊 Data Exploration")

            # Data Preview
            with st.expander("📋 Data Preview", expanded=True):
                rows = st.slider("Rows to preview", 5, 100, 10)
                preview = core.get_preview(rows)
                st.dataframe(pd.DataFrame(preview['data'], columns=preview['columns']), use_container_width=True)

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Charts")
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Histogram", "Box Plot", "Correlation Heatmap", "Missing Values Map",
                     "Target Distribution", "Scatter Plot", "Pair Plot"]
                )

                numeric_cols = core.get_info()['numeric_cols']
                cat_cols = core.get_info()['categorical_cols']
                all_cols = core.get_info()['columns']

                if chart_type in ["Histogram", "Box Plot"]:
                    col = st.selectbox("Column", numeric_cols)
                    if col:
                        if chart_type == "Histogram":
                            bins = st.slider("Bins", 5, 100, 30)
                            fig = plotly_histogram(core.data_loader.df, col, bins)
                        else:
                            fig = plotly_boxplot(core.data_loader.df, col)
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Correlation Heatmap":
                    max_cols = st.slider("Max Columns", 5, 50, 20)
                    fig = plotly_correlation_heatmap(core.data_loader.df, max_cols)
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Missing Values Map":
                    fig = plotly_missing_heatmap(core.data_loader.df)
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Target Distribution":
                    target = st.selectbox("Target Column", all_cols,
                        index=all_cols.index(core.target_col) if core.target_col in all_cols else 0)
                    fig = plotly_target_distribution(core.data_loader.df, target)
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Scatter Plot":
                    x_col = st.selectbox("X Column", numeric_cols)
                    y_col = st.selectbox("Y Column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                    hue_col = st.selectbox("Color By (optional)", ["None"] + cat_cols)
                    if x_col and y_col:
                        fig = plotly_scatter(core.data_loader.df, x_col, y_col,
                            hue_col if hue_col != "None" else None)
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Pair Plot":
                    sel_cols = st.multiselect("Columns (max 5)", numeric_cols, default=numeric_cols[:5])
                    hue = st.selectbox("Color By (optional)", ["None"] + cat_cols)
                    if len(sel_cols) >= 2:
                        plot_df = core.data_loader.df[sel_cols + ([hue] if hue != "None" else [])].dropna()
                        if hue != "None":
                            plot_df[hue] = plot_df[hue].astype(str)
                        fig = px.scatter_matrix(plot_df, color=hue if hue != "None" else None)
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Statistics")
                if st.button("Show Statistics"):
                    stats = core.visualizer.get_stats()
                    st.dataframe(stats, use_container_width=True)
                    df_to_csv_download(stats.reset_index(), "statistics.csv")

                st.subheader("🔗 Top Correlations")
                if st.button("Show Top Correlations"):
                    target = st.selectbox("Target (optional)", ["None"] + all_cols, key="corr_target")
                    if target != "None":
                        corr_df = core.visualizer.get_top_correlations(target, 15)
                    else:
                        corr_df = core.visualizer.get_top_correlations(None, 15)
                    st.dataframe(corr_df, use_container_width=True)

                st.subheader("📋 Value Counts")
                vc_col = st.selectbox("Column for Value Counts", cat_cols + numeric_cols)
                if vc_col:
                    vc_df = core.visualizer.get_value_counts(vc_col, 20)
                    st.dataframe(vc_df, use_container_width=True)

    # ==================== TAB 3: CLEAN DATA ====================
    with tabs[2]:
        if not st.session_state.data_loaded:
            st.warning("Please upload a dataset first.")
        else:
            st.header("🔧 Data Cleaning")

            # Missing Values Report
            with st.expander("📊 Missing Values Report", expanded=True):
                miss_report = core.get_missing_report()
                if len(miss_report) > 0:
                    st.dataframe(miss_report, use_container_width=True)
                else:
                    st.success("No missing values!")

            # Cleaning Operations
            st.subheader("Cleaning Operations")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Missing Values**")
                strategy = st.selectbox(
                    "Strategy",
                    ["median", "mean", "mode", "ffill", "bfill", "zero", "unknown", "knn", "iterative", "drop_rows", "drop_cols"]
                )
                if st.button("Apply Missing Handling"):
                    with st.spinner("Processing..."):
                        result = core.handle_missing(strategy)
                        if result["success"]:
                            st.success(result["message"])
                            st.rerun()
                        else:
                            st.error(result["message"])

            with col2:
                st.markdown("**Other Operations**")
                if st.button("🗑️ Remove Duplicates"):
                    result = core.remove_duplicates()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()

                if st.button("🏷️ Encode Categorical (Label)"):
                    result = core.encode_categorical()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()

                if st.button("🗑️ Remove Constant Columns"):
                    result = core.remove_constant_columns()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()

                if st.button("🔄 Fix Data Types"):
                    result = core.fix_dtypes()
                    if result["success"]:
                        st.success(f"Converted: {result['converted']}")
                        st.rerun()

                if st.button("📊 Remove High Cardinality"):
                    result = core.remove_high_cardinality(50)
                    if result["success"]:
                        st.success(f"Removed: {result['removed']}")
                        st.rerun()

            with col3:
                st.markdown("**Outliers**")
                outlier_method = st.selectbox("Method", ["Isolation Forest", "IQR (per column)"])
                if outlier_method == "Isolation Forest":
                    contamination = st.slider("Contamination", 0.01, 0.3, 0.1)
                    if st.button("🔍 Detect Outliers"):
                        result = core.detect_outliers_isolation_forest(contamination)
                        if result["success"]:
                            st.info(f"Found {result['count']} outliers ({result['percent']}%)")
                    if st.button("🗑️ Remove Outliers"):
                        result = core.remove_outliers_isolation_forest(contamination)
                        if result["success"]:
                            st.success(result["message"])
                            st.rerun()

            # Auto Clean
            st.divider()
            st.subheader("✨ Auto Clean")
            target_for_auto = st.selectbox("Target Column (optional)", ["None"] + core.get_info()['columns'])
            if st.button("Run Auto Clean"):
                with st.spinner("Auto cleaning..."):
                    target = target_for_auto if target_for_auto != "None" else None
                    result = core.auto_clean(target)
                    if result["success"]:
                        for step in result["steps"]:
                            st.write(f"• {step}")
                        st.rerun()

    # ==================== TAB 4: FEATURES ====================
    with tabs[3]:
        if not st.session_state.data_loaded:
            st.warning("Please upload a dataset first.")
        elif not st.session_state.split_done:
            st.warning("Please split data first in the Train Model tab.")
        else:
            st.header("🎯 Feature Engineering")

            # Feature Selection
            st.subheader("📋 Feature Selection")

            col1, col2 = st.columns(2)

            with col1:
                method = st.selectbox(
                    "Selection Method",
                    ["rf", "mutual_info", "anova", "correlation", "rfe", "variance", "manual"],
                    format_func=lambda x: {
                        "rf": "Random Forest Importance",
                        "mutual_info": "Mutual Information",
                        "anova": "ANOVA F-test",
                        "correlation": "Correlation with Target",
                        "rfe": "Recursive Feature Elimination",
                        "variance": "Remove Constant Columns",
                        "manual": "Manual Selection"
                    }[x]
                )

                if method != "manual" and method != "variance":
                    k = st.number_input("Keep Top K Features", 1, len(core.feature_engineer._all_input_names), min(10, len(core.feature_engineer._all_input_names)))

                if method == "manual":
                    manual_features = st.text_input(
                        "Feature Names (comma-separated)",
                        placeholder="feature1, feature2, feature3"
                    )

                if st.button("Apply Selection"):
                    with st.spinner("Applying selection..."):
                        if method == "manual":
                            features = [f.strip() for f in manual_features.split(",") if f.strip()]
                            result = core.apply_selection("manual", features=features)
                        elif method == "variance":
                            result = core.apply_selection("variance")
                        else:
                            result = core.apply_selection(method, k)

                        if result["success"]:
                            st.success(f"Selected {result['total']} features: {', '.join(result['selected'][:10])}{'...' if len(result['selected']) > 10 else ''}")
                            st.rerun()
                        else:
                            st.error(result.get("message", "Failed"))

            with col2:
                if st.button("Show Feature Scores"):
                    if method in ["rf", "mutual_info", "anova", "correlation"]:
                        scores = core.get_feature_scores(method, 20)
                        fig = px.bar(
                            x=scores.values, y=scores.index,
                            orientation='h', title=f"Feature Scores: {method}"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("This method doesn't provide ranking scores.")

                if st.button("Restore All Features"):
                    result = core.restore_selection()
                    if result["success"]:
                        st.success(f"Restored {result['total']} features")
                        st.rerun()

            # Feature Status
            status = core.get_feature_status()
            st.info(f"Using {status['selected']}/{status['total']} input columns" +
                   (f" | {status['extraction'].upper()} extraction: {len(status['feature_names'])} components" if status['extraction'] else ""))

            # Feature Extraction
            st.divider()
            st.subheader("🧩 Feature Extraction")

            col1, col2 = st.columns(2)
            with col1:
                ext_method = st.selectbox(
                    "Extraction Method",
                    ["pca", "svd", "lda"],
                    format_func=lambda x: {"pca": "PCA", "svd": "Truncated SVD", "lda": "LDA"}[x]
                )
                n_components = st.number_input("Components", 1, 50, 2)

                if st.button("Apply Extraction"):
                    with st.spinner("Applying extraction..."):
                        result = core.apply_extraction(ext_method, n_components)
                        if result["success"]:
                            st.success(f"Created {result['n_components']} components")
                            if 'total_variance' in result:
                                st.info(f"Explained variance: {result['total_variance']:.2%}")
                            st.rerun()
                        else:
                            st.error(result.get("message", "Failed"))

            with col2:
                if st.button("Restore Original Features"):
                    result = core.restore_extraction()
                    if result["success"]:
                        st.success("Restored original features")
                        st.rerun()

            # Permutation Importance
            st.divider()
            st.subheader("🔀 Permutation Importance")
            if st.button("Calculate Permutation Importance"):
                with st.spinner("Calculating..."):
                    perm_df = core.get_permutation_importance(15)
                    if not perm_df.empty:
                        fig = px.bar(perm_df.head(15), x='importance', y='feature', orientation='h',
                            error_x='std', title="Permutation Importance")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(perm_df, use_container_width=True)

    # ==================== TAB 5: TRAIN MODEL ====================
    with tabs[4]:
        if not st.session_state.data_loaded:
            st.warning("Please upload a dataset first.")
        else:
            st.header("🤖 Train Model")

            # Target Selection
            all_cols = core.get_info()['columns']
            target_col = st.selectbox(
                "Target Column",
                all_cols,
                index=all_cols.index(core.target_col) if core.target_col in all_cols else 0
            )

            # Training Configuration
            col1, col2, col3 = st.columns(3)

            with col1:
                problem_type = st.selectbox(
                    "Problem Type",
                    ["auto", "classification", "regression"],
                    index=0
                )

                from backend.modules.modeling import model_catalog
                task = problem_type
                if task == "auto":
                    task = core.insights_generator._auto_detect_problem_type(target_col)
                model_type = st.selectbox(
                    "Model",
                    list(model_catalog(task))
                )

            with col2:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                use_smote = st.checkbox("Use SMOTE (imbalanced classification)")
                class_weight = st.selectbox("Class Weight", ["none", "balanced"])

            with col3:
                st.markdown("**Hyperparameters**")
                n_estimators = st.slider("n_estimators", 10, 500, 100)
                max_depth = st.slider("max_depth", 1, 50, 10)
                learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)

            # Split & Train
            if st.button("🚀 Split & Train", type="primary", use_container_width=True):
                with st.spinner("Splitting data and training model..."):
                    # Split
                    split_result = core.split_data(
                        target_col,
                        test_size=test_size,
                        problem_type=problem_type
                    )

                    if split_result["success"]:
                        st.session_state.split_done = True
                        st.success(f"Split: {split_result['train_size']} train / {split_result['test_size']} test ({split_result['problem_type']})")

                        # Train
                        hyperparams = {}
                        if model_type in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                            hyperparams["n_estimators"] = n_estimators
                            hyperparams["max_depth"] = max_depth
                        if model_type in ["Gradient Boosting", "XGBoost"]:
                            hyperparams["learning_rate"] = learning_rate
                        if model_type in ["Logistic Regression", "SVM", "SVR"]:
                            hyperparams["C"] = 1.0
                        if model_type == "KNN":
                            hyperparams["n_neighbors"] = 5

                        train_result = core.train(
                            model_type,
                            use_smote=use_smote,
                            class_weight=class_weight if class_weight != "none" else None,
                            **hyperparams
                        )

                        if train_result["success"]:
                            st.session_state.model_trained = True
                            st.success(f"Trained {train_result['model']} in {train_result['elapsed']}s")

                            # Show metrics
                            metrics = core.get_metrics()
                            st.subheader("Model Metrics")
                            if core.problem_type == "classification":
                                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                                mcol1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                                mcol2.metric("Balanced Acc", f"{metrics.get('balanced_accuracy', 0):.4f}")
                                mcol3.metric("F1 (weighted)", f"{metrics.get('f1', 0):.4f}")
                                mcol4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
                            else:
                                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                                mcol1.metric("R²", f"{metrics.get('r2', 0):.4f}")
                                mcol2.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                                mcol3.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                                mcol4.metric("RMSE Skill", f"{metrics.get('rmse_skill', 0):.4f}")
                        else:
                            st.error(train_result.get("message", "Training failed"))
                    else:
                        st.error(split_result.get("message", "Split failed"))

            # Compare All Models
            st.divider()
            st.subheader("🏆 Compare All Models")
            if st.button("Compare All Models"):
                with st.spinner("Training and comparing all models..."):
                    split_result = core.split_data(target_col, test_size=test_size, problem_type=problem_type)
                    if split_result["success"]:
                        comparison = core.train_all_models(use_smote=use_smote, class_weight=class_weight if class_weight != "none" else None)
                        st.dataframe(comparison, use_container_width=True)

                        # Highlight best
                        best_model = core.model_name
                        st.success(f"Best model: {best_model}")

            # Hyperparameter Tuning
            st.divider()
            st.subheader("🎯 Hyperparameter Tuning")

            tune_method = st.selectbox("Tuning Method", ["Optuna", "GridSearchCV"])
            tune_model = st.selectbox("Model to Tune", ["Random Forest", "Gradient Boosting", "XGBoost", "Logistic Regression", "SVM", "KNN"])
            n_trials = st.number_input("Trials (Optuna)", 10, 500, 50)
            cv_folds = st.number_input("CV Folds", 2, 10, 3)

            if st.button("Run Hyperparameter Tuning"):
                with st.spinner(f"Running {tune_method}..."):
                    # Need to split first if not done
                    if not st.session_state.split_done:
                        core.split_data(target_col, test_size=test_size, problem_type=problem_type)

                    if tune_method == "Optuna":
                        result = core.tune_with_optuna(tune_model, n_trials=n_trials, cv_folds=cv_folds)
                    else:
                        # Grid search params
                        param_grids = {
                            "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]},
                            "Gradient Boosting": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
                            "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
                            "SVM": {"C": [0.01, 0.1, 1, 10, 100]},
                            "KNN": {"n_neighbors": [3, 5, 7, 9, 11]},
                        }
                        result = core.tune_with_gridsearch(tune_model, param_grids.get(tune_model, {}), cv_folds=cv_folds)

                    if result["success"]:
                        st.success(f"Best Score: {result['best_score']:.4f}")
                        st.json(result["best_params"])
                    else:
                        st.error(result.get("message", "Tuning failed"))

            # Stacked Model
            st.divider()
            st.subheader("🔗 Stacked Model")
            trained_models = core.get_trained_models()
            if len(trained_models) >= 2:
                base1 = st.selectbox("Base Model 1", trained_models, key="stack1")
                base2 = st.selectbox("Base Model 2", trained_models, key="stack2")
                meta = st.selectbox("Meta Learner", ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"])

                if st.button("Train Stacked Model") and base1 != base2:
                    with st.spinner("Training stacked model..."):
                        result = core.train_stacked_model(base1, base2, meta)
                        if result["success"]:
                            st.success(f"Stacked model trained: {result['model']}")
                            st.session_state.model_trained = True
                            st.rerun()
                        else:
                            st.error(result.get("message", "Failed"))
            else:
                st.info("Train at least 2 models to create a stacked model.")

    # ==================== TAB 6: EVALUATE ====================
    with tabs[5]:
        if not st.session_state.model_trained:
            st.warning("Please train a model first.")
        elif core.feature_engineer._raw_test is None:
            st.info("Imported model: no holdout data. Use Predict with new records.")
        else:
            st.header("📈 Model Evaluation")

            # Model Selector
            trained_models = core.get_trained_models()
            if len(trained_models) > 1:
                selected_model = st.selectbox("Select Model", trained_models, index=trained_models.index(core.model_name))
                if st.button("Load Model"):
                    core.select_model(selected_model)
                    st.success(f"Loaded {selected_model}")
                    st.rerun()

            # Metrics
            st.subheader("📊 Metrics")
            metrics = core.get_metrics()

            if core.problem_type == "classification":
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                mcol2.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.4f}")
                mcol3.metric("Macro F1", f"{metrics.get('macro_f1', 0):.4f}")
                mcol4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                mcol2.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                mcol3.metric("F1 (weighted)", f"{metrics.get('f1', 0):.4f}")
                mcol4.metric("Majority Baseline", f"{metrics.get('majority_baseline_accuracy', 0):.4f}")

                with st.expander("Classification Report"):
                    st.text(metrics.get('classification_report', 'N/A'))
            else:
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("R²", f"{metrics.get('r2', 0):.4f}")
                mcol2.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                mcol3.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                mcol4.metric("RMSE Skill", f"{metrics.get('rmse_skill', 0):.4f}")

                mcol1, mcol2 = st.columns(2)
                mcol1.metric("Baseline MAE", f"{metrics.get('baseline_mae', 0):.4f}")
                mcol2.metric("Baseline RMSE", f"{metrics.get('baseline_rmse', 0):.4f}")

            # Visualizations
            st.divider()
            st.subheader("📈 Visualizations")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                if core.problem_type == "classification":
                    if st.button("Confusion Matrix"):
                        cm = core.get_confusion_matrix()
                        fig = plotly_confusion_matrix(cm["matrix"], cm["labels"])
                        st.plotly_chart(fig, use_container_width=True)

                    if st.button("ROC Curve"):
                        try:
                            from sklearn.metrics import roc_curve
                            core.model_trainer._compute_predictions()
                            fpr, tpr, _ = roc_curve(core.feature_engineer.y_test, core.model_trainer.y_proba[:, 1])
                            fig = plotly_roc_curve(fpr, tpr)
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.error("ROC requires binary classification with probabilities")

                    if st.button("Precision-Recall Curve"):
                        try:
                            from sklearn.metrics import precision_recall_curve
                            precision, recall, _ = precision_recall_curve(core.feature_engineer.y_test, core.model_trainer.y_proba[:, 1])
                            fig = plotly_precision_recall(precision, recall)
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.error("PR curve requires binary classification with probabilities")
                else:
                    if st.button("Residuals Plot"):
                        core.model_trainer._compute_predictions()
                        fig = plotly_residuals(core.model_trainer.y_pred, core.feature_engineer.y_test)
                        st.plotly_chart(fig, use_container_width=True)

            with viz_col2:
                if st.button("Feature Importance"):
                    fig = plotly_feature_importance(core.feature_importance)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No feature importance available")

                if st.button("Learning Curve"):
                    try:
                        from sklearn.model_selection import learning_curve
                        train_sizes, train_scores, test_scores = learning_curve(
                            core.model, core.feature_engineer._raw_train, core.feature_engineer.y_train,
                            train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=1,
                            scoring="accuracy" if core.problem_type == "classification" else "r2"
                        )
                        fig = plotly_learning_curve(train_sizes, train_scores, test_scores)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Learning curve failed: {e}")

                if st.button("Partial Dependence"):
                    features = list(core.feature_importance.keys())[:5] if core.feature_importance else core.feature_engineer.feature_names[:5]
                    feat = st.selectbox("Feature", features)
                    if feat:
                        try:
                            from sklearn.inspection import PartialDependenceDisplay
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(8, 5))
                            PartialDependenceDisplay.from_estimator(
                                core.model, core.feature_engineer._raw_train,
                                [core.feature_engineer.feature_names.index(feat)],
                                feature_names=core.feature_engineer.feature_names, ax=ax
                            )
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Partial dependence failed: {e}")

            # Cross Validation
            st.divider()
            st.subheader("🔍 Cross-Validation")
            if st.button("Run Cross-Validation"):
                with st.spinner("Running CV..."):
                    cv = core.get_cross_val_score(5)
                    st.metric(f"Mean {cv['metric'].upper()}", f"{cv['mean']:.4f}")
                    st.metric("Std", f"{cv['std']:.4f}")
                    st.write("Fold Scores:", [f"{s:.4f}" for s in cv['scores']])
                    if 'warning' in cv:
                        st.warning(cv['warning'])

            # Misclassified
            if core.problem_type == "classification":
                st.divider()
                st.subheader("⚠️ Misclassified Samples")
                mis_df = core.get_misclassified_samples(30)
                if len(mis_df) > 0:
                    st.dataframe(mis_df, use_container_width=True)
                    df_to_csv_download(mis_df, "misclassified_samples.csv")
                else:
                    st.success("No misclassified samples!")

    # ==================== TAB 7: PREDICT ====================
    with tabs[6]:
        if not st.session_state.model_trained:
            st.warning("Please train a model first.")
        else:
            st.header("🔮 Make Predictions")

            predict_mode = st.radio(
                "Prediction Mode",
                (["Manual Input", "Test Record Comparison", "Full Test Set", "Batch CSV"] if core.feature_engineer._raw_test is not None else ["Manual Input", "Batch CSV"]),
                horizontal=True
            )

            if predict_mode == "Manual Input":
                st.subheader("Manual Input Form")
                schema = core.get_input_form_schema()

                if schema:
                    inputs = {}
                    cols = st.columns(2)
                    for i, field in enumerate(schema):
                        with cols[i % 2]:
                            if field["type"] == "categorical":
                                inputs[field["name"]] = st.selectbox(
                                    field["name"], field.get("options", []), key=f"pred_{field['name']}"
                                )
                            else:
                                inputs[field["name"]] = st.number_input(
                                    field["name"], value=float(field.get("mean", 0)), key=f"pred_{field['name']}"
                                )

                    if st.button("🔮 Predict", type="primary"):
                        result = core.predict(inputs)
                        st.success(f"Prediction: **{result['prediction']}**")
                        if 'confidence' in result:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        if 'probabilities' in result:
                            st.write("Probabilities:")
                            for k, v in result['probabilities'].items():
                                st.write(f"  • {k}: {v:.2%}")

            elif predict_mode == "Test Record Comparison":
                st.subheader("Test Record Comparison")

                n_test = len(core.feature_engineer.y_test)
                test_idx = st.number_input("Test Record Index", 0, n_test - 1, 0)

                if st.button("🔮 Predict & Compare"):
                    # Get test record
                    test_row = core.feature_engineer._raw_test.iloc[[test_idx]].to_dict('records')[0]

                    # Predict
                    result = core.predict(test_row)

                    # Get actual
                    if core.problem_type == "classification":
                        actual = core.feature_engineer.target_encoder.inverse_transform([core.feature_engineer.y_test[test_idx]])[0]
                    else:
                        actual = core.feature_engineer.y_test[test_idx]

                    # Display
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", result['prediction'])
                    col2.metric("Actual", actual)

                    if core.problem_type == "classification":
                        match = result['prediction'] == actual
                        st.metric("Match", "✅ YES" if match else "❌ NO")
                        if 'confidence' in result:
                            st.metric("Confidence", f"{result['confidence']:.2%}")

                    with st.expander("Input Features"):
                        st.json(test_row)

            elif predict_mode == "Full Test Set":
                st.subheader("Full Test Set Predictions")

                limit = st.number_input("Limit", 1, len(core.feature_engineer.y_test), min(50, len(core.feature_engineer.y_test)))

                if st.button("Generate Predictions"):
                    with st.spinner("Predicting..."):
                        pred_df = core.predict_on_test_set(list(range(limit)))
                        st.dataframe(pred_df, use_container_width=True)
                        df_to_csv_download(pred_df, "test_predictions.csv")

            elif predict_mode == "Batch CSV":
                st.subheader("Batch Prediction from CSV")

                batch_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'])

                if batch_file is not None:
                    batch_df = pd.read_csv(batch_file)
                    st.write("Preview:")
                    st.dataframe(batch_df.head(), use_container_width=True)

                    if st.button("Run Batch Prediction"):
                        with st.spinner("Predicting..."):
                            with temporary_file(batch_file.getvalue(), '.csv') as in_path:
                                result_df = core.predict_from_csv(in_path)
                            st.success(f"Predictions complete: {len(result_df)} rows")
                            st.dataframe(result_df, use_container_width=True)
                            df_to_csv_download(result_df, "batch_predictions.csv")

    # ==================== TAB 8: INSIGHTS ====================
    with tabs[7]:
        if not st.session_state.data_loaded:
            st.warning("Please upload a dataset first.")
        else:
            st.header("💡 Data Insights")

            if st.button("✨ Generate Insights", type="primary"):
                with st.spinner("Generating insights..."):
                    insights = core.generate_insights()
                    for insight in insights:
                        st.write(insight)

            st.divider()

            # Health Score
            st.subheader("📊 Dataset Health Score")
            if st.button("Show Health Score"):
                health = core.get_dataset_health_score()

                col1, col2 = st.columns([1, 3])
                with col1:
                    # Grade color
                    grade_colors = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '🔴'}
                    st.metric("Score", f"{health['score']}/100")
                    st.metric("Grade", f"{grade_colors.get(health['grade'], '⚪')} {health['grade']}")

                with col2:
                    if health['issues']:
                        for issue in health['issues']:
                            st.warning(f"⚠️ {issue}")
                    else:
                        st.success("✅ No issues detected!")

                # Progress bars
                info = core.get_info()
                total_cells = info['rows'] * info['cols']
                missing_pct = (info['total_missing'] / total_cells * 100) if total_cells > 0 else 0
                dup_pct = (info['duplicates'] / info['rows'] * 100) if info['rows'] > 0 else 0

                st.write("**Completeness**")
                st.progress(max(0, 1 - missing_pct / 100))
                st.write(f"{100 - missing_pct:.1f}% complete")

                st.write("**Uniqueness**")
                st.progress(max(0, 1 - dup_pct / 100))
                st.write(f"{100 - dup_pct:.1f}% unique")

            # Model Card
            if st.session_state.model_trained:
                st.divider()
                st.subheader("📋 Model Card")
                if st.button("Generate Model Card"):
                    card = core.generate_model_card()
                    st.json(card)

    # ==================== TAB 9: VALIDATION ====================
    with tabs[8]:
        if not st.session_state.data_loaded:
            st.warning("Please upload a dataset first.")
        else:
            st.header("✅ Pipeline Validation")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🔍 Validate Pipeline", type="primary"):
                    with st.spinner("Validating..."):
                        result = core.validate_pipeline()
                        if result['valid']:
                            st.success("✅ Pipeline Valid")
                        else:
                            st.error("❌ Pipeline Invalid")

                        for err in result['errors']:
                            st.error(f"❌ {err}")
                        for warn in result['warnings']:
                            st.warning(f"⚠️ {warn}")

            with col2:
                if st.button("🚀 Validate for Deployment"):
                    with st.spinner("Validating..."):
                        result = core.validate_before_deployment()
                        if result['valid']:
                            st.success("✅ Ready for Deployment")
                        else:
                            st.error("❌ Not Ready for Deployment")

                        for err in result['errors']:
                            st.error(f"❌ {err}")
                        for warn in result['warnings']:
                            st.warning(f"⚠️ {warn}")
                        if 'prediction_test' in result:
                            st.info(f"Prediction Test: {result['prediction_test']}")

            with col3:
                if st.button("📊 Data Quality Report"):
                    with st.spinner("Generating..."):
                        report = core.get_data_quality_report()
                        st.metric("Quality Score", f"{report['quality_score']}/100 ({report['quality_grade']})")
                        st.metric("Rows", f"{report['shape'][0]:,}")
                        st.metric("Columns", report['shape'][1])
                        st.metric("Missing", f"{sum(report['missing'].values()):,}")
                        st.metric("Duplicates", f"{report['duplicates']:,}")

                        if report['constant_columns']:
                            st.warning(f"Constant columns: {report['constant_columns']}")
                        if report['high_cardinality']:
                            st.warning(f"High cardinality: {report['high_cardinality']}")
                        if report['issues']:
                            for issue in report['issues']:
                                st.warning(f"⚠️ {issue}")

            # Export/Import
            st.divider()
            st.subheader("📦 Export / Import")

            exp_col1, exp_col2, exp_col3 = st.columns(3)

            with exp_col1:
                if core.model is not None:
                    prepare_export(core, 'pipeline', 'validation_pipeline')
            with exp_col2:
                st.download_button('Download cleaned CSV', core.df.to_csv(index=False).encode('utf-8-sig'), 'datayab_cleaned.csv', 'text/csv')
            with exp_col3:
                prepare_export(core, 'report', 'validation_report')
            st.subheader('Import model package')
            st.caption('Only open trusted joblib files: they can execute code. Imported models support prediction; they contain no holdout dataset.')
            import_file = st.file_uploader('Trusted .joblib pipeline', type=['joblib'])
            trusted = st.checkbox('I trust this model file')
            if st.button('Import Pipeline', disabled=import_file is None or not trusted):
                try:
                    with temporary_file(import_file.getvalue(), '.joblib') as path:
                        result = core.load_pipeline(path)
                    if result.get('success'):
                        st.session_state.model_trained = True
                        st.session_state.split_done = False
                        st.rerun()
                    else:
                        st.error(result.get('message', 'Import failed'))
                except Exception as error:
                    st.error(str(error))

    # ==================== ADVANCED ANALYTICS ====================
    with tabs[9]:
        st.header("Time Series")
        st.caption("Analysis uses a separate copy sorted by time. Use unique, regularly spaced dates. Forecasts have not been validated on a time-ordered holdout.")
        if core.df is None:
            st.info("Upload a dataset first.")
        else:
            numeric = core.df.select_dtypes(include=np.number).columns.tolist()
            dates = core.advanced_analytics.detect_time_column()
            if not dates or not numeric:
                st.info("Need a date column and a numeric value column.")
            else:
                date = st.selectbox("Date column", dates, key="ts_date")
                value = st.selectbox("Value column", numeric, key="ts_value")
                method = st.selectbox("Analysis", ["Decomposition", "ARIMA forecast", "Stationarity (ADF)"])
                period = st.number_input("Seasonal period", 2, 1000, 12)
                steps = st.number_input("Forecast steps", 1, 1000, 12)
                if st.button("Run time series analysis"):
                    try:
                        from backend.modules.data_loader import DataLoader
                        from backend.modules.advanced_analytics import AdvancedAnalytics
                        loader = DataLoader()
                        frame = core.df[[date, value]].copy()
                        frame.index = pd.to_datetime(frame.pop(date), errors='raise')
                        frame = frame.sort_index()
                        if frame.index.has_duplicates or frame.index.hasnans:
                            raise ValueError("Dates must be unique and present. Aggregate duplicates first.")
                        if len(frame) < 3 or pd.infer_freq(frame.index) is None:
                            raise ValueError("Use regular time intervals; fill or aggregate missing intervals first.")
                        if not np.isfinite(frame[value]).all():
                            raise ValueError("Series values must be finite with no missing observations.")
                        loader.df = frame
                        aa = AdvancedAnalytics(loader)
                        if method == "Decomposition":
                            result = aa.time_series_decomposition(value, int(period))
                            if result.get('success'):
                                result['table'] = pd.DataFrame(result['decomposition'], index=frame.index)
                        elif method == "ARIMA forecast":
                            result = aa.forecast_arima(value, int(steps), (1, 1, 1))
                            if result.get('success'):
                                ci = np.asarray(result['confidence_intervals'])
                                result['table'] = pd.DataFrame({'forecast': result['forecast'], 'lower_95': ci[:, 0], 'upper_95': ci[:, 1]}, index=result['future_dates'])
                        else:
                            from statsmodels.tsa.stattools import adfuller
                            stat, pv, *_ = adfuller(frame[value])
                            result = {'success': True, 'ADF statistic': stat, 'p-value': pv}
                        if not result.get('success'): raise ValueError(result.get('message'))
                        st.session_state.ts_result = result
                    except Exception as error:
                        st.error(str(error))
                result = st.session_state.get('ts_result')
                if result:
                    if 'table' in result:
                        st.line_chart(result['table'])
                        df_to_csv_download(result['table'].reset_index(), 'time_series.csv')
                    st.json({k: v for k, v in result.items() if k not in ('table', 'decomposition')})
                    st.caption("ADF p < 0.05 provides evidence against a unit root, not proof of forecast accuracy.")

    with tabs[10]:
        st.header("Clustering")
        st.caption("Numeric inputs are standardized; rows with missing inputs are excluded. Cluster numbers are arbitrary labels; DBSCAN uses -1 for noise. Working data is preserved.")
        if core.df is None:
            st.info("Upload a dataset first.")
        else:
            numeric = core.df.select_dtypes(include=np.number).columns.tolist()
            features = st.multiselect("Clustering features", numeric, default=numeric[:5])
            method = st.selectbox("Clustering method", ["K-Means", "DBSCAN", "Hierarchical", "Gaussian mixture"])
            count = st.number_input("Number of clusters", 2, 30, 3)
            eps = st.number_input("DBSCAN epsilon", .01, 100., .5)
            minimum = st.number_input("DBSCAN minimum samples", 2, 1000, 5)
            if st.button("Run clustering"):
                try:
                    if not features: raise ValueError("Select numeric features first.")
                    if not np.isfinite(core.df[features].dropna()).all().all(): raise ValueError("Clean infinite values first.")
                    aa = core.advanced_analytics
                    methods = {"K-Means": lambda: aa.run_kmeans(features, int(count)), "DBSCAN": lambda: aa.run_dbscan(features, eps, int(minimum)), "Hierarchical": lambda: aa.run_hierarchical(features, int(count)), "Gaussian mixture": lambda: aa.run_gmm(features, int(count))}
                    result = methods[method]()
                    if not result.get('success'): raise ValueError(result.get('message'))
                    st.session_state.cluster_result = (result, features)
                except Exception as error:
                    st.error(str(error))
            if st.session_state.get('cluster_result'):
                result, selected = st.session_state.cluster_result
                frame = result['result_df']
                view = frame.sample(min(5000, len(frame)), random_state=42).copy()
                view['cluster'] = view['cluster'].astype(str)
                fig = px.scatter(view, x=selected[0], y=selected[1], color='cluster') if len(selected) > 1 else px.histogram(view, x=selected[0], color='cluster')
                st.plotly_chart(fig, use_container_width=True)
                st.json({k: v for k, v in result.items() if k not in ('result_df', 'labels', 'probabilities', 'covariances')})
                df_to_csv_download(frame, 'cluster_assignments.csv')

    with tabs[11]:
        st.header("Anomaly Detection")
        st.caption("Flags are prompts for investigation. Incomplete rows are excluded and the working table stays intact. Z-score and IQR flag a row when any selected feature is unusual.")
        if core.df is None:
            st.info("Upload a dataset first.")
        else:
            numeric = core.df.select_dtypes(include=np.number).columns.tolist()
            features = st.multiselect("Anomaly features", numeric, default=numeric[:5])
            method = st.selectbox("Anomaly method", ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Z-score", "IQR"])
            fraction = st.slider("Expected anomaly fraction", .01, .5, .1)
            neighbors = st.number_input("LOF neighbors", 2, 1000, 20)
            if st.button("Detect anomalies", key='advanced_anomalies'):
                try:
                    if not features: raise ValueError("Select numeric features first.")
                    if not np.isfinite(core.df[features].dropna()).all().all(): raise ValueError("Clean infinite values first.")
                    aa = core.advanced_analytics
                    methods = {"Isolation Forest": lambda: aa.detect_anomalies_isolation_forest(features, fraction), "Local Outlier Factor": lambda: aa.detect_anomalies_lof(features, int(neighbors), fraction), "One-Class SVM": lambda: aa.detect_anomalies_one_class_svm(features, fraction), "Z-score": lambda: aa.detect_anomalies_zscore(features), "IQR": lambda: aa.detect_anomalies_iqr(features)}
                    result = methods[method]()
                    if not result.get('success'): raise ValueError(result.get('message'))
                    st.session_state.anomaly_result = (result, features)
                except Exception as error:
                    st.error(str(error))
            if st.session_state.get('anomaly_result'):
                result, selected = st.session_state.anomaly_result
                frame = result['result_df']
                st.metric("Flagged records", result['n_anomalies'])
                st.metric("Flagged percentage", f"{result['anomaly_pct']:.2f}%")
                df_to_csv_download(frame.loc[frame['is_anomaly']], 'anomaly_records.csv')
                df_to_csv_download(frame, 'all_scored_records.csv')

if __name__ == "__main__":
    main()
