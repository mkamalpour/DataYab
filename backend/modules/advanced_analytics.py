"""Advanced analytics module: time series, clustering, anomaly detection."""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats
from scipy.signal import find_peaks


class AdvancedAnalytics:
    """Advanced analytics for tabular data: time series, clustering, anomaly detection."""

    def __init__(self, data_loader):
        self.dl = data_loader

    def _require_data(self):
        if self.dl.df is None:
            raise ValueError("No dataset loaded.")

    # ==================== TIME SERIES ANALYSIS ====================

    def detect_time_column(self) -> List[str]:
        """Detect potential time/date columns."""
        self._require_data()
        time_cols = []
        for col in self.dl.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.dl.df[col]):
                time_cols.append(col)
            elif self.dl.df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(self.dl.df[col].dropna().iloc[:100])
                    time_cols.append(col)
                except:
                    pass
        return time_cols

    def set_time_index(self, time_col: str) -> Dict:
        """Set a column as time index for time series analysis."""
        self._require_data()
        if time_col not in self.dl.df.columns:
            return {"success": False, "message": f"Column '{time_col}' not found"}

        try:
            df = self.dl.df.copy()
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col).sort_index()
            self.dl.df = df
            self.dl._log(f"Set '{time_col}' as time index")
            return {"success": True, "message": f"Time index set to '{time_col}'", "shape": df.shape}
        except Exception as e:
            return {"success": False, "message": f"Failed to set time index: {e}"}

    def resample_time_series(self, rule: str, agg: str = "mean") -> Dict:
        """Resample time series data."""
        self._require_data()
        if not isinstance(self.dl.df.index, pd.DatetimeIndex):
            return {"success": False, "message": "DataFrame must have DatetimeIndex"}

        try:
            numeric_cols = self.dl.df.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                return {"success": False, "message": "No numeric columns to resample"}

            if agg == "mean":
                resampled = self.dl.df[numeric_cols].resample(rule).mean()
            elif agg == "sum":
                resampled = self.dl.df[numeric_cols].resample(rule).sum()
            elif agg == "last":
                resampled = self.dl.df[numeric_cols].resample(rule).last()
            elif agg == "first":
                resampled = self.dl.df[numeric_cols].resample(rule).first()
            elif agg == "ohlc":
                resampled = self.dl.df[numeric_cols].resample(rule).ohlc()
            else:
                return {"success": False, "message": f"Unknown aggregation: {agg}"}

            self.dl.df = resampled
            self.dl._log(f"Resampled to {rule} using {agg}")
            return {"success": True, "message": f"Resampled to {rule}", "shape": resampled.shape}
        except Exception as e:
            return {"success": False, "message": f"Resample failed: {e}"}

    def time_series_decomposition(self, column: str, period: int = None) -> Dict:
        """Decompose time series into trend, seasonal, residual."""
        self._require_data()
        if not isinstance(self.dl.df.index, pd.DatetimeIndex):
            return {"success": False, "message": "DataFrame must have DatetimeIndex"}
        if column not in self.dl.df.columns:
            return {"success": False, "message": f"Column '{column}' not found"}

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            series = self.dl.df[column].dropna()
            if period is None:
                # Auto-detect period based on frequency
                freq = pd.infer_freq(series.index)
                if freq:
                    if 'D' in freq: period = 7  # Weekly
                    elif 'H' in freq: period = 24  # Daily
                    elif 'M' in freq: period = 12  # Yearly
                    elif 'Q' in freq: period = 4  # Yearly
                    else: period = 12
                else:
                    period = min(12, len(series) // 2)

            if len(series) < 2 * period:
                return {"success": False, "message": f"Need at least {2*period} observations for period {period}"}

            decomposition = seasonal_decompose(series, model='additive', period=period)

            result_df = pd.DataFrame({
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            })

            return {
                "success": True,
                "decomposition": result_df.to_dict(orient='list'),
                "period": period,
                "message": f"Decomposed with period={period}"
            }
        except ImportError:
            return {"success": False, "message": "statsmodels required for decomposition. Install: pip install statsmodels"}
        except Exception as e:
            return {"success": False, "message": f"Decomposition failed: {e}"}

    def detect_seasonality(self, column: str, max_lag: int = 50) -> Dict:
        """Detect seasonality using autocorrelation."""
        self._require_data()
        if column not in self.dl.df.columns:
            return {"success": False, "message": f"Column '{column}' not found"}

        series = self.dl.df[column].dropna()
        if len(series) < 20:
            return {"success": False, "message": "Need at least 20 observations"}

        # Compute autocorrelation
        acf_values = []
        for lag in range(1, min(max_lag, len(series)//2)):
            corr = series.autocorr(lag=lag)
            if not np.isnan(corr):
                acf_values.append({"lag": lag, "autocorr": corr})

        # Find peaks in ACF
        acf_array = np.array([x['autocorr'] for x in acf_values])
        peaks, _ = find_peaks(acf_array, height=0.3, distance=5)

        seasonal_periods = [acf_values[p]['lag'] for p in peaks if acf_values[p]['autocorr'] > 0.3]

        return {
            "success": True,
            "acf": acf_values,
            "seasonal_periods": seasonal_periods,
            "message": f"Detected seasonal periods: {seasonal_periods}" if seasonal_periods else "No strong seasonality detected"
        }

    def forecast_arima(self, column: str, steps: int = 10, order: Tuple = None) -> Dict:
        """Simple ARIMA forecasting."""
        self._require_data()
        if column not in self.dl.df.columns:
            return {"success": False, "message": f"Column '{column}' not found"}

        try:
            from statsmodels.tsa.arima.model import ARIMA

            series = self.dl.df[column].dropna()
            if len(series) < 30:
                return {"success": False, "message": "Need at least 30 observations for ARIMA"}

            # Auto-select order if not provided
            if order is None:
                # Simple heuristic
                order = (1, 1, 1)

            model = ARIMA(series, order=order)
            fitted = model.fit()

            forecast = fitted.forecast(steps=steps)
            conf_int = fitted.get_forecast(steps=steps).conf_int()

            # Create future dates
            last_date = series.index[-1]
            if isinstance(series.index, pd.DatetimeIndex):
                freq = pd.infer_freq(series.index) or 'D'
                future_dates = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
            else:
                future_dates = range(len(series), len(series) + steps)

            return {
                "success": True,
                "forecast": forecast.tolist(),
                "confidence_intervals": conf_int.values.tolist(),
                "future_dates": [str(d) for d in future_dates],
                "aic": fitted.aic,
                "order": order
            }
        except ImportError:
            return {"success": False, "message": "statsmodels required for ARIMA. Install: pip install statsmodels"}
        except Exception as e:
            return {"success": False, "message": f"ARIMA failed: {e}"}

    # ==================== CLUSTERING ====================

    def find_optimal_clusters(self, columns: List[str] = None, max_k: int = 10) -> Dict:
        """Find optimal number of clusters using elbow method and silhouette score."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        if len(columns) < 2:
            return {"success": False, "message": "Need at least 2 numeric columns for clustering"}

        data = self.dl.df[columns].dropna()
        if len(data) < 10:
            return {"success": False, "message": "Need at least 10 rows for clustering"}

        # Scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        results = []
        for k in range(2, min(max_k + 1, len(data))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_scaled)

            inertia = kmeans.inertia_
            silhouette = silhouette_score(data_scaled, labels) if len(set(labels)) > 1 else -1
            calinski = calinski_harabasz_score(data_scaled, labels) if len(set(labels)) > 1 else -1
            davies = davies_bouldin_score(data_scaled, labels) if len(set(labels)) > 1 else -1

            results.append({
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies
            })

        # Find optimal k (max silhouette)
        best_k = max(results, key=lambda x: x['silhouette'])['k'] if results else 2

        return {
            "success": True,
            "results": results,
            "optimal_k": best_k,
            "columns_used": columns
        }

    def run_kmeans(self, columns: List[str] = None, n_clusters: int = 3) -> Dict:
        """Run K-Means clustering."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < n_clusters:
            return {"success": False, "message": f"Need at least {n_clusters} rows"}

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)

        # Add cluster labels to dataframe
        result_df = self.dl.df.loc[data.index].copy()
        result_df['cluster'] = labels

        # Cluster centers (in original scale)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=columns)

        # Cluster sizes
        cluster_sizes = pd.Series(labels).value_counts().sort_index()

        return {
            "success": True,
            "labels": labels.tolist(),
            "cluster_sizes": cluster_sizes.to_dict(),
            "centers": centers_df.to_dict(orient='records'),
            "inertia": kmeans.inertia_,
            "silhouette": silhouette_score(data_scaled, labels, sample_size=min(2000, len(labels)), random_state=42) if 1 < len(set(labels)) < len(labels) else None,
            "result_df": result_df
        }

    def run_dbscan(self, columns: List[str] = None, eps: float = 0.5, min_samples: int = 5) -> Dict:
        """Run DBSCAN clustering."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < min_samples:
            return {"success": False, "message": f"Need at least {min_samples} rows"}

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        result_df = self.dl.df.loc[data.index].copy()
        result_df['cluster'] = labels

        return {
            "success": True,
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "result_df": result_df
        }

    def run_hierarchical(self, columns: List[str] = None, n_clusters: int = 3, linkage: str = 'ward') -> Dict:
        """Run Agglomerative Hierarchical Clustering."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < n_clusters:
            return {"success": False, "message": f"Need at least {n_clusters} rows"}

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = clustering.fit_predict(data_scaled)

        result_df = self.dl.df.loc[data.index].copy()
        result_df['cluster'] = labels

        return {
            "success": True,
            "labels": labels.tolist(),
            "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
            "result_df": result_df
        }

    def run_gmm(self, columns: List[str] = None, n_components: int = 3) -> Dict:
        """Run Gaussian Mixture Model clustering."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < n_components:
            return {"success": False, "message": f"Need at least {n_components} rows"}

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data_scaled)
        labels = gmm.predict(data_scaled)
        probs = gmm.predict_proba(data_scaled)

        result_df = self.dl.df.loc[data.index].copy()
        result_df['cluster'] = labels
        for i in range(n_components):
            result_df[f'cluster_{i}_prob'] = probs[:, i]

        return {
            "success": True,
            "labels": labels.tolist(),
            "probabilities": probs.tolist(),
            "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
            "means": scaler.inverse_transform(gmm.means_).tolist(),
            "covariances": gmm.covariances_.tolist(),
            "result_df": result_df
        }

    def compare_clustering(self, columns: List[str] = None) -> Dict:
        """Compare multiple clustering algorithms."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        # Find optimal k
        optimal = self.find_optimal_clusters(columns)
        if not optimal['success']:
            return optimal

        k = optimal['optimal_k']

        results = {}

        # K-Means
        km = self.run_kmeans(columns, k)
        if km['success']:
            results['kmeans'] = {
                "silhouette": km['silhouette'],
                "inertia": km['inertia'],
                "cluster_sizes": km['cluster_sizes']
            }

        # Hierarchical
        hc = self.run_hierarchical(columns, k)
        if hc['success']:
            labels = hc['labels']
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(self.dl.df[columns].dropna())
            results['hierarchical'] = {
                "silhouette": silhouette_score(data_scaled, labels) if len(set(labels)) > 1 else -1,
                "cluster_sizes": hc['cluster_sizes']
            }

        # GMM
        gmm = self.run_gmm(columns, k)
        if gmm['success']:
            labels = gmm['labels']
            data_scaled = StandardScaler().fit_transform(self.dl.df[columns].dropna())
            results['gmm'] = {
                "silhouette": silhouette_score(data_scaled, labels) if len(set(labels)) > 1 else -1,
                "cluster_sizes": gmm['cluster_sizes']
            }

        return {
            "success": True,
            "optimal_k": k,
            "comparison": results
        }

    # ==================== ANOMALY DETECTION ====================

    def detect_anomalies_isolation_forest(self, columns: List[str] = None, contamination: float = 0.1) -> Dict:
        """Detect anomalies using Isolation Forest."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < 10:
            return {"success": False, "message": "Need at least 10 rows"}

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        scores = iso_forest.fit_predict(data)
        anomaly_scores = iso_forest.decision_function(data)

        # -1 = anomaly, 1 = normal
        anomalies = scores == -1

        result_df = self.dl.df.loc[data.index].copy()
        result_df['is_anomaly'] = anomalies
        result_df['anomaly_score'] = anomaly_scores

        return {
            "success": True,
            "n_anomalies": int(anomalies.sum()),
            "anomaly_pct": float(anomalies.mean() * 100),
            "anomaly_indices": data.index[anomalies].tolist(),
            "result_df": result_df
        }

    def detect_anomalies_lof(self, columns: List[str] = None, n_neighbors: int = 20, contamination: float = 0.1) -> Dict:
        """Detect anomalies using Local Outlier Factor."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < n_neighbors + 1:
            return {"success": False, "message": f"Need at least {n_neighbors + 1} rows"}

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
        scores = lof.fit_predict(data)
        anomaly_scores = -lof.negative_outlier_factor_  # Higher = more anomalous

        anomalies = scores == -1

        result_df = self.dl.df.loc[data.index].copy()
        result_df['is_anomaly'] = anomalies
        result_df['anomaly_score'] = anomaly_scores

        return {
            "success": True,
            "n_anomalies": int(anomalies.sum()),
            "anomaly_pct": float(anomalies.mean() * 100),
            "anomaly_indices": data.index[anomalies].tolist(),
            "result_df": result_df
        }

    def detect_anomalies_one_class_svm(self, columns: List[str] = None, nu: float = 0.1, kernel: str = 'rbf') -> Dict:
        """Detect anomalies using One-Class SVM."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < 10:
            return {"success": False, "message": "Need at least 10 rows"}

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma='scale')
        scores = ocsvm.fit_predict(data_scaled)
        anomaly_scores = ocsvm.decision_function(data_scaled)

        anomalies = scores == -1

        result_df = self.dl.df.loc[data.index].copy()
        result_df['is_anomaly'] = anomalies
        result_df['anomaly_score'] = anomaly_scores

        return {
            "success": True,
            "n_anomalies": int(anomalies.sum()),
            "anomaly_pct": float(anomalies.mean() * 100),
            "anomaly_indices": data.index[anomalies].tolist(),
            "result_df": result_df
        }

    def detect_anomalies_zscore(self, columns: List[str] = None, threshold: float = 3.0) -> Dict:
        """Detect anomalies using Z-score method."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < 10:
            return {"success": False, "message": "Need at least 10 rows"}

        # Constant columns carry no anomaly signal.
        spread = data.std(ddof=0).replace(0, np.nan)
        z_scores = ((data - data.mean()) / spread).abs().fillna(0).to_numpy()
        anomalies = (z_scores > threshold).any(axis=1)

        result_df = self.dl.df.loc[data.index].copy()
        result_df['is_anomaly'] = anomalies
        result_df['max_zscore'] = z_scores.max(axis=1)

        return {
            "success": True,
            "n_anomalies": int(anomalies.sum()),
            "anomaly_pct": float(anomalies.mean() * 100),
            "anomaly_indices": data.index[anomalies].tolist(),
            "result_df": result_df
        }

    def detect_anomalies_iqr(self, columns: List[str] = None, multiplier: float = 1.5) -> Dict:
        """Detect anomalies using IQR method."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        data = self.dl.df[columns].dropna()
        if len(data) < 10:
            return {"success": False, "message": "Need at least 10 rows"}

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        anomalies = ((data < lower) | (data > upper)).any(axis=1)

        result_df = self.dl.df.loc[data.index].copy()
        result_df['is_anomaly'] = anomalies

        return {
            "success": True,
            "n_anomalies": int(anomalies.sum()),
            "anomaly_pct": float(anomalies.mean() * 100),
            "anomaly_indices": data.index[anomalies].tolist(),
            "bounds": {"lower": lower.to_dict(), "upper": upper.to_dict()},
            "result_df": result_df
        }

    def compare_anomaly_methods(self, columns: List[str] = None) -> Dict:
        """Compare multiple anomaly detection methods."""
        self._require_data()

        if columns is None:
            columns = self.dl.df.select_dtypes(include=np.number).columns.tolist()

        methods = {
            "isolation_forest": lambda: self.detect_anomalies_isolation_forest(columns),
            "lof": lambda: self.detect_anomalies_lof(columns),
            "zscore": lambda: self.detect_anomalies_zscore(columns),
            "iqr": lambda: self.detect_anomalies_iqr(columns),
        }

        results = {}
        for name, method in methods.items():
            try:
                result = method()
                if result['success']:
                    results[name] = {
                        "n_anomalies": result['n_anomalies'],
                        "anomaly_pct": result['anomaly_pct'],
                        "anomaly_indices": set(result['anomaly_indices'])
                    }
            except Exception as e:
                results[name] = {"error": str(e)}

        # Find consensus anomalies (detected by multiple methods)
        all_indices = set()
        for r in results.values():
            if 'anomaly_indices' in r:
                all_indices.update(r['anomaly_indices'])

        consensus = {}
        for idx in all_indices:
            count = sum(1 for r in results.values() if 'anomaly_indices' in r and idx in r['anomaly_indices'])
            if count > 1:
                consensus[idx] = count

        return {
            "success": True,
            "methods": results,
            "consensus_anomalies": consensus,
            "n_consensus": len(consensus)
        }

    # ==================== VISUALIZATION HELPERS ====================

    def plot_clusters(self, result_df: pd.DataFrame, columns: List[str], cluster_col: str = 'cluster') -> str:
        """Create cluster visualization."""
        import plotly.express as px

        if len(columns) >= 2:
            fig = px.scatter(
                result_df, x=columns[0], y=columns[1], color=cluster_col,
                title=f"Clusters: {columns[0]} vs {columns[1]}"
            )
        elif len(columns) == 1:
            fig = px.histogram(
                result_df, x=columns[0], color=cluster_col,
                title=f"Cluster Distribution: {columns[0]}"
            )
        else:
            return None

        return fig.to_json()

    def plot_anomalies(self, result_df: pd.DataFrame, columns: List[str], anomaly_col: str = 'is_anomaly') -> str:
        """Create anomaly visualization."""
        import plotly.express as px

        if len(columns) >= 2:
            fig = px.scatter(
                result_df, x=columns[0], y=columns[1], color=anomaly_col,
                title=f"Anomalies: {columns[0]} vs {columns[1]}",
                color_discrete_map={True: 'red', False: 'blue'}
            )
        elif len(columns) == 1:
            fig = px.histogram(
                result_df, x=columns[0], color=anomaly_col,
                title=f"Anomaly Distribution: {columns[0]}",
                color_discrete_map={True: 'red', False: 'blue'}
            )
        else:
            return None

        return fig.to_json()

    def plot_time_series(self, column: str, decomposition: Dict = None) -> str:
        """Create time series visualization."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not isinstance(self.dl.df.index, pd.DatetimeIndex):
            return None

        if decomposition:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])

            for i, comp in enumerate(['observed', 'trend', 'seasonal', 'residual']):
                fig.add_trace(
                    go.Scatter(x=self.dl.df.index, y=decomposition[comp], name=comp.capitalize()),
                    row=i+1, col=1
                )
            fig.update_layout(height=800, title=f"Time Series Decomposition: {column}")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.dl.df.index, y=self.dl.df[column], name=column))
            fig.update_layout(title=f"Time Series: {column}")

        return fig.to_json()
