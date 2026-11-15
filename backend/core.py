"""DataYab Core - Unified interface using modular backend components."""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from .modules import (
    DataLoader, DataCleaner, FeatureEngineer,
    ModelTrainer, ModelEvaluator, HyperparameterTuner,
    Predictor, Visualizer, InsightsGenerator,
    PipelineManager, ValidationGuard, AdvancedAnalytics
)


class DataYabCore:
    """Unified core interface using modular backend components."""

    def __init__(self):
        # Initialize all modules
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner(self.data_loader)
        self.feature_engineer = FeatureEngineer(self.data_loader)
        self.model_trainer = ModelTrainer(self.feature_engineer)
        self.model_evaluator = ModelEvaluator(self.model_trainer, self.feature_engineer)
        self.hyperparameter_tuner = HyperparameterTuner(self.model_trainer, self.feature_engineer)
        self.predictor = Predictor(self.model_trainer, self.feature_engineer)
        self.visualizer = Visualizer(self.data_loader, self.feature_engineer, self.model_trainer)
        self.insights_generator = InsightsGenerator(self.data_loader, self.feature_engineer, self.model_trainer)
        self.pipeline_manager = PipelineManager(
            self.data_loader, self.feature_engineer, self.model_trainer, self.model_evaluator
        )
        self.validation_guard = ValidationGuard(
            self.data_loader, self.feature_engineer, self.model_trainer
        )
        self.advanced_analytics = AdvancedAnalytics(self.data_loader)

        # Delegate attributes for backward compatibility
        self.df = None
        self.original_df = None
        self.file_path = ""
        self.log = []
        self.insights = []

        # Target and problem type (managed by feature_engineer)
        self.target_col = ""
        self.problem_type = ""

        # Model state (managed by model_trainer)
        self.model = None
        self.model_name = ""
        self.eval_model = None
        self.eval_model_name = None
        self.trained_models = {}
        self.available_models = []
        self.metrics = {}
        self.feature_importance = {}
        self.y_pred = None
        self.y_proba = None
        self.misclassified_indices = []
        self._model_schema = []
        self._used_feature_selection = False
        self._selected_feature_names = None
        self._selected_features_indices = None
        self._active_input_names = []

        # Label encoders (managed by data_cleaner)
        self.label_encoders = {}
        self.data_loader.on_dataset_changed = self._dataset_replaced

# ==================== DELEGATION PROPERTIES ====================

    @property
    def df(self):
        return self.data_loader.df

    @df.setter
    def df(self, value):
        self.data_loader.df = value

    @property
    def original_df(self):
        return self.data_loader.original_df

    @original_df.setter
    def original_df(self, value):
        self.data_loader.original_df = value

    @property
    def file_path(self):
        return self.data_loader.file_path

    @file_path.setter
    def file_path(self, value):
        self.data_loader.file_path = value

    @property
    def log(self):
        return self.data_loader.log

    @log.setter
    def log(self, value):
        self.data_loader.log = value

    @property
    def label_encoders(self):
        return self.data_cleaner.label_encoders

    @label_encoders.setter
    def label_encoders(self, value):
        self.data_cleaner.label_encoders = value

    @property
    def target_col(self):
        return self.feature_engineer.target_col

    @target_col.setter
    def target_col(self, value):
        self.feature_engineer.target_col = value
        self.data_cleaner.target_col = value

    @property
    def problem_type(self):
        return self.feature_engineer.problem_type

    @problem_type.setter
    def problem_type(self, value):
        self.feature_engineer.problem_type = value

    @property
    def model(self):
        return self.model_trainer.model

    @model.setter
    def model(self, value):
        self.model_trainer.model = value

    @property
    def model_name(self):
        return self.model_trainer.model_name

    @model_name.setter
    def model_name(self, value):
        self.model_trainer.model_name = value

    @property
    def eval_model(self):
        return self.model_trainer.eval_model

    @eval_model.setter
    def eval_model(self, value):
        self.model_trainer.eval_model = value

    @property
    def eval_model_name(self):
        return self.model_trainer.eval_model_name

    @eval_model_name.setter
    def eval_model_name(self, value):
        self.model_trainer.eval_model_name = value

    @property
    def trained_models(self):
        return self.model_trainer.trained_models

    @trained_models.setter
    def trained_models(self, value):
        self.model_trainer.trained_models = value

    @property
    def available_models(self):
        return self.model_trainer.available_models

    @available_models.setter
    def available_models(self, value):
        self.model_trainer.available_models = value

    @property
    def metrics(self):
        return self.model_trainer.metrics

    @metrics.setter
    def metrics(self, value):
        self.model_trainer.metrics = value

    @property
    def feature_importance(self):
        return self.model_trainer.feature_importance

    @feature_importance.setter
    def feature_importance(self, value):
        self.model_trainer.feature_importance = value

    @property
    def y_pred(self):
        return self.model_trainer.y_pred

    @y_pred.setter
    def y_pred(self, value):
        self.model_trainer.y_pred = value

    @property
    def y_proba(self):
        return self.model_trainer.y_proba

    @y_proba.setter
    def y_proba(self, value):
        self.model_trainer.y_proba = value

    @property
    def misclassified_indices(self):
        return self.model_trainer.misclassified_indices

    @misclassified_indices.setter
    def misclassified_indices(self, value):
        self.model_trainer.misclassified_indices = value

    @property
    def _model_schema(self):
        return self.model_trainer._model_schema

    @_model_schema.setter
    def _model_schema(self, value):
        self.model_trainer._model_schema = value

    @property
    def _active_input_names(self):
        return self.model_trainer._active_input_names

    @_active_input_names.setter
    def _active_input_names(self, value):
        self.model_trainer._active_input_names = value

    @property
    def feature_names(self):
        return self.feature_engineer.feature_names

    @feature_names.setter
    def feature_names(self, value):
        self.feature_engineer.feature_names = value

    @property
    def _used_feature_selection(self):
        return self.model_trainer._used_feature_selection

    @_used_feature_selection.setter
    def _used_feature_selection(self, value):
        self.model_trainer._used_feature_selection = value

    @property
    def _selected_feature_names(self):
        return self.model_trainer._selected_feature_names

    @_selected_feature_names.setter
    def _selected_feature_names(self, value):
        self.model_trainer._selected_feature_names = value

    @property
    def _selected_features_indices(self):
        return self.model_trainer._selected_features_indices

    @_selected_features_indices.setter
    def _selected_features_indices(self, value):
        self.model_trainer._selected_features_indices = value

    @property
    def target_encoder(self):
        return self.feature_engineer.target_encoder

    @target_encoder.setter
    def target_encoder(self, value):
        self.feature_engineer.target_encoder = value

    # Feature engineering delegations
    @property
    def _all_input_names(self):
        return self.feature_engineer._all_input_names

    @_all_input_names.setter
    def _all_input_names(self, value):
        self.feature_engineer._all_input_names = value

    @property
    def _selected_input_names(self):
        return self.feature_engineer._selected_input_names

    @_selected_input_names.setter
    def _selected_input_names(self, value):
        self.feature_engineer._selected_input_names = value

    @property
    def _split_signature(self):
        return self.feature_engineer._split_signature

    @_split_signature.setter
    def _split_signature(self, value):
        self.feature_engineer._split_signature = value

    @property
    def X_train(self):
        return self.feature_engineer.X_train

    @X_train.setter
    def X_train(self, value):
        self.feature_engineer.X_train = value

    @property
    def X_test(self):
        return self.feature_engineer.X_test

    @X_test.setter
    def X_test(self, value):
        self.feature_engineer.X_test = value

    @property
    def y_train(self):
        return self.feature_engineer.y_train

    @y_train.setter
    def y_train(self, value):
        self.feature_engineer.y_train = value

    @property
    def y_test(self):
        return self.feature_engineer.y_test

    @y_test.setter
    def y_test(self, value):
        self.feature_engineer.y_test = value

    @property
    def _raw_train(self):
        return self.feature_engineer._raw_train

    @_raw_train.setter
    def _raw_train(self, value):
        self.feature_engineer._raw_train = value

    @property
    def _raw_test(self):
        return self.feature_engineer._raw_test

    @_raw_test.setter
    def _raw_test(self, value):
        self.feature_engineer._raw_test = value

    @property
    def _numeric_columns(self):
        return self.feature_engineer._numeric_columns

    @_numeric_columns.setter
    def _numeric_columns(self, value):
        self.feature_engineer._numeric_columns = value

    @property
    def _extraction_method(self):
        return self.feature_engineer._extraction_method

    @_extraction_method.setter
    def _extraction_method(self, value):
        self.feature_engineer._extraction_method = value

    @property
    def _extraction_components(self):
        return self.feature_engineer._extraction_components

    @_extraction_components.setter
    def _extraction_components(self, value):
        self.feature_engineer._extraction_components = value

    @property
    def _selection_data_driven(self):
        return self.feature_engineer._selection_data_driven

    @_selection_data_driven.setter
    def _selection_data_driven(self, value):
        self.feature_engineer._selection_data_driven = value

    @property
    def _pca_components(self):
        return self.feature_engineer._pca_components

    @_pca_components.setter
    def _pca_components(self, value):
        self.feature_engineer._pca_components = value

    # ==================== DATA LOADING ====================

    def load_csv(self, file_path: str, sep: Optional[str] = None) -> Dict:
        return self.data_loader.load_csv(file_path, sep)

    def load_excel(self, file_path: str, sheet_name: Any = 0) -> Dict:
        return self.data_loader.load_excel(file_path, sheet_name)

    def load_json(self, file_path: str) -> Dict:
        return self.data_loader.load_json(file_path)

    def load_parquet(self, file_path: str) -> Dict:
        return self.data_loader.load_parquet(file_path)

    def load_arr(self, file_path: str) -> Dict:
        return self.data_loader.load_arr(file_path)

    def load_arff(self, file_path: str) -> Dict:
        return self.data_loader.load_arff(file_path)

    def load_file(self, file_path: str) -> Dict:
        return self.data_loader.load_file(file_path)

    def reset_to_original(self) -> Dict:
        return self.data_loader.reset_to_original()

    def get_info(self) -> Dict:
        return self.data_loader.get_info()

    def get_preview(self, rows: int = 10) -> Dict:
        return self.data_loader.get_preview(rows)

    def get_column_types(self) -> Dict:
        return self.data_loader.get_column_types()

    def export_cleaned_csv(self, file_path: str) -> Dict:
        return self.data_loader.export_cleaned_csv(file_path)

    def export_cleaned_arff(self, file_path: str, relation: str = "datayab_dataset") -> Dict:
        return self.data_loader.export_cleaned_arff(file_path, relation)

    def export_cleaned_data(self, file_path: str) -> Dict:
        return self.data_loader.export_cleaned_data(file_path)

    def get_log(self) -> List[str]:
        return self.data_loader.get_log()

    # ==================== DATA CLEANING ====================

    def get_missing_report(self) -> pd.DataFrame:
        return self.data_cleaner.get_missing_report()

    def handle_missing(self, strategy: str = "mean", columns: Optional[List[str]] = None) -> Dict:
        return self.data_cleaner.handle_missing(strategy, columns)

    def remove_duplicates(self) -> Dict:
        return self.data_cleaner.remove_duplicates()

    def encode_categorical(self, columns: Optional[List[str]] = None, method: str = "label") -> Dict:
        return self.data_cleaner.encode_categorical(columns, method)

    def remove_constant_columns(self) -> Dict:
        return self.data_cleaner.remove_constant_columns()

    def remove_high_cardinality(self, threshold: int = 50) -> Dict:
        return self.data_cleaner.remove_high_cardinality(threshold)

    def detect_outliers(self, column: str) -> Dict:
        return self.data_cleaner.detect_outliers(column)

    def cap_outliers(self, column: str) -> Dict:
        return self.data_cleaner.cap_outliers(column)

    def detect_all_outliers(self) -> pd.DataFrame:
        return self.data_cleaner.detect_all_outliers()

    def detect_outliers_isolation_forest(self, contamination: float = 0.1) -> Dict:
        return self.data_cleaner.detect_outliers_isolation_forest(contamination)

    def remove_outliers_isolation_forest(self, contamination: float = 0.1) -> Dict:
        return self.data_cleaner.remove_outliers_isolation_forest(contamination)

    def fix_dtypes(self) -> Dict:
        return self.data_cleaner.fix_dtypes()

    def auto_clean(self, target_col: Optional[str] = None) -> Dict:
        result = self.data_cleaner.auto_clean(target_col)
        if result.get("success") and target_col:
            self.target_col = target_col
        return result

    def get_dataset_quality(self) -> Dict:
        return self.data_cleaner.get_dataset_quality()

    # ==================== FEATURE ENGINEERING ====================

    def split_data(
        self,
        target: str,
        test_size: float = 0.2,
        scale: bool = True,
        scaler_type: str = "standard",
        problem_type: str = "auto"
    ) -> Dict:
        previous = self.feature_engineer._raw_train
        result = self.feature_engineer.split_data(target, test_size, scale, scaler_type, problem_type)
        if self.feature_engineer._raw_train is not previous:
            self.model_trainer._reset_learning()
        # Sync target_col and problem_type
        self.target_col = self.feature_engineer.target_col
        self.problem_type = self.feature_engineer.problem_type
        return result

    def apply_selection(self, method: str = "rf", k: int = 10, features: Optional[List[str]] = None) -> Dict:
        return self.feature_engineer.apply_selection(method, k, features)

    def select_features_manual(self, feature_list: List[str]) -> Dict:
        return self.feature_engineer.select_features_manual(feature_list)

    def restore_selection(self) -> Dict:
        return self.feature_engineer.restore_selection()

    def apply_extraction(self, method: str = "pca", n_components: int = 2) -> Dict:
        return self.feature_engineer.apply_extraction(method, n_components)

    def restore_extraction(self) -> Dict:
        return self.feature_engineer.restore_extraction()

    def apply_pca(self, n_components: int = 2) -> Dict:
        return self.feature_engineer.apply_extraction("pca", n_components)

    def restore_from_pca(self) -> Dict:
        return self.feature_engineer.restore_extraction()

    def get_feature_status(self) -> Dict:
        return self.feature_engineer.get_feature_status()

    def get_feature_scores(self, method: str = "rf", top_n: int = 20):
        return self.feature_engineer.get_feature_scores(method, top_n)

    def get_fast_feature_importance(self, top_n: int = 20):
        return self.feature_engineer.get_fast_feature_importance(top_n)

    def get_mutual_information_top(self, k: int = 10):
        return self.feature_engineer.get_mutual_information_top(k)

    def get_permutation_importance(self, top_n: int = 15, n_repeats: int = 5):
        self.feature_engineer.model = self.model
        return self.feature_engineer.get_permutation_importance(top_n, n_repeats)

    # ==================== MODEL TRAINING ====================

    def train(
        self, model_name: str = "Random Forest", use_smote: bool = False,
        class_weight: Optional[str] = None, **kwargs
    ) -> Dict:
        return self.model_trainer.train(model_name, use_smote, class_weight, **kwargs)

    def get_trained_models(self) -> List[str]:
        return self.model_trainer.get_trained_models()

    def select_model(self, model_name: str) -> Dict:
        return self.model_trainer.select_model(model_name)

    def set_evaluation_model(self, model_name: str) -> Dict:
        return self.model_trainer.set_evaluation_model(model_name)

    def get_evaluation_model(self) -> str:
        return self.model_trainer.get_evaluation_model()

    def train_all_models(self, use_smote: bool = False, class_weight: Optional[str] = None, **kwargs) -> pd.DataFrame:
        return self.model_trainer.train_all_models(use_smote, class_weight, **kwargs)

    def train_stacked_model(
        self, base_model_1: str, base_model_2: str, meta_model: str = "Logistic Regression"
    ) -> Dict:
        return self.model_trainer.train_stacked_model(base_model_1, base_model_2, meta_model)

    def get_cross_val_score(self, cv: int = 5) -> Dict:
        return self.model_trainer.get_cross_val_score(cv)

    # ==================== MODEL EVALUATION ====================

    def get_metrics(self) -> Dict:
        return self.model_evaluator.get_metrics()

    def get_confusion_matrix(self) -> Dict:
        return self.model_evaluator.get_confusion_matrix()

    def get_misclassified_samples(self, max_samples: int = 20) -> pd.DataFrame:
        return self.model_evaluator.get_misclassified_samples(max_samples)

    # ==================== HYPERPARAMETER TUNING ====================

    def tune_with_optuna(
        self,
        model_name: str,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        cv_folds: int = 3,
        use_smote: bool = False,
        class_weight: Optional[str] = None,
    ) -> Dict:
        return self.hyperparameter_tuner.tune_with_optuna(
            model_name, n_trials, timeout, cv_folds, use_smote, class_weight
        )

    def tune_with_gridsearch(
        self,
        model_name: str,
        param_grid: Dict,
        cv_folds: int = 3,
        use_smote: bool = False,
        class_weight: Optional[str] = None,
    ) -> Dict:
        return self.hyperparameter_tuner.tune_with_gridsearch(
            model_name, param_grid, cv_folds, use_smote, class_weight
        )

    def get_model_recommendation(self, target: str = None) -> Dict:
        return self.hyperparameter_tuner.get_model_recommendation(target)

    # ==================== PREDICTION ====================

    def get_input_form_schema(self) -> List[Dict]:
        return self.predictor.get_input_form_schema()

    def predict(self, input_dict: Dict[str, Any]) -> Dict:
        return self.predictor.predict(input_dict)

    def predict_from_csv(self, file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        return self.predictor.predict_from_csv(file_path, output_path)

    def predict_on_test_set(self, test_indices: Optional[List[int]] = None) -> pd.DataFrame:
        return self.predictor.predict_on_test_set(test_indices)

    # ==================== VISUALIZATION ====================

    def plot_histogram(self, column: str, bins: int = 30) -> str:
        return self.visualizer.plot_histogram(column, bins)

    def plot_boxplot(self, column: str) -> str:
        return self.visualizer.plot_boxplot(column)

    def plot_correlation_heatmap(self, max_cols: int = 20) -> str:
        return self.visualizer.plot_correlation_heatmap(max_cols)

    def plot_target_distribution(self, target: str = None) -> str:
        return self.visualizer.plot_target_distribution(target)

    def plot_scatter(self, x_col: str, y_col: str, hue_col: str = None) -> str:
        return self.visualizer.plot_scatter(x_col, y_col, hue_col)

    def plot_missing_heatmap(self) -> str:
        return self.visualizer.plot_missing_heatmap()

    def plot_feature_importance(self) -> str:
        return self.visualizer.plot_feature_importance()

    def plot_pairplot(self, columns: List[str] = None, hue: str = None) -> str:
        return self.visualizer.plot_pairplot(columns, hue)

    def plot_confusion_matrix(self) -> str:
        return self.visualizer.plot_confusion_matrix()

    def plot_roc_curve(self) -> str:
        return self.visualizer.plot_roc_curve()

    def plot_precision_recall_curve(self) -> str:
        return self.visualizer.plot_precision_recall_curve()

    def plot_residuals(self) -> str:
        return self.visualizer.plot_residuals()

    def plot_learning_curve(self, train_sizes: np.ndarray = None) -> str:
        return self.visualizer.plot_learning_curve(train_sizes)

    def plot_validation_curve(self, param_name: str, param_range: np.ndarray) -> str:
        return self.visualizer.plot_validation_curve(param_name, param_range)

    def plot_partial_dependence(self, features: List[str], kind: str = "average") -> str:
        return self.visualizer.plot_partial_dependence(features, kind)

    def get_stats(self) -> pd.DataFrame:
        return self.visualizer.get_stats()

    def get_correlation_matrix(self) -> pd.DataFrame:
        return self.visualizer.get_correlation_matrix()

    def get_top_correlations(self, target: str = None, n: int = 15) -> pd.DataFrame:
        return self.visualizer.get_top_correlations(target, n)

    def get_value_counts(self, column: str, top_n: int = 20) -> pd.DataFrame:
        return self.visualizer.get_value_counts(column, top_n)

    # ==================== INSIGHTS ====================

    def generate_observations(self, n: int = 10) -> List[str]:
        return self.insights_generator.generate_observations(n)

    def get_dataset_health_score(self) -> Dict:
        return self.insights_generator.get_dataset_health_score()

    def generate_insights(self) -> List[str]:
        return self.insights_generator.generate_insights()

    def generate_model_card(self) -> Dict:
        return self.insights_generator.generate_model_card()

    # ==================== PIPELINE MANAGEMENT ====================

    def save_pipeline(self, file_path: str) -> Dict:
        return self.pipeline_manager.save_pipeline(file_path)

    def load_pipeline(self, file_path: str) -> Dict:
        return self.pipeline_manager.load_pipeline(file_path)

    def export_report(self, file_path: str) -> Dict:
        return self.pipeline_manager.export_report(file_path)

    def get_pipeline_summary(self) -> Dict:
        return self.pipeline_manager.get_pipeline_summary()

    # ==================== VALIDATION ====================

    def validate_pipeline(self) -> Dict:
        return self.validation_guard.validate_pipeline()

    def validate_before_deployment(self) -> Dict:
        return self.validation_guard.validate_before_deployment()

    def get_data_quality_report(self) -> Dict:
        return self.validation_guard.get_data_quality_report()

    # ==================== PRIVATE HELPERS ====================

    def _require_data(self):
        if self.data_loader.df is None:
            raise ValueError("No dataset loaded. Call load_csv() or load_file() first.")

    def _require_model(self):
        if self.model_trainer.model is None:
            raise ValueError("Train or import a model first.")

    def _log(self, message: str):
        self.data_loader._log(message)

    def _reset_learning(self):
        self.model_trainer._reset_learning()
        # Keep the shared FeatureEngineer object; every module references it.
        self.feature_engineer.__init__(self.data_loader)

    def _dataset_replaced(self):
        self._reset_learning()
        self.target_col = ""
        self.label_encoders = {}
        self.insights = []

    def _set_dataset(self, df, path=""):
        return self.data_loader._set_dataset(df, path)


# Decorator for dataset editing operations
from functools import wraps

def _dataset_edit(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._require_data()
        if getattr(self, "_editing_dataset", False):
            return method(self, *args, **kwargs)
        original = self.data_loader.df.copy(deep=True) if self.data_loader.df is not None else None
        original_target = self.target_col
        original_encoders = dict(self.label_encoders)
        original_log = list(self.log)
        self._editing_dataset = True
        try:
            result = method(self, *args, **kwargs)
            if isinstance(result, dict) and result.get("success") is False:
                if original is not None:
                    self.data_loader.df = original
                self.target_col = original_target
                self.label_encoders = original_encoders
                self.log = original_log
                return result
            if self.data_loader.df is not None and self.data_loader.df.empty:
                raise ValueError(
                    "This operation would leave an empty dataset. No changes applied."
                )
        except Exception:
            if original is not None:
                self.data_loader.df = original
            self.target_col = original_target
            self.label_encoders = original_encoders
            self.log = original_log
            raise
        finally:
            self._editing_dataset = False
        target = self.target_col
        self._reset_learning()
        self.target_col = target if target in (self.data_loader.df.columns if self.data_loader.df is not None else []) else ""
        self.insights = []
        return result
    return guarded


for _name in [
    "handle_missing",
    "remove_duplicates",
    "encode_categorical",
    "remove_constant_columns",
    "remove_high_cardinality",
    "cap_outliers",
    "remove_outliers_isolation_forest",
    "fix_dtypes",
    "auto_clean",
]:
    if hasattr(DataYabCore, _name):
        setattr(DataYabCore, _name, _dataset_edit(getattr(DataYabCore, _name)))


def _feature_edit(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        # Failed extraction/selection must leave a working model usable.
        previous = self.feature_engineer.__dict__.copy()
        try:
            result = method(self, *args, **kwargs)
        except Exception:
            self.feature_engineer.__dict__.clear()
            self.feature_engineer.__dict__.update(previous)
            raise
        self.model_trainer._reset_learning()
        return result
    return guarded


for _name in ("apply_selection", "select_features_manual", "restore_selection",
              "apply_extraction", "restore_extraction", "apply_pca", "restore_from_pca"):
    setattr(DataYabCore, _name, _feature_edit(getattr(DataYabCore, _name)))
