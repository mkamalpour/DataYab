"""Prediction module for single and batch predictions."""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


class Predictor:
    """Handles single and batch predictions."""

    def __init__(self, model_trainer, feature_engineer):
        self.mt = model_trainer
        self.fe = feature_engineer

    def _require_model(self):
        if self.mt.model is None:
            raise ValueError("Train or import a model first.")

    def get_input_form_schema(self) -> List[Dict]:
        self._require_model()
        return copy.deepcopy(self.mt._model_schema)

    def predict(self, input_dict: Dict[str, Any]) -> Dict:
        self._require_model()

        # Check for missing fields
        missing = [
            item["name"] for item in self.mt._model_schema
            if item["name"] not in input_dict
        ]
        if missing:
            raise ValueError("Missing prediction fields: " + ", ".join(missing))

        row = {}
        for item in self.mt._model_schema:
            name = item["name"]
            value = input_dict[name]
            if item["type"] == "numeric":
                try:
                    value = np.nan if pd.isna(value) or value == "" else float(value)
                except (ValueError, TypeError):
                    raise ValueError(f"{name}: enter a number or leave blank.")
                if np.isinf(value):
                    raise ValueError(f"{name}: infinity is not supported.")
            else:
                value = np.nan if pd.isna(value) or value == "" else str(value)
            row[name] = value

        X = pd.DataFrame([row])
        raw = self.mt.model.predict(X)[0]

        label = (
            self.fe.target_encoder.inverse_transform([int(raw)])[0]
            if self.fe.target_encoder is not None
            else raw
        )

        result = {
            "prediction": label.item() if isinstance(label, np.generic) else label,
            "raw": raw.item() if isinstance(raw, np.generic) else raw,
        }

        if hasattr(self.mt.model, "predict_proba"):
            proba = self.mt.model.predict_proba(X)[0]
            labels = self.fe.target_encoder.inverse_transform(
                self.mt.model.classes_.astype(int)
            )
            result["confidence"] = float(proba.max())
            result["probabilities"] = {str(c): float(p) for c, p in zip(labels, proba)}

        return result

    def predict_from_csv(self, file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        self._require_model()
        frame = pd.read_csv(file_path)
        rows = []

        missing = {item["name"] for item in self.mt._model_schema} - set(frame.columns)
        if missing:
            raise ValueError(
                "Batch file is missing columns: " + ", ".join(sorted(missing))
            )

        for idx, row in frame.iterrows():
            try:
                result = self.predict(row.to_dict())
                rows.append({
                    "sample_id": idx,
                    "prediction": result["prediction"],
                    "confidence": result.get("confidence"),
                })
            except ValueError as error:
                rows.append({
                    "sample_id": idx,
                    "prediction": None,
                    "confidence": None,
                    "error": str(error),
                })

        result_df = pd.DataFrame(rows)

        if output_path:
            from ..storage import atomic_target
            with atomic_target(output_path) as temporary:
                result_df.to_csv(temporary, index=False)

        return result_df

    def predict_on_test_set(self, test_indices: Optional[List[int]] = None) -> pd.DataFrame:
        """Get predictions on the holdout test set with actual vs predicted comparison."""
        self._require_model()
        if self.fe._raw_test is None:
            raise ValueError("No test set available.")

        self.mt._compute_predictions()

        if test_indices is None:
            test_indices = list(range(len(self.fe.y_test)))

        rows = []
        for i in test_indices:
            if i >= len(self.fe.y_test):
                continue

            # Get original row data
            orig_idx = self.fe._test_indices[i]
            row_data = self.fe.data_loader.df.loc[orig_idx].to_dict()

            # Get actual and predicted
            if self.fe.problem_type == "classification":
                actual = self.fe.target_encoder.inverse_transform([self.fe.y_test[i]])[0]
                predicted = self.fe.target_encoder.inverse_transform([self.mt.y_pred[i]])[0]
                confidence = float(self.mt.y_proba[i].max()) if self.mt.y_proba is not None else None
                correct = actual == predicted
            else:
                actual = self.fe.y_test[i]
                predicted = self.mt.y_pred[i]
                confidence = None
                correct = None

            rows.append({
                "index": i,
                "original_index": orig_idx,
                "actual": actual,
                "predicted": predicted,
                "confidence": confidence,
                "correct": correct,
                **row_data
            })

        return pd.DataFrame(rows)


# Need to import copy
import copy
