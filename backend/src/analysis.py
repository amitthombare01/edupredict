"""Core analysis utilities for student performance."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from pandas.errors import EmptyDataError, ParserError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENT_DATA_PATH = os.path.join(DATA_DIR, "student_performance.csv")
MODEL_DIR = os.path.join(DATA_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "student_model.joblib")
MODEL_META_FILE = os.path.join(MODEL_DIR, "model_meta.json")

TARGET_PRIORITIES = [
    "pass",
    "final_exam_score",
    "final_score",
    "grade",
    "score",
    "result",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_column_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _find_target_column(columns: List[str]) -> Optional[str]:
    normalized = {normalize_column_name(col): col for col in columns}
    for candidate in TARGET_PRIORITIES:
        normalized_candidate = normalize_column_name(candidate)
        if normalized_candidate in normalized:
            return normalized[normalized_candidate]

    numeric_columns = [col for col in columns if col.lower().startswith("final")]
    if numeric_columns:
        return numeric_columns[-1]

    return None


def load_student_data(path: str = STUDENT_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (EmptyDataError, ParserError, FileNotFoundError):
        return pd.DataFrame()


def save_student_data(df: pd.DataFrame, path: str = STUDENT_DATA_PATH) -> None:
    _ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def _ensure_model_dir() -> None:
    _ensure_dir(MODEL_DIR)


def _load_pipeline() -> Tuple[Optional[Pipeline], Dict[str, Any]]:
    if not os.path.exists(MODEL_FILE) or not os.path.exists(MODEL_META_FILE):
        return None, {}

    try:
        pipeline = load(MODEL_FILE)
        with open(MODEL_META_FILE, "r") as f:
            meta = json.load(f)
        return pipeline, meta
    except (IOError, ValueError):
        return None, {}


def _save_pipeline(pipeline: Pipeline, metadata: Dict[str, Any]) -> None:
    _ensure_model_dir()
    dump(pipeline, MODEL_FILE)
    with open(MODEL_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def _prepare_training_data(
    df: pd.DataFrame, target: str, features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    subset = df[features + [target]].copy()
    subset = subset.dropna()
    return subset[features], subset[target]


def train_model(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"status": "no_data"}

    target = _find_target_column(df.columns.tolist())
    if not target or target not in df.columns:
        return {"status": "no_target"}

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col != target]
    if not feature_columns:
        return {"status": "no_features"}

    X, y = _prepare_training_data(df, target, feature_columns)
    if X.empty or y.empty:
        return {"status": "insufficient_data"}

    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[finite_mask]
    y = y[finite_mask]
    if X.shape[0] < 5:
        return {"status": "too_few_rows"}

    model_type = "logistic" if set(y.unique()).issubset({0, 1}) else "regression"
    if model_type == "logistic":
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("estimator", LogisticRegression(max_iter=1000))]
        )
    else:
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("estimator", LinearRegression())]
        )

    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    metadata = {
        "target": target,
        "model_type": model_type,
        "features": feature_columns,
        "trained_on": datetime.utcnow().isoformat() + "Z",
        "score": float(score),
    }
    _save_pipeline(pipeline, metadata)
    return {"status": "trained", **metadata}


def apply_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pipeline, metadata = _load_pipeline()
    if not pipeline or not metadata.get("features"):
        return df, {"status": "no_model"}

    features = metadata["features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        return df, {"status": "features_missing", "missing": missing}

    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.mean())

    new_df = df.copy()
    if metadata["model_type"] == "logistic":
        proba = pipeline.predict_proba(X)[:, 1]
        new_df["predicted_pass_probability"] = proba
        new_df["predicted_pass"] = pipeline.predict(X).astype(int)
    else:
        predictions = pipeline.predict(X)
        new_df[f"predicted_{metadata['target']}"] = predictions

    return new_df, {"status": "scored", **metadata}


def summarize_data(df: pd.DataFrame) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"total_records": len(df)}
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    stats["numeric_summary"] = []
    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            continue
        stats["numeric_summary"].append(
            {
                "column": column,
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        )

    pass_column = next(
        (
            col
            for col in df.columns
            if normalize_column_name(col) == "pass"
            or normalize_column_name(col) == normalize_column_name("predicted_pass")
        ),
        None,
    )
    if pass_column and pass_column in df.columns:
        stats["pass_rate"] = float(df[pass_column].mean())

    if "predicted_pass_probability" in df.columns:
        stats["predicted_pass_probability"] = float(
            df["predicted_pass_probability"].mean()
        )

    return stats


def predict_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    pipeline, metadata = _load_pipeline()
    if not pipeline or not metadata:
        raise ValueError("No trained model available")

    features = metadata["features"]
    missing = [feat for feat in features if feat not in payload]
    if missing:
        raise ValueError(f"Missing features for prediction: {', '.join(missing)}")

    X = pd.DataFrame([{feat: payload[feat] for feat in features}])
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("All feature values must be numeric")

    result: Dict[str, Any] = {"target": metadata["target"], "model_type": metadata["model_type"]}
    if metadata["model_type"] == "logistic":
        probability = pipeline.predict_proba(X)[:, 1][0]
        prediction = int(pipeline.predict(X)[0])
        result["predicted_pass_probability"] = float(probability)
        result["predicted_pass"] = prediction
    else:
        value = pipeline.predict(X)[0]
        result[f"predicted_{metadata['target']}"] = float(value)

    return result


def get_model_metadata() -> Dict[str, Any]:
    """Return metadata about the last trained model."""
    _, metadata = _load_pipeline()
    return metadata
