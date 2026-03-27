from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
import os
import re
import io
import math
import warnings
import csv

import hashlib
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from pymongo import MongoClient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.pkl"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip()
MONGODB_DB = os.getenv("MONGODB_DB", "student_ai").strip() or "student_ai"

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]

students_col = db["students"]
history_col = db["history"]
users_col = db["users"]
attendance_events_col = db["attendance_events"]
attendance_sessions_col = db["attendance_sessions"]
attendance_settings_col = db["attendance_settings"]
rfid_mappings_col = db["rfid_mappings"]

# ---------------- API APP ----------------

api_app = FastAPI()

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROOT APP ----------------

app = FastAPI()

MISSING_TEXT_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "nil",
    "-",
    "--",
    "?",
    "unknown",
}
TARGET_HINTS = {
    "result",
    "remark",
    "status",
    "grade",
    "pass",
    "passed",
    "outcome",
    "label",
    "target",
    "finalresult",
    "performance",
}
BACKLOG_HINTS = {
    "atkt",
    "backlog",
    "arrear",
    "arrears",
    "kt",
    "failcount",
    "failedsubject",
    "failedsubjects",
    "supplementary",
    "reappear",
}
SCORE_HINTS = {
    "score",
    "marks",
    "mark",
    "total",
    "overall",
    "aggregate",
    "percentage",
    "percent",
    "cgpa",
    "sgpa",
    "gpa",
    "finalscore",
    "finalmarks",
    "totalscore",
    "obtainedmarks",
}
ATTENDANCE_HINTS = {
    "attendance",
    "attendancepercent",
    "attendancepct",
    "present",
    "presence",
    "absent",
    "lecture",
    "lectures",
    "class",
    "classes",
    "workingday",
    "workingdays",
    "day",
    "days",
    "conducted",
    "held",
}
ATTENDANCE_CONTEXT_HINTS = {
    "attendance",
    "present",
    "absent",
    "lecture",
    "lectures",
    "class",
    "classes",
    "workingday",
    "workingdays",
    "day",
    "days",
    "month",
    "monthly",
    "conducted",
    "held",
    "session",
    "sessions",
}
NON_PERFORMANCE_HINTS = {
    "income",
    "salary",
    "job",
    "occupation",
    "profession",
    "age",
    "gender",
    "social",
    "socialmedia",
    "internet",
    "device",
    "sports",
    "hobby",
    "family",
    "studyhours",
    "sleephours",
    "extraclasses",
}
IDENTIFIER_HINTS = {
    "id",
    "studentid",
    "roll",
    "rollno",
    "registration",
    "email",
    "phone",
    "mobile",
    "name",
}
RFID_HINTS = {
    "rfid",
    "uid",
    "card",
    "cardid",
    "carduid",
    "tag",
    "tagid",
    "nfctag",
    "nfcid",
}
BOOLEAN_LABELS = {
    "pass": 1,
    "passed": 1,
    "passing": 1,
    "yes": 1,
    "y": 1,
    "true": 1,
    "1": 1,
    "success": 1,
    "successful": 1,
    "clear": 1,
    "cleared": 1,
    "eligible": 1,
    "promoted": 1,
    "promotednextclass": 1,
    "qualified": 1,
    "completed": 1,
    "complete": 1,
    "firstclass": 1,
    "secondclass": 1,
    "distinction": 1,
    "satisfactory": 1,
    "goodstanding": 1,
    "present": 1,
    "fail": 0,
    "failed": 0,
    "failing": 0,
    "no": 0,
    "n": 0,
    "false": 0,
    "0": 0,
    "unsuccessful": 0,
    "notqualified": 0,
    "noteligible": 0,
    "notpromoted": 0,
    "rejected": 0,
    "reappear": 0,
    "supplementary": 0,
    "withheld": 0,
    "withheldresult": 0,
    "malpractice": 0,
    "debarred": 0,
    "dropout": 0,
    "drop": 0,
    "incomplete": 0,
    "notcleared": 0,
    "unsuccess": 0,
    "detain": 0,
    "detained": 0,
    "atkt": 0,
    "atktin": 0,
    "backlog": 0,
    "arrear": 0,
    "arrears": 0,
    "kt": 0,
    "notsuccessful": 0,
    "absent": 0,
}

PDF_TABLE_SEPARATORS = ("|", ",", ";", "\t")
CANONICAL_COLUMN_HINTS = {
    "student_id": {"roll", "rollno", "rollnumber", "studentid", "seatno", "seatnumber", "srno"},
    "student_name": {"name", "nameofstudent", "studentname", "nameofstudents"},
    "attendance": {
        "attendance",
        "attendancepercent",
        "attendancepct",
        "presentpercentage",
        "presentdays",
        "dayspresent",
        "daysabsent",
        "absentdays",
        "lectures",
        "lecture",
        "totalnooflectures",
        "workingdays",
        "classheld",
    },
    "total_score": {"total", "totalscore", "overall", "overallscore", "aggregate", "aggregatescore", "grandtotal"},
    "percentage": {"percentage", "percent", "finalpercentage", "overallpercentage"},
    "result": {"result", "remark", "remarks", "status", "finalresult"},
}
NON_STUDENT_ROW_HINTS = {
    "nameofstudent",
    "studentname",
    "maximummarks",
    "maximum",
    "subjectcode",
    "subjectname",
    "rollno",
    "rollnumber",
    "remark",
    "remarks",
    "result",
    "total",
    "percentage",
    "percent",
}
DATASET_ROLE_PERFORMANCE = "performance"
DATASET_ROLE_ATTENDANCE = "attendance"
DATASET_ROLE_UNSUPPORTED = "unsupported"


def normalize_token(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


NORMALIZED_BOOLEAN_LABELS = {
    normalize_token(label): value for label, value in BOOLEAN_LABELS.items()
}


def token_matches(token, hints):
    return any(hint in token for hint in hints)


def is_attendance_related_token(token):
    return token_matches(token, ATTENDANCE_CONTEXT_HINTS)


def is_rfid_related_token(token):
    return token_matches(token, RFID_HINTS)


def make_unique_columns(columns):
    seen = {}
    cleaned = []

    for idx, col in enumerate(columns, start=1):
        label = str(col).strip()
        if not label or label.lower().startswith("unnamed"):
            label = f"column_{idx}"

        label = re.sub(r"\s+", "_", label)
        label = re.sub(r"[^A-Za-z0-9_%-]", "", label)
        label = label.strip("_") or f"column_{idx}"

        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}_{seen[label]}"

        cleaned.append(label)

    return cleaned


def canonicalize_column_name(column_name):
    token = normalize_token(column_name)
    for canonical_name, hints in CANONICAL_COLUMN_HINTS.items():
        if token == canonical_name:
            return canonical_name
        if any(hint in token for hint in hints):
            return canonical_name
    return column_name


def canonicalize_dataframe_columns(df):
    if df.empty:
        return df

    renamed = []
    for col in df.columns:
        renamed.append(canonicalize_column_name(col))

    df = df.copy()
    df.columns = make_unique_columns(renamed)
    return df


def normalize_cell(value):
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in MISSING_TEXT_VALUES:
            return pd.NA
    return value


def row_numeric_ratio(row):
    values = [str(cell).strip() for cell in row if str(cell).strip()]
    if not values:
        return 0.0
    numeric_like = 0
    for value in values:
        cleaned = re.sub(r"[%,/()-]", "", value)
        if re.fullmatch(r"\d+(\.\d+)?", cleaned):
            numeric_like += 1
    return numeric_like / len(values)


def merge_header_rows(rows):
    if len(rows) < 2:
        return rows

    first_row = [str(cell).strip() for cell in rows[0]]
    second_row = [str(cell).strip() for cell in rows[1]]
    if len(first_row) != len(second_row):
        return rows

    if row_numeric_ratio(first_row) > 0.35:
        return rows

    # PDF marksheets often split headers across two lines like
    # "H-AI Applicatio" + "n/35". Merge them into a single stable header.
    if row_numeric_ratio(second_row) <= 0.6:
        merged = []
        for first, second in zip(first_row, second_row):
            parts = [part for part in [first, second] if part]
            merged.append(" ".join(parts).strip())
        return [merged] + rows[2:]

    return rows


def promote_embedded_pdf_header(df):
    if df is None or df.empty:
        return df

    candidate_limit = min(len(df), 6)
    for idx in range(candidate_limit):
        row = df.iloc[idx]
        values = [str(value).strip() for value in row.tolist()]
        normalized_values = [normalize_token(value) for value in values if value]
        if not normalized_values:
            continue

        has_student_name = any("nameofstudent" in value or "studentname" in value for value in normalized_values)
        has_roll = any("rollno" in value or value == "roll" or value == "no" for value in normalized_values)
        subject_like = sum(
            1 for value in normalized_values
            if any(hint in value for hint in ("design", "system", "communication", "application", "mobile", "embedded", "vlsi", "electronic", "ai"))
        )
        has_result = any("remark" in value or "result" in value for value in normalized_values)

        if not has_student_name:
            continue
        if not (has_roll or subject_like >= 2 or has_result):
            continue

        promoted = df.copy()
        promoted.columns = make_unique_columns(values)
        promoted = promoted.iloc[idx + 1:].reset_index(drop=True)
        return promoted

    return df


def coerce_boolean_series(series):
    non_null = series.dropna()
    if non_null.empty:
        return None

    normalized = non_null.astype(str).map(normalize_token)
    unique_values = set(normalized.unique())
    if unique_values and unique_values.issubset(NORMALIZED_BOOLEAN_LABELS):
        return series.map(
            lambda value: NORMALIZED_BOOLEAN_LABELS.get(normalize_token(value), pd.NA)
            if pd.notna(value)
            else pd.NA
        )

    pattern_supported = normalized.map(
        lambda value: bool(re.fullmatch(r"f\d+", value))
        or bool(re.fullmatch(r"pass\d*", value))
        or bool(re.fullmatch(r"p\d+", value))
    )
    if pattern_supported.mean() >= 0.7:
        return series.map(
            lambda value: (
                0 if re.fullmatch(r"f\d+", normalize_token(value)) else
                (1 if re.fullmatch(r"(pass\d*|p\d+)", normalize_token(value)) else pd.NA)
            ) if pd.notna(value) else pd.NA
        )

    return None


def is_likely_outcome_column(series):
    non_null = series.dropna()
    if non_null.empty:
        return False

    boolean_series = coerce_boolean_series(series)
    if boolean_series is not None:
        return True

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return False
        unique_count = numeric.nunique()
        if unique_count <= 3:
            return True

        sorted_values = sorted(numeric.unique().tolist())
        if len(sorted_values) >= 4 and all(
            abs(sorted_values[idx + 1] - sorted_values[idx] - 1) < 1e-9
            for idx in range(len(sorted_values) - 1)
        ):
            return False

        return unique_count <= min(10, max(4, len(numeric) // 3))

    text = non_null.astype(str).str.strip()
    unique_count = text.nunique()
    unique_ratio = unique_count / max(len(text), 1)
    return 1 < unique_count <= 12 and unique_ratio < 0.8


def coerce_numeric_series(series, min_ratio=0.6):
    non_null = series.dropna()
    if non_null.empty:
        return None

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[%,$]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    cleaned = cleaned.where(series.notna(), pd.NA)
    numeric = pd.to_numeric(cleaned, errors="coerce")

    success_ratio = numeric.notna().sum() / max(non_null.shape[0], 1)
    if success_ratio >= min_ratio:
        return numeric

    return None


def coerce_datetime_series(series, min_ratio=0.75):
    non_null = series.dropna()
    if non_null.empty:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        converted = pd.to_datetime(series, errors="coerce")
    success_ratio = converted.notna().sum() / max(non_null.shape[0], 1)
    if success_ratio >= min_ratio:
        return converted.map(lambda value: value.toordinal() if pd.notna(value) else pd.NA)

    return None


def sanitize_value(value):
    if isinstance(value, dict):
        return {key: sanitize_value(val) for key, val in value.items()}

    if isinstance(value, list):
        return [sanitize_value(item) for item in value]

    if value is pd.NA:
        return None

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass

    return value


def sanitize_records(records):
    return [sanitize_value(record) for record in records]


def clean_dataframe(df):
    df = df.copy()
    df.columns = make_unique_columns(df.columns)
    df = canonicalize_dataframe_columns(df)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.apply(lambda col: col.map(normalize_cell))
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue

        boolean_series = coerce_boolean_series(series)
        if boolean_series is not None:
            df[col] = boolean_series
            continue

        numeric_series = coerce_numeric_series(series)
        if numeric_series is not None:
            df[col] = numeric_series
            continue

        datetime_series = coerce_datetime_series(series)
        if datetime_series is not None:
            df[col] = datetime_series
            continue

        df[col] = series.map(lambda value: value.strip() if isinstance(value, str) else value)

    return df


def choose_performance_score_column(df):
    numeric = df.select_dtypes(include=["number"])
    candidates = []

    for col in numeric.columns:
        token = normalize_token(col)
        if token in IDENTIFIER_HINTS or is_attendance_related_token(token) or token_matches(token, NON_PERFORMANCE_HINTS):
            continue

        series = pd.to_numeric(numeric[col], errors="coerce").dropna()
        if series.empty or series.nunique() <= 1:
            continue

        alias_bonus = 5 if token_matches(token, SCORE_HINTS) else 0
        aggregate_bonus = 4 if any(marker in token for marker in {"total", "overall", "aggregate", "percentage", "percent"}) else 0
        range_bonus = 2 if series.between(0, 100).mean() >= 0.8 else 0
        total_range_bonus = 3 if series.max() > 100 else 0
        variance_bonus = 1 if series.nunique() >= 5 else 0
        candidates.append((alias_bonus + aggregate_bonus + range_bonus + total_range_bonus + variance_bonus, float(series.max()), col))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2] if candidates else None


def estimate_numeric_risk_threshold(series):
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty or non_null.nunique() <= 1:
        return None

    if non_null.between(0, 100).mean() >= 0.8:
        quantile_cutoff = float(non_null.quantile(0.25))
        std = float(non_null.std(ddof=0)) if non_null.shape[0] > 1 else 0.0
        median_cutoff = float(non_null.median() - max(2.0, std * 0.4))
        return min(quantile_cutoff, median_cutoff)

    high_value = float(non_null.quantile(0.9))
    if high_value > 0:
        quantile_cutoff = float(non_null.quantile(0.25))
        scaled_cutoff = round(high_value * 0.45, 2)
        return min(quantile_cutoff, scaled_cutoff)

    return float(non_null.median())


def derive_score_outcome_series(series):
    threshold = estimate_numeric_risk_threshold(series)
    if threshold is None:
        return None

    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric >= threshold).astype("Int64")


def derive_subject_outcome_series(df, subject_columns):
    if len(subject_columns) < 2:
        return None

    derived = pd.Series(pd.NA, index=df.index, dtype="Int64")
    fail_mask = pd.Series(False, index=df.index)
    seen_mask = pd.Series(False, index=df.index)

    for col in subject_columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        threshold = estimate_numeric_risk_threshold(numeric)
        if threshold is None:
            continue

        has_value = numeric.notna()
        seen_mask = seen_mask | has_value
        fail_mask = fail_mask | (has_value & (numeric < threshold))

    if not seen_mask.any():
        return None

    derived.loc[seen_mask] = 1
    derived.loc[fail_mask] = 0
    return derived


def derive_attendance_outcome_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty or non_null.nunique() <= 1:
        return None

    low_cutoff = float(non_null.quantile(0.25))
    high_cutoff = float(non_null.quantile(0.75))
    if high_cutoff <= low_cutoff:
        high_cutoff = float(non_null.median())

    derived = pd.Series(pd.NA, index=series.index, dtype="Int64")
    derived.loc[numeric < low_cutoff] = 0
    derived.loc[numeric >= high_cutoff] = 1
    return derived


def add_derived_feature_columns(df, schema=None):
    schema = schema or infer_dataset_schema(df)
    feature_df = df.copy()

    attendance_key = schema.get("attendance_key")
    score_key = schema.get("score_key")

    if attendance_key and attendance_key in feature_df.columns:
        attendance = pd.to_numeric(feature_df[attendance_key], errors="coerce")
        attendance_non_null = attendance.dropna()
        low_cutoff = float(attendance_non_null.quantile(0.25)) if not attendance_non_null.empty else None
        high_cutoff = float(attendance_non_null.quantile(0.75)) if not attendance_non_null.empty else None
        feature_df["attendance_available"] = attendance.notna().astype(int)
        feature_df["attendance_low_flag"] = attendance.map(
            lambda value: 1 if low_cutoff is not None and pd.notna(value) and value < low_cutoff else 0
        )
        feature_df["attendance_high_flag"] = attendance.map(
            lambda value: 1 if high_cutoff is not None and pd.notna(value) and value >= high_cutoff else 0
        )

        if score_key and score_key in feature_df.columns:
            score = pd.to_numeric(feature_df[score_key], errors="coerce")
            feature_df["score_attendance_product"] = score * attendance
            feature_df["score_minus_attendance"] = score - attendance

    return feature_df


def derive_composite_target(df, schema=None):
    schema = schema or infer_dataset_schema(df)
    candidates = []

    def add_candidate(candidate, source_name, priority):
        if candidate is None:
            return
        candidate = pd.to_numeric(candidate, errors="coerce").astype("Int64")
        non_null = candidate.dropna()
        if non_null.empty:
            return
        variation = int(non_null.nunique())
        coverage = int(non_null.shape[0])
        candidates.append({
            "source": source_name,
            "series": candidate,
            "variation": variation,
            "coverage": coverage,
            "priority": priority,
        })

    grade_key = schema.get("grade_key")
    if grade_key and grade_key in df.columns:
        add_candidate(coerce_boolean_series(df[grade_key]), f"{grade_key}:label", 100)

    for col in df.columns:
        if col == grade_key:
            continue
        token = normalize_token(col)
        if token in IDENTIFIER_HINTS or is_attendance_related_token(token):
            continue

        if any(hint in token for hint in BACKLOG_HINTS):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                backlog_outcome = numeric.map(
                    lambda value: 0 if pd.notna(value) and value > 0 else (1 if pd.notna(value) else pd.NA)
                )
                add_candidate(backlog_outcome, f"{col}:backlog", 95)
                continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            add_candidate(coerce_boolean_series(df[col]), f"{col}:label", 90)

    score_key = schema.get("score_key")
    if score_key and score_key in df.columns:
        add_candidate(derive_score_outcome_series(df[score_key]), f"{score_key}:score", 80)

    subject_candidate = derive_subject_outcome_series(df, schema.get("subject_columns") or [])
    add_candidate(subject_candidate, "subject_columns", 70)

    attendance_key = schema.get("attendance_key")
    if attendance_key and attendance_key in df.columns:
        add_candidate(derive_attendance_outcome_series(df[attendance_key]), f"{attendance_key}:attendance", 60)

    if not candidates:
        return df, None

    ranked = sorted(
        candidates,
        key=lambda item: (item["variation"] >= 2, item["variation"], item["priority"], item["coverage"]),
        reverse=True,
    )
    best = ranked[0]

    if best["variation"] >= 2:
        target_name = "AI_Target"
        source_name = "AI_Target_Source"
        result_df = df.copy()
        result_df[target_name] = best["series"]
        result_df[source_name] = best["source"]
        return result_df, target_name

    # If no single signal has enough variation, try combining them row-wise.
    derived = pd.Series(pd.NA, index=df.index, dtype="Int64")
    source = pd.Series(pd.NA, index=df.index, dtype="object")
    for candidate in ranked:
        mask = candidate["series"].notna() & derived.isna()
        if mask.any():
            derived.loc[mask] = candidate["series"].loc[mask]
            source.loc[mask] = candidate["source"]

    if derived.dropna().nunique() >= 2:
        target_name = "AI_Target"
        source_name = "AI_Target_Source"
        result_df = df.copy()
        result_df[target_name] = derived
        result_df[source_name] = source
        return result_df, target_name

    return df, None


def infer_subject_columns(df, schema=None):
    schema = schema or {}
    excluded = {
        schema.get("name_key"),
        schema.get("id_key"),
        schema.get("attendance_key"),
        schema.get("score_key"),
        schema.get("grade_key"),
        "AI_Target",
        "AI_Prediction",
        "AI_Prediction_Label",
    }
    excluded = {col for col in excluded if col}

    subject_columns = []
    numeric = df.select_dtypes(include=["number"])
    for col in numeric.columns:
        if col in excluded:
            continue

        token = normalize_token(col)
        raw_col = str(col).strip().lower()
        if raw_col in {"%", "percent", "percentage"}:
            continue
        if re.fullmatch(r"column\d+", token):
            continue
        if token in IDENTIFIER_HINTS:
            continue
        if is_attendance_related_token(token):
            continue
        if token_matches(token, NON_PERFORMANCE_HINTS):
            continue
        if token_matches(token, TARGET_HINTS):
            continue
        if any(marker in token for marker in {"total", "overall", "aggregate", "percent", "percentage", "cgpa", "sgpa", "gpa"}):
            continue

        series = pd.to_numeric(numeric[col], errors="coerce").dropna()
        if series.empty or series.nunique() <= 1:
            continue
        if series.max() <= 10 and series.nunique() <= 8:
            continue

        subject_columns.append(col)

    return subject_columns


def infer_dataset_schema(df):
    columns = list(df.columns)
    normalized_lookup = {col: normalize_token(col) for col in columns}

    def find_exact_or_contains(hints, numeric_only=False):
        ranked = []
        for col in columns:
            token = normalized_lookup[col]
            if numeric_only and not pd.api.types.is_numeric_dtype(df[col]):
                continue

            score = 0
            if token in hints:
                score += 10
            if any(hint in token for hint in hints):
                score += 4
            if score:
                ranked.append((score, col))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1] if ranked else None

    def find_best_name_column():
        ranked = []
        for col in columns:
            token = normalized_lookup[col]
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            if token_matches(token, TARGET_HINTS) or token_matches(token, NON_PERFORMANCE_HINTS):
                continue
            if is_attendance_related_token(token):
                continue

            series = df[col].dropna().astype(str).str.strip()
            if series.empty:
                continue

            non_blank = series[series != ""]
            if non_blank.empty:
                continue

            unique_count = non_blank.nunique()
            unique_ratio = unique_count / max(len(non_blank), 1)
            avg_length = non_blank.map(len).mean()
            numeric_like_ratio = non_blank.map(lambda value: bool(re.fullmatch(r"[\d\W]+", value))).mean()

            score = 0
            if token_matches(token, CANONICAL_COLUMN_HINTS["student_name"]):
                score += 10
            if unique_ratio >= 0.5:
                score += 4
            if 4 <= avg_length <= 40:
                score += 3
            if numeric_like_ratio <= 0.2:
                score += 3
            if token in IDENTIFIER_HINTS:
                score -= 3

            if score > 0:
                ranked.append((score, unique_ratio, avg_length, col))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return ranked[0][3] if ranked else None

    def find_best_outcome_column():
        ranked = []
        for col in columns:
            token = normalized_lookup[col]
            if token in IDENTIFIER_HINTS or is_attendance_related_token(token):
                continue

            if not is_likely_outcome_column(df[col]):
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                outcome_series = pd.to_numeric(df[col], errors="coerce")
            else:
                outcome_series = coerce_boolean_series(df[col])
                if outcome_series is None:
                    continue

            non_null = pd.to_numeric(outcome_series, errors="coerce").dropna()
            if non_null.empty or non_null.nunique() <= 1:
                continue

            score = 0
            if token_matches(token, CANONICAL_COLUMN_HINTS["result"]) or token_matches(token, TARGET_HINTS):
                score += 10
            if pd.api.types.is_numeric_dtype(df[col]) and non_null.nunique() <= 3:
                score += 6
            if re.fullmatch(r"column\d+", token):
                score += 2
            score += min(int(non_null.shape[0] // 5), 6)
            ranked.append((score, int(non_null.shape[0]), col))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked[0][2] if ranked else None

    name_key = find_exact_or_contains(CANONICAL_COLUMN_HINTS["student_name"]) or find_best_name_column()
    id_key = find_exact_or_contains(CANONICAL_COLUMN_HINTS["student_id"])
    attendance_key = find_exact_or_contains(CANONICAL_COLUMN_HINTS["attendance"], numeric_only=True)
    grade_key = find_exact_or_contains(CANONICAL_COLUMN_HINTS["result"]) or find_best_outcome_column()
    if grade_key and not is_likely_outcome_column(df[grade_key]):
        grade_key = None
    if grade_key is None:
        grade_key = find_best_outcome_column()
    score_key = (
        find_exact_or_contains(CANONICAL_COLUMN_HINTS["percentage"], numeric_only=True)
        or find_exact_or_contains(CANONICAL_COLUMN_HINTS["total_score"], numeric_only=True)
        or choose_performance_score_column(df)
    )

    schema = {
        "name_key": name_key,
        "id_key": id_key,
        "attendance_key": attendance_key,
        "grade_key": grade_key,
        "score_key": score_key,
    }
    schema["subject_columns"] = infer_subject_columns(df, schema=schema)
    schema["has_attendance"] = attendance_key is not None
    schema["has_result"] = grade_key is not None
    schema["prediction_key"] = (
        "AI_Target" if "AI_Target" in columns
        else ("AI_Prediction" if "AI_Prediction" in columns else None)
    )
    return schema


def filter_non_student_rows(df, schema=None):
    if df.empty:
        return df

    schema = schema or infer_dataset_schema(df)
    filtered = df.copy()

    name_key = schema.get("name_key")
    if name_key and name_key in filtered.columns:
        name_series = filtered[name_key].astype(str).str.strip()
        normalized_name = name_series.map(normalize_token)
        valid_name_mask = (
            name_series.ne("")
            & ~normalized_name.isin(NON_STUDENT_ROW_HINTS)
            & ~normalized_name.str.contains(r"maximummarks|nameofstudent|subjectcode|subjectname", regex=True, na=False)
        )
        filtered = filtered.loc[valid_name_mask]

    id_key = schema.get("id_key")
    if id_key and id_key in filtered.columns:
        id_numeric = pd.to_numeric(filtered[id_key], errors="coerce")
        if id_numeric.notna().sum() >= max(3, len(filtered) // 3):
            filtered = filtered.loc[id_numeric.notna()]

    score_key = schema.get("score_key")
    if score_key and score_key in filtered.columns:
        score_numeric = pd.to_numeric(filtered[score_key], errors="coerce")
        filtered = filtered.loc[score_numeric.notna()]

    filtered = filtered.dropna(axis=0, how="all")
    return filtered.reset_index(drop=True)


def enrich_dataframe_for_learning(df):
    df = df.copy()
    schema = infer_dataset_schema(df)
    df = filter_non_student_rows(df, schema=schema)
    schema = infer_dataset_schema(df)

    score_key = schema.get("score_key")
    subject_columns = schema.get("subject_columns") or []
    grade_key = schema.get("grade_key")

    if not score_key and len(subject_columns) >= 2:
        df["derived_total_score"] = df[subject_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        score_key = "derived_total_score"
        schema["score_key"] = score_key

    if not grade_key and score_key and score_key in df.columns:
        numeric_scores = pd.to_numeric(df[score_key], errors="coerce")
        non_null = numeric_scores.dropna()
        if not non_null.empty and non_null.nunique() > 1:
            threshold = estimate_numeric_risk_threshold(numeric_scores)
            df["derived_result"] = numeric_scores.map(
                lambda value: "Pass" if pd.notna(value) and value >= threshold else ("Fail" if pd.notna(value) else pd.NA)
            )
            schema["grade_key"] = "derived_result"
            schema["has_result"] = True

    df, composite_target = derive_composite_target(df, schema=schema)
    if composite_target:
        schema["prediction_key"] = composite_target

    schema["subject_columns"] = infer_subject_columns(df, schema=schema)
    return df, schema


def column_has_explicit_performance_signal(column_name):
    token = normalize_token(column_name)
    if not token:
        return False
    if is_attendance_related_token(token):
        return False
    return token_matches(token, SCORE_HINTS) or token_matches(token, TARGET_HINTS)


def classify_dataset_role(df, schema=None, filename=None):
    schema = schema or infer_dataset_schema(df)

    name_key = schema.get("name_key")
    id_key = schema.get("id_key")
    score_key = schema.get("score_key")
    grade_key = schema.get("grade_key")
    attendance_key = schema.get("attendance_key")
    subject_columns = schema.get("subject_columns") or []
    attendance_context_columns = [
        col for col in df.columns
        if is_attendance_related_token(normalize_token(col))
    ]

    has_student_identity = bool(name_key or id_key)
    has_attendance_signal = bool(attendance_key)
    has_attendance_context = bool(attendance_context_columns)
    has_explicit_performance_signal = bool(
        grade_key
        or (score_key and column_has_explicit_performance_signal(score_key))
    )
    has_marksheet_like_subjects = len(subject_columns) >= 2 and not has_attendance_context

    normalized_filename = normalize_token(filename or "")
    filename_suggests_attendance = "attendance" in normalized_filename
    filename_suggests_result = any(hint in normalized_filename for hint in ("result", "marks", "score", "grade"))

    if not has_student_identity:
        return DATASET_ROLE_UNSUPPORTED

    if has_attendance_context and not has_explicit_performance_signal:
        return DATASET_ROLE_ATTENDANCE

    if filename_suggests_attendance and has_attendance_context and not has_explicit_performance_signal:
        return DATASET_ROLE_ATTENDANCE

    if has_explicit_performance_signal:
        return DATASET_ROLE_PERFORMANCE

    if has_marksheet_like_subjects and not filename_suggests_attendance:
        return DATASET_ROLE_PERFORMANCE

    if filename_suggests_result and (score_key or grade_key or subject_columns):
        return DATASET_ROLE_PERFORMANCE

    if has_attendance_context:
        return DATASET_ROLE_ATTENDANCE

    return DATASET_ROLE_UNSUPPORTED


def reject_unsupported_dataset(source_type):
    raise HTTPException(
        400,
        f"This {source_type.upper()} file is not valid for learning. Upload a student result file, or a combined result-and-attendance file.",
    )


def reject_attendance_only_dataset(source_type):
    raise HTTPException(
        400,
        f"This {source_type.upper()} file appears to be an attendance sheet only. Performance cannot be predicted from attendance alone. Upload a student result file, or a PDF/CSV that contains both result and attendance together.",
    )


def build_merge_key_series(series):
    return series.map(lambda value: normalize_token(value) if pd.notna(value) else "")


def choose_merge_columns(existing_df, incoming_df):
    existing_schema = infer_dataset_schema(existing_df)
    incoming_schema = infer_dataset_schema(incoming_df)

    for schema_key in ("id_key", "name_key"):
        existing_key = existing_schema.get(schema_key)
        incoming_key = incoming_schema.get(schema_key)
        if not existing_key or not incoming_key:
            continue
        if existing_key not in existing_df.columns or incoming_key not in incoming_df.columns:
            continue

        existing_keys = set(build_merge_key_series(existing_df[existing_key]))
        incoming_keys = set(build_merge_key_series(incoming_df[incoming_key]))
        overlap = len({key for key in existing_keys & incoming_keys if key})
        if overlap:
            return existing_key, incoming_key

    return None, None


def merge_student_datasets(existing_df, incoming_df):
    if existing_df.empty:
        return incoming_df.copy()

    existing_key, incoming_key = choose_merge_columns(existing_df, incoming_df)
    if not existing_key or not incoming_key:
        raise HTTPException(
            400,
            "Attendance file uploaded, but it could not be matched with the existing student result data. Use a common student ID or student name column.",
        )

    base = existing_df.copy()
    extra = incoming_df.copy()

    base["_merge_key"] = build_merge_key_series(base[existing_key])
    extra["_merge_key"] = build_merge_key_series(extra[incoming_key])
    extra = extra.loc[extra["_merge_key"] != ""].drop_duplicates("_merge_key", keep="first")

    if extra.empty:
        raise HTTPException(400, "Attendance file did not contain usable student rows to merge.")

    skip_columns = {incoming_key}
    incoming_columns = [col for col in extra.columns if col not in skip_columns]
    merged = base.merge(extra[["_merge_key"] + incoming_columns], on="_merge_key", how="left", suffixes=("", "__incoming"))

    for col in incoming_columns:
        incoming_col = f"{col}__incoming"
        if incoming_col not in merged.columns:
            continue

        if col in merged.columns:
            merged[col] = merged[col].combine_first(merged[incoming_col])
            merged = merged.drop(columns=[incoming_col])
        else:
            merged = merged.rename(columns={incoming_col: col})

    return merged.drop(columns=["_merge_key"], errors="ignore")


def decode_bytes(raw):
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def read_csv_bytes(raw):
    if not raw:
        raise HTTPException(400, "Empty CSV")

    separators = [None, ",", ";", "\t", "|"]

    text = decode_bytes(raw)
    if text is None:
        raise HTTPException(400, "CSV Error: Could not decode this file")

    for separator in separators:
        try:
            kwargs = {
                "skipinitialspace": True,
                "on_bad_lines": "skip",
            }
            if separator is None:
                kwargs["sep"] = None
                kwargs["engine"] = "python"
            else:
                kwargs["sep"] = separator

            df = pd.read_csv(io.StringIO(text), **kwargs)
            if not df.empty and df.shape[1] >= 1:
                return df
        except Exception:
            continue

    raise HTTPException(400, "CSV Error: Could not read this file format")


def read_uploaded_csv(upload):
    return read_csv_bytes(upload.file.read())


def table_from_text_rows(rows):
    if len(rows) < 2:
        return None

    rows = merge_header_rows(rows)

    header = make_unique_columns(rows[0])
    expected_width = len(header)
    if expected_width < 2:
        return None

    normalized_rows = []
    for row in rows[1:]:
        if len(row) != expected_width:
            continue
        normalized_rows.append(row)

    if not normalized_rows:
        return None

    return pd.DataFrame(normalized_rows, columns=header)


def extract_table_rows_from_pdf_text(text):
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line and line.strip()]
    candidates = []

    for separator in PDF_TABLE_SEPARATORS:
        split_rows = []
        for line in lines:
            if separator not in line:
                continue
            if separator == "\t":
                parts = [part.strip() for part in line.split(separator)]
            else:
                parts = next(
                    csv.reader([line], delimiter=separator, skipinitialspace=True),
                    [],
                )
                parts = [part.strip() for part in parts]

            parts = [part for part in parts if part != ""]
            if len(parts) >= 2:
                split_rows.append(parts)

        if split_rows:
            width_counts = {}
            for row in split_rows:
                width_counts[len(row)] = width_counts.get(len(row), 0) + 1

            best_width = max(width_counts, key=width_counts.get)
            best_rows = [row for row in split_rows if len(row) == best_width]
            if len(best_rows) >= 2:
                candidates.append(best_rows)

    whitespace_rows = []
    for line in lines:
        parts = [part.strip() for part in re.split(r"\s{2,}", line) if part.strip()]
        if len(parts) >= 2:
            whitespace_rows.append(parts)
    if whitespace_rows:
        width_counts = {}
        for row in whitespace_rows:
            width_counts[len(row)] = width_counts.get(len(row), 0) + 1
        best_width = max(width_counts, key=width_counts.get)
        best_rows = [row for row in whitespace_rows if len(row) == best_width]
        if len(best_rows) >= 2:
            candidates.append(best_rows)

    candidates.sort(key=lambda rows: (len(rows), len(rows[0])), reverse=True)
    return candidates[0] if candidates else []


def extract_pdf_tables(raw):
    text_chunks = []
    frames = []

    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text)

                for table in page.extract_tables() or []:
                    cleaned_rows = []
                    for row in table:
                        if not row:
                            continue
                        cleaned_row = [
                            str(cell).strip() if cell is not None else ""
                            for cell in row
                        ]
                        if any(cleaned_row):
                            cleaned_rows.append(cleaned_row)

                    frame = table_from_text_rows(cleaned_rows)
                    if frame is not None:
                        frames.append(frame)
    except ImportError:
        pass
    except Exception:
        pass

    if not text_chunks:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(io.BytesIO(raw))
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text)
        except ImportError:
            pass
        except Exception:
            pass

    return frames, "\n".join(text_chunks)


def choose_pdf_dataframe(frames, text):
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if valid_frames:
        valid_frames.sort(key=lambda frame: (frame.shape[0], frame.shape[1]), reverse=True)
        return promote_embedded_pdf_header(valid_frames[0])

    rows = extract_table_rows_from_pdf_text(text)
    frame = table_from_text_rows(rows)
    if frame is not None:
        return frame

    raise HTTPException(
        400,
        "PDF Error: Could not find a usable table. Upload a PDF with a clear student table, or install pdfplumber/pypdf for better extraction.",
    )


def read_uploaded_pdf(upload):
    raw = upload.file.read()
    if not raw:
        raise HTTPException(400, "Empty PDF")

    frames, text = extract_pdf_tables(raw)
    return choose_pdf_dataframe(frames, text)


def read_uploaded_dataframe(upload):
    filename = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()

    if filename.endswith(".pdf") or "pdf" in content_type:
        return read_uploaded_pdf(upload), "pdf"

    if filename.endswith(".csv") or "csv" in content_type or content_type in {
        "application/vnd.ms-excel",
        "text/plain",
        "",
    }:
        return read_uploaded_csv(upload), "csv"

    raise HTTPException(400, "Unsupported file. Please upload a CSV or PDF file.")


def build_ai_target(df, schema=None):
    schema = schema or infer_dataset_schema(df)
    if "AI_Target" in df.columns:
        target_series = pd.to_numeric(df["AI_Target"], errors="coerce")
        if target_series.dropna().nunique() >= 2:
            return "AI_Target"

    score_column = schema.get("score_key") or choose_performance_score_column(df)
    if score_column:
        numeric_scores = pd.to_numeric(df[score_column], errors="coerce")
        non_null = numeric_scores.dropna()
        if not non_null.empty and non_null.nunique() > 1:
            threshold = estimate_numeric_risk_threshold(numeric_scores)
            df["AI_Target"] = (numeric_scores >= threshold).astype(int)
            return "AI_Target"

    numeric = df.select_dtypes(include=["number"])
    usable_numeric = []
    for col in numeric.columns:
        token = normalize_token(col)
        if token in IDENTIFIER_HINTS:
            continue
        if is_attendance_related_token(token):
            continue
        if numeric[col].dropna().nunique() > 1:
            usable_numeric.append(col)

    if usable_numeric:
        source_col = usable_numeric[-1]
        threshold = numeric[source_col].median()
        df["AI_Target"] = (numeric[source_col] >= threshold).astype(int)
        return "AI_Target"

    text_candidates = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        token = normalize_token(col)
        if token in IDENTIFIER_HINTS:
            continue
        if is_attendance_related_token(token):
            continue

        unique_count = series.astype(str).nunique()
        unique_ratio = unique_count / max(len(series), 1)
        if 1 < unique_count <= 10 and unique_ratio < 0.8:
            text_candidates.append(col)

    if text_candidates:
        return text_candidates[-1]

    if len(df.columns) >= 2:
        return df.columns[-1]

    raise Exception(
        "Could not detect a target column. Add a result-like column or include at least one meaningful numeric column."
    )

# ---------------- TARGET DETECTION ----------------

def detect_target(df, schema=None):
    schema = schema or infer_dataset_schema(df)

    if "AI_Target" in df.columns and pd.to_numeric(df["AI_Target"], errors="coerce").dropna().nunique() >= 2:
        return "AI_Target"

    grade_key = schema.get("grade_key")
    if grade_key and grade_key in df.columns:
        return grade_key

    score_key = schema.get("score_key")
    subject_columns = schema.get("subject_columns") or []
    if score_key or len(subject_columns) >= 2:
        return build_ai_target(df, schema=schema)

    for col in df.columns:
        token = normalize_token(col)
        if token in TARGET_HINTS:
            return col

    for col in reversed(df.columns):
        token = normalize_token(col)
        if token in IDENTIFIER_HINTS:
            continue
        if is_attendance_related_token(token):
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        unique_count = series.astype(str).nunique()
        unique_ratio = unique_count / max(len(series), 1)
        if 1 < unique_count <= min(20, max(4, len(series) // 2)) and unique_ratio < 0.9:
            return col

    return build_ai_target(df, schema=schema)


def prepare_target(series):
    series = series.copy()
    series = series.map(normalize_cell)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        boolean_series = coerce_boolean_series(series)
        if boolean_series is not None:
            return pd.to_numeric(boolean_series, errors="coerce"), {"labels": {0: "negative", 1: "positive"}}

        numeric_candidate = coerce_numeric_series(series, min_ratio=0.8)
        numeric = numeric_candidate if numeric_candidate is not None else None

    if numeric is not None:
        numeric = pd.to_numeric(numeric, errors="coerce")
        non_null = numeric.dropna()
        if non_null.empty or non_null.nunique() <= 1:
            raise Exception("Target column does not contain enough variation")

        # Convert continuous numeric targets into a binary outcome so prediction
        # still works cleanly in the dashboard.
        if non_null.nunique() > min(12, max(5, len(non_null) // 3)):
            threshold = float(non_null.median())
            binary = (numeric >= threshold).astype("Int64")
            return binary, {"labels": {0: "Below Median", 1: "At or Above Median"}, "threshold": threshold}

        return numeric, None

    text = series.map(lambda value: str(value).strip() if pd.notna(value) else pd.NA)
    non_null = text.dropna()
    if non_null.empty or non_null.nunique() <= 1:
        raise Exception("Target column does not contain enough variation")

    categories = sorted(non_null.unique().tolist())
    mapping = {label: idx for idx, label in enumerate(categories)}
    inverse_mapping = {idx: label for label, idx in mapping.items()}
    encoded = text.map(lambda value: mapping.get(value, pd.NA) if pd.notna(value) else pd.NA)
    return pd.to_numeric(encoded, errors="coerce"), {"labels": inverse_mapping}


def select_feature_columns(df, target=None, schema=None):
    schema = schema or infer_dataset_schema(df)
    allow_attendance = bool(schema.get("attendance_key"))
    kept_columns = []

    for col in df.columns:
        if col == target:
            continue

        token = normalize_token(col)
        if token in IDENTIFIER_HINTS:
            continue
        if token_matches(token, NON_PERFORMANCE_HINTS):
            continue
        if is_attendance_related_token(token) and not allow_attendance:
            continue

        # Prevent the model from learning from outcome/result columns that would
        # leak the answer instead of predicting performance from inputs.
        if token_matches(token, TARGET_HINTS) and not is_attendance_related_token(token):
            continue

        if target:
            target_token = normalize_token(target)
            if token_matches(target_token, TARGET_HINTS):
                if token_matches(token, SCORE_HINTS) and any(marker in token for marker in {"final", "total", "overall", "percentage", "percent", "grade", "result"}):
                    continue

        kept_columns.append(col)

    if not kept_columns:
        kept_columns = [col for col in df.columns if col != target]

    return df[kept_columns].copy()


def prepare_features(df, target=None, feature_columns=None, schema=None):
    feature_df = select_feature_columns(df, target=target, schema=schema)
    feature_df = add_derived_feature_columns(feature_df, schema=schema)

    feature_df = pd.get_dummies(feature_df, dummy_na=True)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.fillna(feature_df.mean()).fillna(0)

    if feature_columns is not None:
        feature_df = feature_df.reindex(columns=feature_columns, fill_value=0)

    return feature_df

# ---------------- MODELS ----------------


class RegisterPayload(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginPayload(BaseModel):
    email: EmailStr
    password: str


class AdminLoginPayload(BaseModel):
    email: EmailStr
    password: str


class ForgotPasswordPayload(BaseModel):
    email: EmailStr
    new_password: str


class AdminUserUpdatePayload(BaseModel):
    original_email: EmailStr
    name: str
    email: EmailStr
    password: str | None = None


class AttendanceUpdatePayload(BaseModel):
    student_id: str | None = None
    student_name: str | None = None
    attendance: float
    source: str | None = "manual"


class RFIDAttendancePayload(BaseModel):
    rfid: str
    lecture_id: str | None = None
    source: str | None = "rfid"


class AttendanceConfigPayload(BaseModel):
    total_lectures: int


class RFIDLinkPayload(BaseModel):
    rfid: str
    student_id: str | None = None
    student_name: str | None = None


class AttendanceResetPayload(BaseModel):
    student_id: str | None = None
    student_name: str | None = None
    lecture_id: str | None = None


PASSWORD_RULE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*?&]{8,}$")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@edupredict.com").strip().lower()
ADMIN_PASSWORD_HASH = hashlib.sha256(
    os.getenv("ADMIN_PASSWORD", "Admin1234").encode()
).hexdigest()


def ensure_valid_password(password: str):
    if not PASSWORD_RULE.fullmatch(password):
        raise HTTPException(
            400,
            "Password must be at least 8 characters and include at least 1 letter and 1 number.",
        )


def authenticate_admin(payload: AdminLoginPayload):
    normalized_email = payload.email.strip().lower()
    hashed_password = hashlib.sha256(payload.password.encode()).hexdigest()

    if normalized_email != ADMIN_EMAIL or hashed_password != ADMIN_PASSWORD_HASH:
        raise HTTPException(401, "Invalid admin credentials")

    return {
        "message": "Admin login successful",
        "email": normalized_email,
        "role": "admin",
    }

# ---------------- TRAIN ----------------

def train_model(df):
    schema = infer_dataset_schema(df)
    target = detect_target(df, schema=schema)

    X = prepare_features(df, target=target, schema=schema)
    y, target_meta = prepare_target(df[target])

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if X.empty:
        raise Exception("No usable feature columns")

    if y.nunique() <= 1:
        raise Exception("Target column does not contain enough variation")

    if len(X) < 2:
        raise Exception("Need at least two usable rows to train the model")

    test_size = 0.2 if len(X) >= 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump({
        "model": model,
        "features": X.columns.tolist(),
        "target": target,
        "target_meta": target_meta or {}
    }, MODEL_FILE)

    return acc

# ---------------- PREDICT ----------------

def predict(df):

    if not MODEL_FILE.exists():
        return df

    saved = joblib.load(MODEL_FILE)

    model = saved["model"]
    features = saved["features"]
    target_meta = saved.get("target_meta", {})

    df = df.copy()

    X = prepare_features(df, feature_columns=features)

    try:
        predictions = pd.Series(model.predict(X), index=df.index)
        df["AI_Prediction"] = predictions

        labels = target_meta.get("labels")
        if labels:
            label_map = {int(key): value for key, value in labels.items()}
            df["AI_Prediction_Label"] = predictions.map(
                lambda value: label_map.get(int(value), value) if pd.notna(value) else value
            )
    except Exception:
        pass

    return df


def derive_outcomes_from_uploaded_data(df, schema=None):
    schema = schema or infer_dataset_schema(df)
    derived_df = df.copy()

    if "AI_Target" in derived_df.columns:
        target_series = pd.to_numeric(derived_df["AI_Target"], errors="coerce").astype("Int64")
        if target_series.dropna().nunique() >= 1:
            derived_df["AI_Prediction"] = target_series
            derived_df["AI_Prediction_Label"] = target_series.map(
                lambda value: "positive" if pd.notna(value) and int(value) == 1 else ("negative" if pd.notna(value) else pd.NA)
            )
            return derived_df, True

    grade_key = schema.get("grade_key")
    if grade_key and grade_key in derived_df.columns:
        derived = coerce_boolean_series(derived_df[grade_key])
        if derived is not None and derived.dropna().nunique() >= 1:
            derived_df["AI_Prediction"] = derived.astype("Int64")
            derived_df["AI_Prediction_Label"] = derived.map(
                lambda value: "positive" if pd.notna(value) and int(value) == 1 else ("negative" if pd.notna(value) else pd.NA)
            )
            return derived_df, True

    score_key = schema.get("score_key")
    if score_key and score_key in derived_df.columns:
        numeric_scores = pd.to_numeric(derived_df[score_key], errors="coerce")
        non_null = numeric_scores.dropna()
        if not non_null.empty and non_null.nunique() > 1:
            threshold = estimate_numeric_risk_threshold(numeric_scores)
            derived = (numeric_scores >= threshold).astype("Int64")
            derived_df["AI_Prediction"] = derived
            derived_df["AI_Prediction_Label"] = derived.map(
                lambda value: "positive" if pd.notna(value) and int(value) == 1 else ("negative" if pd.notna(value) else pd.NA)
            )
            return derived_df, True

    return derived_df, False


def load_existing_students_dataframe():
    records = list(students_col.find({}, {"_id": 0}))
    records = sanitize_records(records)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def retrain_and_persist_dataset(full_df, source_type, message_prefix):
    schema = infer_dataset_schema(full_df)
    fallback_message = None
    try:
        acc = train_model(full_df)
        full_df = predict(full_df)
    except Exception as e:
        full_df, derived = derive_outcomes_from_uploaded_data(full_df, schema=schema)
        if not derived:
            raise HTTPException(400, f"ML Error: {str(e)}")
        acc = None
        fallback_message = (
            f"{message_prefix} The data was stored and analyzed, "
            f"but the model could not be retrained on this dataset. Reason: {str(e)}."
        )

    return persist_analysis(
        full_df,
        acc,
        source_type,
        dataset_role=DATASET_ROLE_PERFORMANCE,
        training_used=acc is not None,
        message=fallback_message or f"{message_prefix} The model was retrained from the updated dataset.",
    )


def apply_daily_attendance_update(payload: AttendanceUpdatePayload):
    existing_df = load_existing_students_dataframe()
    if existing_df.empty:
        raise HTTPException(400, "No student result dataset found yet. Upload result data before updating daily attendance.")

    existing_df = clean_dataframe(existing_df)
    if existing_df.empty:
        raise HTTPException(400, "Stored student data is empty. Upload result data before updating daily attendance.")

    schema = infer_dataset_schema(existing_df)
    name_key = schema.get("name_key")
    id_key = schema.get("id_key")

    attendance_value = float(payload.attendance)
    if not (0 <= attendance_value <= 100):
        raise HTTPException(400, "Attendance must be between 0 and 100.")

    identifier_mask = pd.Series(False, index=existing_df.index)
    if payload.student_id and id_key and id_key in existing_df.columns:
        identifier_mask = identifier_mask | (
            existing_df[id_key].astype(str).str.strip().map(normalize_token) == normalize_token(payload.student_id)
        )
    if payload.student_name and name_key and name_key in existing_df.columns:
        identifier_mask = identifier_mask | (
            existing_df[name_key].astype(str).str.strip().map(normalize_token) == normalize_token(payload.student_name)
        )

    if not identifier_mask.any():
        raise HTTPException(404, "Student not found. Use a matching student name or student ID from the current dataset.")

    attendance_key = schema.get("attendance_key")
    if not attendance_key or attendance_key not in existing_df.columns:
        attendance_key = "attendance"
        if attendance_key not in existing_df.columns:
            existing_df[attendance_key] = pd.NA

    existing_df.loc[identifier_mask, attendance_key] = attendance_value
    if payload.source:
        if "attendance_source" not in existing_df.columns:
            existing_df["attendance_source"] = pd.NA
        existing_df.loc[identifier_mask, "attendance_source"] = str(payload.source).strip()

    updated_df = clean_dataframe(existing_df)
    updated_df, _ = enrich_dataframe_for_learning(updated_df)
    return retrain_and_persist_dataset(updated_df, "attendance-update", "Daily attendance updated successfully.")


def get_attendance_lecture_id(value=None):
    if value is not None:
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


def ensure_numeric_column(df, column_name, default=0):
    if column_name not in df.columns:
        df[column_name] = default
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce").fillna(default)
    return df


def ensure_text_column(df, column_name):
    if column_name not in df.columns:
        df[column_name] = pd.Series(pd.NA, index=df.index, dtype="object")
    elif df[column_name].dtype != "object":
        df[column_name] = df[column_name].astype("object")
    return df


def find_student_by_rfid(df, rfid, schema=None):
    schema = schema or infer_dataset_schema(df)
    normalized_rfid = normalize_token(rfid)
    if not normalized_rfid:
        return pd.Series(False, index=df.index), None

    candidate_columns = []
    for col in df.columns:
        token = normalize_token(col)
        if is_rfid_related_token(token):
            candidate_columns.append(col)

    id_key = schema.get("id_key")
    if id_key and id_key in df.columns and id_key not in candidate_columns:
        candidate_columns.append(id_key)

    for col in candidate_columns:
        mask = df[col].astype(str).str.strip().map(normalize_token) == normalized_rfid
        if mask.any():
            return mask, col

    return pd.Series(False, index=df.index), None


def find_student_by_identity(df, student_id=None, student_name=None, schema=None):
    schema = schema or infer_dataset_schema(df)
    name_key = schema.get("name_key")
    id_key = schema.get("id_key")

    identifier_mask = pd.Series(False, index=df.index)
    matched_by = None

    if student_id and id_key and id_key in df.columns:
        id_mask = (
            df[id_key].astype(str).str.strip().map(normalize_token) == normalize_token(student_id)
        )
        if id_mask.any():
            identifier_mask = identifier_mask | id_mask
            matched_by = id_key

    if student_name and name_key and name_key in df.columns:
        name_mask = (
            df[name_key].astype(str).str.strip().map(normalize_token) == normalize_token(student_name)
        )
        if name_mask.any():
            identifier_mask = identifier_mask | name_mask
            matched_by = matched_by or name_key

    return identifier_mask, matched_by


def build_scan_response(df, source_type, message, training_used=False, accuracy=None, extra=None):
    schema = infer_dataset_schema(df)
    response = {
        "message": message,
        "accuracy": accuracy,
        "columns": df.columns.tolist(),
        "rows": len(df),
        "source_type": source_type,
        "dataset_role": DATASET_ROLE_PERFORMANCE,
        "training_used": training_used,
        "schema": sanitize_value(schema),
        "preview_records": sanitize_records(df.head(3).to_dict(orient="records")),
    }
    if extra:
        response.update(sanitize_value(extra))
    return response


def get_effective_lecture_id(value=None):
    if value is not None:
        cleaned = str(value).strip()
        if cleaned:
            return cleaned

    settings = get_attendance_settings()
    lecture_id = settings.get("last_scan_lecture_id")
    if lecture_id:
        return lecture_id

    latest_event = attendance_events_col.find_one(sort=[("created_at", -1)])
    return latest_event.get("lecture_id") if latest_event else None


def get_attendance_settings():
    settings = attendance_settings_col.find_one({"_id": "global"}) or {}
    settings.pop("_id", None)
    total_lectures = settings.get("total_lectures")
    try:
        settings["total_lectures"] = int(total_lectures) if total_lectures is not None else None
    except (TypeError, ValueError):
        settings["total_lectures"] = None
    hidden_lecture_ids = settings.get("hidden_lecture_ids")
    if isinstance(hidden_lecture_ids, list):
        settings["hidden_lecture_ids"] = [str(item).strip() for item in hidden_lecture_ids if str(item).strip()]
    else:
        settings["hidden_lecture_ids"] = []
    dashboard_reset_at = settings.get("dashboard_reset_at")
    if dashboard_reset_at:
        settings["dashboard_reset_at"] = str(dashboard_reset_at).strip()
    else:
        settings["dashboard_reset_at"] = None
    return sanitize_value(settings)


def parse_datetime_value(value):
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def is_hidden_attendance_event(event, settings=None):
    settings = settings or get_attendance_settings()
    hidden_lecture_ids = set(settings.get("hidden_lecture_ids") or [])
    lecture_id = str(event.get("lecture_id") or "").strip()
    if lecture_id and lecture_id in hidden_lecture_ids:
        return True

    dashboard_reset_at = parse_datetime_value(settings.get("dashboard_reset_at"))
    created_at = parse_datetime_value(event.get("created_at"))
    if dashboard_reset_at and created_at and created_at <= dashboard_reset_at:
        return True

    return False


def get_rfid_mapping(rfid):
    normalized_rfid = normalize_token(rfid)
    if not normalized_rfid:
        return None
    mapping = rfid_mappings_col.find_one({"rfid": normalized_rfid})
    if not mapping:
        return None
    mapping.pop("_id", None)
    return sanitize_value(mapping)


def get_recent_attendance_scans(limit=8):
    settings = get_attendance_settings()
    scans = list(
        attendance_events_col.find(
            {},
            {
                "_id": 0,
                "lecture_id": 1,
                "rfid": 1,
                "student_name": 1,
                "matched_by": 1,
                "source": 1,
                "status": 1,
                "created_at": 1,
            },
        ).sort("created_at", -1)
    )
    visible_scans = [scan for scan in scans if not is_hidden_attendance_event(scan, settings=settings)]
    return sanitize_records(visible_scans[:limit])


def get_current_lecture_status_list(df, schema=None):
    if df is None or df.empty:
        return []

    schema = schema or infer_dataset_schema(df)
    settings = get_attendance_settings()
    lecture_id = settings.get("last_scan_lecture_id")
    if lecture_id and is_hidden_attendance_event({"lecture_id": lecture_id, "created_at": settings.get("last_scan_at")}, settings=settings):
        return []
    if not lecture_id:
        events = list(
            attendance_events_col.find(
                {},
                {
                    "_id": 0,
                    "lecture_id": 1,
                    "created_at": 1,
                },
            ).sort("created_at", -1)
        )
        latest_visible_event = next((event for event in events if not is_hidden_attendance_event(event, settings=settings)), None)
        lecture_id = latest_visible_event.get("lecture_id") if latest_visible_event else None
    if not lecture_id:
        return []

    lecture_events = list(
        attendance_events_col.find(
            {"lecture_id": lecture_id},
            {
                "_id": 0,
                "rfid": 1,
                "student_name": 1,
                "matched_by": 1,
                "status": 1,
                "created_at": 1,
            },
        )
    )
    if not lecture_events:
        return []

    present_names = set()
    present_entries = []
    undefined_entries = []
    for event in lecture_events:
        status = str(event.get("status") or "").strip().lower()
        if status in {"matched", "duplicate"} and event.get("student_name"):
            normalized_name = normalize_token(event.get("student_name"))
            if normalized_name in present_names:
                continue
            present_names.add(normalized_name)
            present_entries.append(
                {
                    "student_name": event.get("student_name"),
                    "status": "PRESENT",
                    "lecture_id": lecture_id,
                    "created_at": event.get("created_at"),
                }
            )
        elif status == "unmatched":
            undefined_entries.append(
                {
                    "student_name": event.get("student_name") or "Unknown card",
                    "rfid": str(event.get("rfid") or "").upper(),
                    "status": "UNDEFINED",
                    "lecture_id": lecture_id,
                    "created_at": event.get("created_at"),
                }
            )

    rows = present_entries + undefined_entries
    return sanitize_records(rows)


def rebuild_rfid_attendance_fields(df, schema=None):
    if df is None or df.empty:
        return df

    schema = schema or infer_dataset_schema(df)
    df = df.copy()
    ensure_numeric_column(df, "rfid_present_count", default=0)
    ensure_numeric_column(df, "rfid_session_count", default=0)
    ensure_text_column(df, "rfid_last_scan_at")
    ensure_text_column(df, "rfid_last_session_id")
    ensure_text_column(df, "rfid_scan_source")

    total_sessions = attendance_sessions_col.count_documents({})
    df["rfid_session_count"] = total_sessions
    df["rfid_present_count"] = 0
    df["rfid_last_scan_at"] = pd.NA
    df["rfid_last_session_id"] = pd.NA
    df["rfid_scan_source"] = pd.NA

    matched_events = list(
        attendance_events_col.find(
            {"status": {"$in": ["matched", "duplicate"]}},
            {
                "_id": 0,
                "lecture_id": 1,
                "student_name": 1,
                "created_at": 1,
                "source": 1,
            },
        )
    )

    grouped_events = {}
    for event in matched_events:
        normalized_name = normalize_token(event.get("student_name"))
        if not normalized_name:
            continue
        grouped_events.setdefault(normalized_name, []).append(event)

    for index, row in df.iterrows():
        student_name = get_student_display_name(row, schema=schema)
        normalized_name = normalize_token(student_name)
        student_events = grouped_events.get(normalized_name, [])
        if not student_events:
            continue

        unique_lectures = {
            str(event.get("lecture_id")).strip()
            for event in student_events
            if event.get("lecture_id")
        }
        df.at[index, "rfid_present_count"] = len(unique_lectures)

        latest_event = max(
            student_events,
            key=lambda event: event.get("created_at") or datetime.min,
        )
        created_at = latest_event.get("created_at")
        df.at[index, "rfid_last_scan_at"] = (
            created_at.isoformat() if isinstance(created_at, datetime) else sanitize_value(created_at)
        )
        df.at[index, "rfid_last_session_id"] = latest_event.get("lecture_id")
        df.at[index, "rfid_scan_source"] = latest_event.get("source")

    return df


def save_attendance_settings(total_lectures=None, **extra_fields):
    update_fields = {"updated_at": datetime.utcnow()}
    if total_lectures is not None:
        update_fields["total_lectures"] = int(total_lectures)
    update_fields.update(extra_fields)
    attendance_settings_col.update_one(
        {"_id": "global"},
        {"$set": update_fields},
        upsert=True,
    )
    return get_attendance_settings()


def get_student_display_name(row, schema=None):
    schema = schema or {}
    name_key = schema.get("name_key")
    id_key = schema.get("id_key")
    if name_key and row.get(name_key):
        return str(row.get(name_key)).strip()
    if id_key and row.get(id_key):
        return str(row.get(id_key)).strip()
    return "Unknown Student"


def recalculate_attendance_metrics(df, schema=None):
    schema = schema or infer_dataset_schema(df)
    df = df.copy()
    ensure_numeric_column(df, "rfid_present_count", default=0)
    ensure_numeric_column(df, "rfid_session_count", default=0)

    settings = get_attendance_settings()
    configured_total = settings.get("total_lectures")
    current_total = int(pd.to_numeric(df["rfid_session_count"], errors="coerce").fillna(0).max())
    effective_total = max(int(configured_total or 0), current_total)

    attendance_key = schema.get("attendance_key") or "attendance"
    if effective_total > 0:
        present_lectures = pd.to_numeric(df["rfid_present_count"], errors="coerce").fillna(0)
        df[attendance_key] = ((present_lectures / effective_total) * 100).round(2)
    else:
        if attendance_key not in df.columns:
            df[attendance_key] = 0.0
        else:
            df[attendance_key] = pd.to_numeric(df[attendance_key], errors="coerce").fillna(0)

    return df, attendance_key, effective_total


def update_total_lectures(payload: AttendanceConfigPayload):
    total_lectures = int(payload.total_lectures)
    if total_lectures < 0:
        raise HTTPException(400, "Total lectures cannot be negative.")

    settings = save_attendance_settings(total_lectures=total_lectures)
    existing_df = load_existing_students_dataframe()
    if existing_df.empty:
        return {
            "message": "Total lectures saved. Upload student data to start attendance tracking.",
            "attendance_settings": settings,
        }

    existing_df = clean_dataframe(existing_df)
    if existing_df.empty:
        return {
            "message": "Total lectures saved. Upload student data to start attendance tracking.",
            "attendance_settings": settings,
        }

    schema = infer_dataset_schema(existing_df)
    updated_df, _, effective_total = recalculate_attendance_metrics(existing_df, schema=schema)
    updated_df = clean_dataframe(updated_df)
    updated_df, _ = enrich_dataframe_for_learning(updated_df)
    result = retrain_and_persist_dataset(
        updated_df,
        "attendance-config",
        "Total lecture count updated successfully.",
    )
    result["attendance_settings"] = sanitize_value({**settings, "effective_total_lectures": effective_total})
    return result


def link_rfid_to_student(payload: RFIDLinkPayload):
    existing_df = load_existing_students_dataframe()
    if existing_df.empty:
        raise HTTPException(400, "No student dataset found yet. Upload result data before linking RFID cards.")

    existing_df = clean_dataframe(existing_df)
    if existing_df.empty:
        raise HTTPException(400, "Stored student data is empty. Upload result data before linking RFID cards.")

    schema = infer_dataset_schema(existing_df)
    name_key = schema.get("name_key")
    id_key = schema.get("id_key")
    normalized_rfid = normalize_token(payload.rfid)

    if not normalized_rfid:
        raise HTTPException(400, "RFID value is required.")

    identifier_mask, _ = find_student_by_identity(
        existing_df,
        student_id=payload.student_id,
        student_name=payload.student_name,
        schema=schema,
    )

    if not identifier_mask.any():
        raise HTTPException(404, "Student not found. Provide a matching student ID or student name from the current dataset.")

    existing_rfid_mask, existing_rfid_column = find_student_by_rfid(existing_df, payload.rfid, schema=schema)
    if existing_rfid_mask.any():
        existing_match = existing_df.loc[existing_rfid_mask].iloc[0]
        already_linked_name = get_student_display_name(existing_match, schema=schema)
        if not identifier_mask.equals(existing_rfid_mask):
            raise HTTPException(400, f"This RFID card is already linked to {already_linked_name}.")

    rfid_column = existing_rfid_column or "rfid"
    if rfid_column not in existing_df.columns:
        existing_df[rfid_column] = pd.NA
    else:
        ensure_text_column(existing_df, rfid_column)

    ensure_text_column(existing_df, "rfid_linked_at")

    existing_df.loc[identifier_mask, rfid_column] = payload.rfid.strip().upper()
    existing_df.loc[identifier_mask, "rfid_linked_at"] = datetime.utcnow().isoformat()

    matched_row = existing_df.loc[identifier_mask].iloc[0]
    matched_name = get_student_display_name(matched_row, schema=schema)
    matched_student_id = matched_row.get(id_key) if id_key and id_key in matched_row.index else None

    rfid_mappings_col.update_one(
        {"rfid": normalized_rfid},
        {
            "$set": {
                "rfid": normalized_rfid,
                "rfid_display": payload.rfid.strip().upper(),
                "student_name": matched_name,
                "student_id": sanitize_value(matched_student_id),
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )

    updated_df = clean_dataframe(existing_df)
    updated_df, _ = enrich_dataframe_for_learning(updated_df)
    result = retrain_and_persist_dataset(
        updated_df,
        "rfid-link",
        "RFID card linked successfully.",
    )

    matched_row = updated_df.loc[identifier_mask].iloc[0]
    result["rfid_link"] = sanitize_value(
        {
            "rfid": payload.rfid.strip().upper(),
            "student_name": get_student_display_name(matched_row, schema=infer_dataset_schema(updated_df)),
            "student_id": matched_row.get(id_key) if id_key and id_key in matched_row.index else None,
            "rfid_column": rfid_column,
        }
    )
    return result


def apply_rfid_attendance_scan(payload: RFIDAttendancePayload):
    existing_df = load_existing_students_dataframe()
    if existing_df.empty:
        raise HTTPException(400, "No student result dataset found yet. Upload result data before scanning RFID attendance.")

    existing_df = clean_dataframe(existing_df)
    if existing_df.empty:
        raise HTTPException(400, "Stored student data is empty. Upload result data before scanning RFID attendance.")

    schema = infer_dataset_schema(existing_df)
    lecture_id = get_attendance_lecture_id(payload.lecture_id)
    normalized_rfid = normalize_token(payload.rfid)
    identifier_mask, matched_column = find_student_by_rfid(existing_df, payload.rfid, schema=schema)

    mapped_identity = get_rfid_mapping(payload.rfid)
    if not identifier_mask.any() and mapped_identity:
        identifier_mask, matched_column = find_student_by_identity(
            existing_df,
            student_id=mapped_identity.get("student_id"),
            student_name=mapped_identity.get("student_name"),
            schema=schema,
        )
        if identifier_mask.any():
            matched_column = f"rfid_mapping->{matched_column or 'identity'}"

    if not identifier_mask.any():
        attendance_events_col.insert_one(
            {
                "lecture_id": lecture_id,
                "rfid": normalized_rfid,
                "matched_by": None,
                "source": (payload.source or "rfid").strip(),
                "status": "unmatched",
                "student_name": mapped_identity.get("student_name") if mapped_identity else None,
                "mapped_student_name": mapped_identity.get("student_name") if mapped_identity else None,
                "mapped_student_id": mapped_identity.get("student_id") if mapped_identity else None,
                "created_at": datetime.utcnow(),
            }
        )
        raise HTTPException(
            404,
            "RFID card not found in the current dataset. Add an RFID/card UID column or use student IDs that match the card UID.",
        )

    existing_df = existing_df.copy()
    ensure_numeric_column(existing_df, "rfid_present_count", default=0)
    ensure_numeric_column(existing_df, "rfid_session_count", default=0)
    ensure_text_column(existing_df, "rfid_last_scan_at")
    ensure_text_column(existing_df, "rfid_last_session_id")
    ensure_text_column(existing_df, "rfid_scan_source")

    session = attendance_sessions_col.find_one({"lecture_id": lecture_id})
    if not session:
        attendance_sessions_col.insert_one(
            {
                "lecture_id": lecture_id,
                "source": (payload.source or "rfid").strip(),
                "created_at": datetime.utcnow(),
            }
        )
        existing_df["rfid_session_count"] = pd.to_numeric(
            existing_df["rfid_session_count"], errors="coerce"
        ).fillna(0) + 1

    duplicate_event = attendance_events_col.find_one(
        {"lecture_id": lecture_id, "rfid": normalized_rfid}
    )
    if duplicate_event:
        matched_row = existing_df.loc[identifier_mask].iloc[0]
        student_name = get_student_display_name(matched_row, schema=schema)
        attendance_events_col.insert_one(
            {
                "lecture_id": lecture_id,
                "rfid": normalized_rfid,
                "student_name": student_name,
                "matched_by": matched_column,
                "source": (payload.source or "rfid").strip(),
                "status": "duplicate",
                "created_at": datetime.utcnow(),
            }
        )
        settings = save_attendance_settings(
            last_scan_name=student_name,
            last_scan_rfid=payload.rfid.strip(),
            last_scan_lecture_id=lecture_id,
            last_scan_at=datetime.utcnow().isoformat(),
        )
        return build_scan_response(
            existing_df,
            "rfid-scan",
            f"Duplicate RFID scan ignored for lecture '{lecture_id}'.",
            training_used=False,
            extra={
                "attendance_settings": settings,
                "scan_details": {
                    "rfid": payload.rfid.strip(),
                    "lecture_id": lecture_id,
                    "student_name": student_name,
                    "matched_by": matched_column,
                    "duplicate": True,
                    "rfid_present_count": sanitize_value(matched_row.get("rfid_present_count")),
                    "rfid_session_count": sanitize_value(matched_row.get("rfid_session_count")),
                }
            },
        )

    matched_row = existing_df.loc[identifier_mask].iloc[0]
    student_name = get_student_display_name(matched_row, schema=schema)
    attendance_events_col.insert_one(
        {
            "lecture_id": lecture_id,
            "rfid": normalized_rfid,
            "student_name": student_name,
            "matched_by": matched_column,
            "source": (payload.source or "rfid").strip(),
            "status": "matched",
            "created_at": datetime.utcnow(),
        }
    )

    existing_df.loc[identifier_mask, "rfid_present_count"] = (
        pd.to_numeric(existing_df.loc[identifier_mask, "rfid_present_count"], errors="coerce").fillna(0) + 1
    )
    existing_df.loc[identifier_mask, "rfid_last_scan_at"] = datetime.utcnow().isoformat()
    existing_df.loc[identifier_mask, "rfid_last_session_id"] = lecture_id
    existing_df.loc[identifier_mask, "rfid_scan_source"] = (payload.source or "rfid").strip()

    settings = save_attendance_settings(
        last_scan_name=student_name,
        last_scan_rfid=payload.rfid.strip(),
        last_scan_lecture_id=lecture_id,
        last_scan_at=datetime.utcnow().isoformat(),
    )

    existing_df, attendance_key, effective_total = recalculate_attendance_metrics(existing_df, schema=schema)

    updated_df = clean_dataframe(existing_df)
    updated_df, _ = enrich_dataframe_for_learning(updated_df)
    result = retrain_and_persist_dataset(
        updated_df,
        "rfid-scan",
        f"RFID attendance recorded successfully for lecture '{lecture_id}'.",
    )

    matched_row = updated_df.loc[identifier_mask].iloc[0]
    result["scan_details"] = sanitize_value(
        {
            "rfid": payload.rfid.strip(),
            "lecture_id": lecture_id,
            "student_name": student_name,
            "matched_by": matched_column,
            "duplicate": False,
            "rfid_present_count": matched_row.get("rfid_present_count"),
            "rfid_session_count": matched_row.get("rfid_session_count"),
            "attendance": matched_row.get(attendance_key),
            "effective_total_lectures": effective_total,
        }
    )
    result["attendance_settings"] = sanitize_value({**settings, "effective_total_lectures": effective_total})
    return result


def reset_rfid_attendance(payload: AttendanceResetPayload):
    existing_df = load_existing_students_dataframe()
    if existing_df.empty:
        raise HTTPException(400, "No student result dataset found yet. Upload result data before resetting attendance.")

    existing_df = clean_dataframe(existing_df)
    if existing_df.empty:
        raise HTTPException(400, "Stored student data is empty. Upload result data before resetting attendance.")

    schema = infer_dataset_schema(existing_df)
    lecture_id = get_effective_lecture_id(payload.lecture_id)
    if not lecture_id:
        raise HTTPException(404, "No lecture attendance history found to reset.")

    if not (payload.student_id or payload.student_name):
        updated_settings = save_attendance_settings(
            last_scan_name=None,
            last_scan_rfid=None,
            last_scan_lecture_id=None,
            last_scan_at=None,
            dashboard_reset_at=datetime.utcnow().isoformat(),
        )
        return build_scan_response(
            existing_df,
            "attendance-reset",
            f"Attendance list cleared from the dashboard for lecture '{lecture_id}'. Prediction data was kept unchanged.",
            training_used=False,
            extra={
                "attendance_settings": updated_settings,
                "reset_details": {
                    "lecture_id": lecture_id,
                    "dashboard_only": True,
                },
            },
        )

    lecture_events = list(
        attendance_events_col.find(
            {"lecture_id": lecture_id},
            {
                "_id": 1,
                "student_name": 1,
                "status": 1,
            },
        )
    )

    if payload.student_id or payload.student_name:
        identifier_mask, _ = find_student_by_identity(
            existing_df,
            student_id=payload.student_id,
            student_name=payload.student_name,
            schema=schema,
        )
        if not identifier_mask.any():
            raise HTTPException(404, "Student not found. Provide a matching student ID or student name from the current dataset.")

        matched_row = existing_df.loc[identifier_mask].iloc[0]
        student_name = get_student_display_name(matched_row, schema=schema)
        normalized_name = normalize_token(student_name)
        removable_event_ids = [
            event.get("_id")
            for event in lecture_events
            if normalize_token(event.get("student_name")) == normalized_name
        ]
        result_message = f"Attendance reset successfully for {student_name} in lecture '{lecture_id}'."
    else:
        removable_event_ids = [event.get("_id") for event in lecture_events if event.get("_id") is not None]
        result_message = f"Attendance reset successfully for lecture '{lecture_id}'."

    if not removable_event_ids:
        raise HTTPException(404, f"No attendance scans found for lecture '{lecture_id}'.")

    attendance_events_col.delete_many({"_id": {"$in": removable_event_ids}})

    updated_df = rebuild_rfid_attendance_fields(existing_df, schema=schema)
    updated_df, attendance_key, effective_total = recalculate_attendance_metrics(updated_df, schema=schema)
    updated_df = clean_dataframe(updated_df)
    updated_df, _ = enrich_dataframe_for_learning(updated_df)
    result = retrain_and_persist_dataset(
        updated_df,
        "attendance-reset",
        result_message,
    )

    settings = get_attendance_settings()
    result["attendance_settings"] = sanitize_value({**settings, "effective_total_lectures": effective_total})
    reset_details = {
        "lecture_id": lecture_id,
        "removed_events": len(removable_event_ids),
    }
    if payload.student_id or payload.student_name:
        reset_details["student_name"] = student_name
        reset_details["attendance"] = updated_df.loc[identifier_mask].iloc[0].get(attendance_key)
    result["reset_details"] = sanitize_value(reset_details)
    return result


def persist_analysis(full_df, acc, source_type, dataset_role, training_used, message):
    schema = infer_dataset_schema(full_df)
    students_col.delete_many({})
    student_records = sanitize_records(full_df.to_dict(orient="records"))
    if student_records:
        students_col.insert_many(student_records)

    history_col.insert_one({
        "time": datetime.now(),
        "rows": len(full_df),
        "accuracy": float(acc) if acc is not None else None,
        "source_type": source_type,
        "dataset_role": dataset_role,
        "training_used": training_used,
    })

    return {
        "message": message,
        "accuracy": acc,
        "columns": full_df.columns.tolist(),
        "rows": len(full_df),
        "source_type": source_type,
        "dataset_role": dataset_role,
        "training_used": training_used,
        "schema": sanitize_value(schema),
        "preview_records": sanitize_records(full_df.head(3).to_dict(orient="records")),
        "attendance_settings": get_attendance_settings(),
    }


def analyze_uploaded_file(upload):
    try:
        df, source_type = read_uploaded_dataframe(upload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Upload Error: {str(e)}")

    df = clean_dataframe(df)
    df, schema = enrich_dataframe_for_learning(df)

    if df.empty or df.shape[1] == 0:
        raise HTTPException(400, f"Empty {source_type.upper()}")

    dataset_role = classify_dataset_role(df, schema=schema, filename=upload.filename)
    if dataset_role == DATASET_ROLE_UNSUPPORTED:
        reject_unsupported_dataset(source_type)
    if dataset_role == DATASET_ROLE_ATTENDANCE:
        existing_df = load_existing_students_dataframe()
        if existing_df.empty:
            reject_attendance_only_dataset(source_type)

        existing_df = clean_dataframe(existing_df)
        existing_df, existing_schema = enrich_dataframe_for_learning(existing_df)
        existing_role = classify_dataset_role(existing_df, schema=existing_schema)
        if existing_role != DATASET_ROLE_PERFORMANCE:
            reject_attendance_only_dataset(source_type)

        merged_df = merge_student_datasets(existing_df, df)
        merged_df = clean_dataframe(merged_df)
        merged_df, _ = enrich_dataframe_for_learning(merged_df)
        return retrain_and_persist_dataset(
            merged_df,
            source_type,
            "Attendance data merged successfully with the existing result dataset."
        )

    if dataset_role == DATASET_ROLE_PERFORMANCE:
        full_df = df.copy()
        return retrain_and_persist_dataset(
            full_df,
            source_type,
            "Student performance data accepted."
        )

# ---------------- DATA UPLOAD ----------------

@api_app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    return analyze_uploaded_file(file)

@api_app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    return analyze_uploaded_file(file)

# ---------------- GET DATA ----------------

@api_app.get("/students")
def get_students():

    data = list(students_col.find({}, {"_id": 0}))
    data = sanitize_records(data)

    if not data:
        return {
            "records": [],
            "columns": [],
            "schema": {},
            "attendance_settings": get_attendance_settings(),
            "recent_scans": get_recent_attendance_scans(),
            "lecture_attendance": [],
        }

    schema = infer_dataset_schema(pd.DataFrame(data))
    attendance_df = pd.DataFrame(data)
    _, _, effective_total = recalculate_attendance_metrics(attendance_df, schema=schema)

    return {
        "records": data,
        "columns": list(data[0].keys()),
        "schema": sanitize_value(schema),
        "attendance_settings": sanitize_value({**get_attendance_settings(), "effective_total_lectures": effective_total}),
        "recent_scans": get_recent_attendance_scans(),
        "lecture_attendance": get_current_lecture_status_list(attendance_df, schema=schema),
    }


@api_app.post("/attendance/update")
def update_attendance(payload: AttendanceUpdatePayload):
    if not payload.student_id and not payload.student_name:
        raise HTTPException(400, "Provide either student_id or student_name for the attendance update.")
    return apply_daily_attendance_update(payload)


@api_app.post("/attendance/config")
def configure_attendance(payload: AttendanceConfigPayload):
    return update_total_lectures(payload)


@api_app.post("/attendance/link-rfid")
def assign_rfid_to_student(payload: RFIDLinkPayload):
    if not payload.student_id and not payload.student_name:
        raise HTTPException(400, "Provide either student_id or student_name to link an RFID card.")
    return link_rfid_to_student(payload)


@api_app.post("/attendance")
def record_rfid_attendance(payload: RFIDAttendancePayload):
    return apply_rfid_attendance_scan(payload)


@api_app.post("/attendance/rfid-scan")
def record_rfid_attendance_alias(payload: RFIDAttendancePayload):
    return apply_rfid_attendance_scan(payload)


@api_app.post("/attendance/reset")
def reset_attendance_for_student(payload: AttendanceResetPayload):
    return reset_rfid_attendance(payload)


@api_app.post("/register")
def register(payload: RegisterPayload):
    normalized_email = payload.email.strip().lower()
    ensure_valid_password(payload.password)
    if users_col.find_one({"email": normalized_email}):
        raise HTTPException(400, "Email already registered")

    users_col.insert_one({
        "name": payload.name.strip(),
        "email": normalized_email,
        "password": hashlib.sha256(payload.password.encode()).hexdigest(),
        "created_at": datetime.utcnow()
    })

    return {"message": "Registration successful"}


@api_app.post("/login")
def login(payload: LoginPayload):
    normalized_email = payload.email.strip().lower()
    user = users_col.find_one({"email": normalized_email})
    if not user:
        raise HTTPException(401, "Invalid credentials")

    hashed_password = hashlib.sha256(payload.password.encode()).hexdigest()
    stored_password = user.get("password")
    if hashed_password != stored_password:
        raise HTTPException(401, "Invalid credentials")

    return {
        "message": "Login successful",
        "name": user.get("name"),
        "email": normalized_email
    }


@api_app.post("/admin/login")
def admin_login(payload: AdminLoginPayload):
    return authenticate_admin(payload)


@api_app.post("/forgot-password")
def forgot_password(payload: ForgotPasswordPayload):
    normalized_email = payload.email.strip().lower()
    ensure_valid_password(payload.new_password)
    user = users_col.find_one({"email": normalized_email})
    if not user:
        raise HTTPException(404, "No account found for this email")

    users_col.update_one(
        {"email": normalized_email},
        {
            "$set": {
                "password": hashlib.sha256(payload.new_password.encode()).hexdigest(),
                "updated_at": datetime.utcnow(),
            }
        },
    )

    return {"message": "Password updated successfully"}


@api_app.get("/users")
def get_users():
    users = list(
        users_col.find(
            {},
            {
                "_id": 0,
                "name": 1,
                "email": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        ).sort("created_at", -1)
    )

    return {"users": users}


@api_app.put("/users")
def update_user(payload: AdminUserUpdatePayload):
    original_email = payload.original_email.strip().lower()
    updated_email = payload.email.strip().lower()
    updated_name = payload.name.strip()

    if not updated_name:
        raise HTTPException(400, "Name is required")

    user = users_col.find_one({"email": original_email})
    if not user:
        raise HTTPException(404, "User not found")

    if updated_email != original_email and users_col.find_one({"email": updated_email}):
        raise HTTPException(400, "Another account already uses this email")

    update_fields = {
        "name": updated_name,
        "email": updated_email,
        "updated_at": datetime.utcnow(),
    }

    if payload.password:
        ensure_valid_password(payload.password)
        update_fields["password"] = hashlib.sha256(payload.password.encode()).hexdigest()

    users_col.update_one({"email": original_email}, {"$set": update_fields})

    return {"message": "User updated successfully"}


@api_app.delete("/users/{email:path}")
def delete_user(email: str):
    normalized_email = email.strip().lower()

    if not normalized_email:
        raise HTTPException(400, "Email is required")

    if normalized_email == ADMIN_EMAIL:
        raise HTTPException(400, "Admin account cannot be deleted here")

    result = users_col.delete_one({"email": normalized_email})
    if result.deleted_count == 0:
        raise HTTPException(404, "User not found")

    return {"message": "User deleted successfully"}

# ---------------- HEALTH ----------------

@api_app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# ---------------- FRONTEND ----------------

@app.get("/", include_in_schema=False)
def serve():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/admin/login", include_in_schema=False)
def root_admin_login(payload: AdminLoginPayload):
    return authenticate_admin(payload)


@app.post("/admin/login", include_in_schema=False)
def direct_admin_login(payload: AdminLoginPayload):
    return authenticate_admin(payload)

app.mount("/api", api_app)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
