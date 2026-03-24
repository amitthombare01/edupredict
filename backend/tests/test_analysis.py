"""Tests for student performance analysis utilities."""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import analysis
import app as backend_app


def test_load_student_data(tmp_path):
    sample = pd.DataFrame({"student_id": [1, 2], "score": [85, 90]})
    path = tmp_path / "student_performance.csv"
    sample.to_csv(path, index=False)

    df = analysis.load_student_data(str(path))
    assert list(df.columns) == ["student_id", "score"]
    assert len(df) == 2


def test_load_student_data_missing_file(tmp_path):
    path = tmp_path / "missing.csv"
    df = analysis.load_student_data(str(path))
    assert df.empty
    assert list(df.columns) == analysis.STANDARD_COLUMNS


def test_extract_table_rows_from_pdf_text_pipe_delimited():
    text = "\n".join([
        "Name | Attendance | Final Score | Result",
        "Asha | 92 | 88 | Pass",
        "Ravi | 68 | 51 | Fail",
    ])

    rows = backend_app.extract_table_rows_from_pdf_text(text)

    assert rows[0] == ["Name", "Attendance", "Final Score", "Result"]
    assert rows[1] == ["Asha", "92", "88", "Pass"]
    assert rows[2] == ["Ravi", "68", "51", "Fail"]


def test_table_from_text_rows_builds_dataframe():
    rows = [
        ["Name", "Score", "Result"],
        ["Asha", "88", "Pass"],
        ["Ravi", "51", "Fail"],
    ]

    df = backend_app.table_from_text_rows(rows)

    assert list(df.columns) == ["Name", "Score", "Result"]
    assert df.shape == (2, 3)


def test_classify_dataset_role_marks_attendance_only_data():
    df = pd.DataFrame(
        {
            "Student Name": ["Asha", "Ravi"],
            "Roll No": [1, 2],
            "Attendance %": [92, 68],
        }
    )

    cleaned = backend_app.clean_dataframe(df)
    cleaned, schema = backend_app.enrich_dataframe_for_learning(cleaned)

    assert backend_app.classify_dataset_role(cleaned, schema=schema) == backend_app.DATASET_ROLE_ATTENDANCE


def test_classify_dataset_role_prefers_attendance_over_multi_numeric_sheet():
    df = pd.DataFrame(
        {
            "Student Name": ["Asha", "Ravi"],
            "Roll No": [1, 2],
            "Attendance %": [92, 68],
            "Days Present": [210, 175],
            "Days Absent": [8, 27],
        }
    )

    cleaned = backend_app.clean_dataframe(df)
    cleaned, schema = backend_app.enrich_dataframe_for_learning(cleaned)

    assert (
        backend_app.classify_dataset_role(
            cleaned,
            schema=schema,
            filename="student_attendance.pdf",
        )
        == backend_app.DATASET_ROLE_ATTENDANCE
    )


def test_classify_dataset_role_treats_lecture_totals_as_attendance():
    df = pd.DataFrame(
        {
            "Student Name": ["Asha", "Ravi"],
            "Roll No": [1, 2],
            "Total No. of Lectures": [102, 102],
            "Lectures Attended": [100, 12],
            "Attendance %": [98, 12],
        }
    )

    cleaned = backend_app.clean_dataframe(df)
    cleaned, schema = backend_app.enrich_dataframe_for_learning(cleaned)

    assert (
        backend_app.classify_dataset_role(
            cleaned,
            schema=schema,
            filename="attendance.pdf",
        )
        == backend_app.DATASET_ROLE_ATTENDANCE
    )


def test_classify_dataset_role_accepts_combined_result_and_attendance_data():
    df = pd.DataFrame(
        {
            "Student Name": ["Asha", "Ravi"],
            "Roll No": [1, 2],
            "Attendance %": [98, 12],
            "Final Score": [88, 42],
            "Result": ["Pass", "Fail"],
        }
    )

    cleaned = backend_app.clean_dataframe(df)
    cleaned, schema = backend_app.enrich_dataframe_for_learning(cleaned)

    assert (
        backend_app.classify_dataset_role(
            cleaned,
            schema=schema,
            filename="combined_result_attendance.pdf",
        )
        == backend_app.DATASET_ROLE_PERFORMANCE
    )


def test_select_feature_columns_ignores_attendance():
    df = pd.DataFrame(
        {
            "student_name": ["Asha", "Ravi", "Mina"],
            "math": [80, 55, 72],
            "attendance": [95, 62, 81],
            "result": ["Pass", "Fail", "Pass"],
        }
    )

    selected = backend_app.select_feature_columns(df, target="result")

    assert "attendance" not in selected.columns
    assert "math" in selected.columns


def test_merge_student_datasets_adds_attendance_without_replacing_scores():
    existing_df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "student_name": ["Asha", "Ravi"],
            "math": [82, 58],
            "percentage": [84, 57],
            "result": ["Pass", "Fail"],
        }
    )
    incoming_df = pd.DataFrame(
        {
            "roll_no": [1, 2],
            "student_name": ["Asha", "Ravi"],
            "attendance_percent": [93, 71],
        }
    )

    merged = backend_app.merge_student_datasets(
        backend_app.clean_dataframe(existing_df),
        backend_app.clean_dataframe(incoming_df),
    )

    assert "attendance" in merged.columns
    assert list(merged["attendance"]) == [93, 71]
    assert list(merged["percentage"]) == [84, 57]


class _FakeCollection:
    def __init__(self):
        self.docs = []

    def _matches(self, doc, query):
        for key, value in query.items():
            if isinstance(value, dict):
                if "$in" in value:
                    if doc.get(key) not in value["$in"]:
                        return False
                else:
                    return False
            elif doc.get(key) != value:
                return False
        return True

    def find_one(self, query=None, projection=None, sort=None):
        query = query or {}
        docs = [doc for doc in self.docs if self._matches(doc, query)]
        if sort:
            for field, direction in reversed(sort):
                docs.sort(key=lambda item: item.get(field), reverse=direction < 0)
        doc = docs[0] if docs else None
        if doc is None:
            return None
        if projection:
            return {key: value for key, value in doc.items() if key in projection and projection[key]}
        return dict(doc)

    def find(self, query=None, projection=None):
        query = query or {}
        docs = [dict(doc) for doc in self.docs if self._matches(doc, query)]
        if projection:
            docs = [
                {key: value for key, value in doc.items() if key in projection and projection[key]}
                for doc in docs
            ]
        return docs

    def count_documents(self, query):
        return len([doc for doc in self.docs if self._matches(doc, query)])

    def insert_one(self, document):
        payload = dict(document)
        payload.setdefault("_id", len(self.docs) + 1)
        self.docs.append(payload)
        return {"inserted_id": len(self.docs)}

    def delete_many(self, query):
        before = len(self.docs)
        self.docs = [doc for doc in self.docs if not self._matches(doc, query)]
        return type("DeleteResult", (), {"deleted_count": before - len(self.docs)})()


def test_apply_rfid_attendance_scan_counts_one_daily_lecture(monkeypatch):
    stored_df = pd.DataFrame(
        {
            "student_id": ["AA11", "BB22"],
            "student_name": ["Asha", "Ravi"],
            "result": ["Pass", "Fail"],
        }
    )
    saved = {}

    monkeypatch.setattr(backend_app, "attendance_events_col", _FakeCollection())
    monkeypatch.setattr(backend_app, "attendance_sessions_col", _FakeCollection())
    monkeypatch.setattr(backend_app, "load_existing_students_dataframe", lambda: stored_df.copy())

    def fake_retrain(full_df, source_type, message_prefix):
        saved["df"] = full_df.copy()
        return {
            "message": message_prefix,
            "rows": len(full_df),
            "columns": full_df.columns.tolist(),
            "source_type": source_type,
        }

    monkeypatch.setattr(backend_app, "retrain_and_persist_dataset", fake_retrain)

    payload = backend_app.RFIDAttendancePayload(rfid="AA11", lecture_id="2026-03-23")
    result = backend_app.apply_rfid_attendance_scan(payload)

    updated = saved["df"]
    assert result["source_type"] == "rfid-scan"
    assert list(updated["rfid_present_count"]) == [1, 0]
    assert list(updated["rfid_session_count"]) == [1, 1]
    assert list(updated["attendance"]) == [100.0, 0.0]


def test_apply_rfid_attendance_scan_ignores_duplicate_for_same_lecture(monkeypatch):
    stored_df = pd.DataFrame(
        {
            "student_id": ["AA11", "BB22"],
            "student_name": ["Asha", "Ravi"],
            "rfid_present_count": [1, 0],
            "rfid_session_count": [1, 1],
            "attendance": [100.0, 0.0],
            "result": ["Pass", "Fail"],
        }
    )
    events = _FakeCollection()
    events.insert_one({"lecture_id": "2026-03-23", "rfid": "aa11"})
    sessions = _FakeCollection()
    sessions.insert_one({"lecture_id": "2026-03-23"})

    monkeypatch.setattr(backend_app, "attendance_events_col", events)
    monkeypatch.setattr(backend_app, "attendance_sessions_col", sessions)
    monkeypatch.setattr(backend_app, "load_existing_students_dataframe", lambda: stored_df.copy())

    retrain_called = {"value": False}

    def fake_retrain(full_df, source_type, message_prefix):
        retrain_called["value"] = True
        return {}

    monkeypatch.setattr(backend_app, "retrain_and_persist_dataset", fake_retrain)

    payload = backend_app.RFIDAttendancePayload(rfid="AA11", lecture_id="2026-03-23")
    result = backend_app.apply_rfid_attendance_scan(payload)

    assert retrain_called["value"] is False
    assert result["scan_details"]["duplicate"] is True
    assert "Duplicate RFID scan ignored" in result["message"]


def test_apply_rfid_attendance_scan_handles_existing_float_metadata_columns(monkeypatch):
    stored_df = pd.DataFrame(
        {
            "student_id": [54.0, 55.0],
            "student_name": ["PAWAR PRATIKSHA SHIVAJI", "Ravi"],
            "rfid_present_count": [0.0, 0.0],
            "rfid_session_count": [1.0, 1.0],
            "rfid_last_scan_at": [float("nan"), float("nan")],
            "rfid_last_session_id": [float("nan"), float("nan")],
            "rfid_scan_source": [float("nan"), float("nan")],
            "result": ["Pass", "Fail"],
        }
    )
    saved = {}
    events = _FakeCollection()
    sessions = _FakeCollection()
    sessions.insert_one({"lecture_id": "2026-03-23"})

    monkeypatch.setattr(backend_app, "attendance_events_col", events)
    monkeypatch.setattr(backend_app, "attendance_sessions_col", sessions)
    monkeypatch.setattr(backend_app, "load_existing_students_dataframe", lambda: stored_df.copy())
    monkeypatch.setattr(
        backend_app,
        "get_rfid_mapping",
        lambda rfid: {
            "student_id": 54.0,
            "student_name": "PAWAR PRATIKSHA SHIVAJI",
        },
    )

    def fake_retrain(full_df, source_type, message_prefix):
        saved["df"] = full_df.copy()
        return {
            "message": message_prefix,
            "rows": len(full_df),
            "columns": full_df.columns.tolist(),
            "source_type": source_type,
        }

    monkeypatch.setattr(backend_app, "retrain_and_persist_dataset", fake_retrain)

    payload = backend_app.RFIDAttendancePayload(
        rfid="09F96905",
        lecture_id="2026-03-23",
        source="esp8266-rfid",
    )
    result = backend_app.apply_rfid_attendance_scan(payload)

    updated = saved["df"]
    assert result["source_type"] == "rfid-scan"
    assert updated.loc[0, "rfid_last_session_id"] == "2026-03-23"
    assert updated.loc[0, "rfid_scan_source"] == "esp8266-rfid"
    assert isinstance(updated.loc[0, "rfid_last_scan_at"], str)


def test_reset_rfid_attendance_removes_student_from_lecture_and_rebuilds_counts(monkeypatch):
    stored_df = pd.DataFrame(
        {
            "student_id": [1.0, 2.0],
            "student_name": ["THOMBARE AMIT TANAJI", "PAWAR PRATIKSHA SHIVAJI"],
            "rfid_present_count": [2.0, 1.0],
            "rfid_session_count": [2.0, 2.0],
            "attendance": [100.0, 50.0],
            "result": ["Pass", "Pass"],
        }
    )
    saved = {}
    events = _FakeCollection()
    events.insert_one(
        {
            "lecture_id": "2026-03-24-16",
            "student_name": "THOMBARE AMIT TANAJI",
            "status": "matched",
            "created_at": backend_app.datetime(2026, 3, 24, 16, 5, 19),
            "source": "esp8266-rfid",
        }
    )
    events.insert_one(
        {
            "lecture_id": "2026-03-24-15",
            "student_name": "THOMBARE AMIT TANAJI",
            "status": "matched",
            "created_at": backend_app.datetime(2026, 3, 24, 15, 58, 55),
            "source": "esp8266-rfid",
        }
    )
    events.insert_one(
        {
            "lecture_id": "2026-03-24-16",
            "student_name": "PAWAR PRATIKSHA SHIVAJI",
            "status": "matched",
            "created_at": backend_app.datetime(2026, 3, 24, 15, 57, 19),
            "source": "esp8266-rfid",
        }
    )
    sessions = _FakeCollection()
    sessions.insert_one({"lecture_id": "2026-03-24-15"})
    sessions.insert_one({"lecture_id": "2026-03-24-16"})

    monkeypatch.setattr(backend_app, "attendance_events_col", events)
    monkeypatch.setattr(backend_app, "attendance_sessions_col", sessions)
    monkeypatch.setattr(backend_app, "load_existing_students_dataframe", lambda: stored_df.copy())
    monkeypatch.setattr(
        backend_app,
        "get_attendance_settings",
        lambda: {"last_scan_lecture_id": "2026-03-24-16", "total_lectures": 2},
    )

    def fake_retrain(full_df, source_type, message_prefix):
        saved["df"] = full_df.copy()
        return {
            "message": message_prefix,
            "rows": len(full_df),
            "columns": full_df.columns.tolist(),
            "source_type": source_type,
        }

    monkeypatch.setattr(backend_app, "retrain_and_persist_dataset", fake_retrain)

    payload = backend_app.AttendanceResetPayload(student_name="THOMBARE AMIT TANAJI")
    result = backend_app.reset_rfid_attendance(payload)

    updated = saved["df"]
    assert result["source_type"] == "attendance-reset"
    assert "Attendance reset successfully" in result["message"]
    assert updated.loc[0, "rfid_present_count"] == 1
    assert updated.loc[0, "rfid_session_count"] == 2
    assert updated.loc[0, "attendance"] == 50.0
    assert len(events.docs) == 2
