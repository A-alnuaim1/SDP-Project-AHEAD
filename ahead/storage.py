"""Small local prototype store. Set AHEAD_DB_PATH to a durable, protected volume."""

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ahead.config import BASE, ROLE_DOCTOR, ROLE_PATIENT


def _db_path():
    return Path(os.environ.get("AHEAD_DB_PATH", str(BASE / "ahead.sqlite3")))


@contextmanager
def connection():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys=ON")
        yield db
        db.commit()
    finally:
        db.close()


def password_hash(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password, encoded):
    try:
        salt, digest = encoded.split(":", 1)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000)
        return hmac.compare_digest(actual, bytes.fromhex(digest))
    except (ValueError, AttributeError):
        return False


def init_db():
    with connection() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
                role TEXT NOT NULL, name TEXT NOT NULL, institution TEXT NOT NULL DEFAULT '',
                preferences_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
                cohort TEXT NOT NULL, disease TEXT NOT NULL, patient_id TEXT NOT NULL,
                source_row INTEGER NOT NULL, values_json TEXT NOT NULL, dob TEXT,
                issues_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Incomplete',
                note TEXT NOT NULL DEFAULT '', model TEXT, score REAL, threshold REAL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS records_owner ON records(owner_id, disease, cohort);
            CREATE TABLE IF NOT EXISTS screenings (
                id INTEGER PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
                disease TEXT NOT NULL, model TEXT NOT NULL, score REAL NOT NULL,
                prediction INTEGER NOT NULL, threshold REAL NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS screenings_owner ON screenings(owner_id, created_at);
        """)
        if "preferences_json" not in {row["name"] for row in db.execute("PRAGMA table_info(users)")}:
            db.execute("ALTER TABLE users ADD COLUMN preferences_json TEXT NOT NULL DEFAULT '{}'")
        screening_columns = {row["name"] for row in db.execute("PRAGMA table_info(screenings)")}
        for column, definition in (("values_json", "TEXT NOT NULL DEFAULT '{}'"),
                                   ("source", "TEXT NOT NULL DEFAULT 'legacy'"),
                                   ("previous_id", "INTEGER")):
            if column not in screening_columns:
                db.execute(f"ALTER TABLE screenings ADD COLUMN {column} {definition}")
    # Demo accounts are explicitly enabled for classroom use only.
    if os.environ.get("AHEAD_ENABLE_DEMO_ACCOUNTS") == "1":
        create_user("user@ahead.demo", "user123", ROLE_PATIENT, "Sarah Ahmed", demo=True)
        create_user("doctor@ahead.demo", "doctor123", ROLE_DOCTOR, "Dr. Sarah Ahmed", "AHEAD Medical Center", demo=True)
    doctor_email = os.environ.get("AHEAD_DOCTOR_EMAIL")
    doctor_password = os.environ.get("AHEAD_DOCTOR_PASSWORD")
    if doctor_email and doctor_password:
        try:
            create_user(doctor_email, doctor_password, ROLE_DOCTOR, "Doctor")
        except ValueError:
            # A misconfigured secret must not take the whole site down (including sign-in).
            logging.getLogger(__name__).warning(
                "AHEAD_DOCTOR_PASSWORD is shorter than 12 characters; the doctor account was not created.")


def has_doctor_account():
    with connection() as db:
        return db.execute("SELECT 1 FROM users WHERE role=? LIMIT 1", (ROLE_DOCTOR,)).fetchone() is not None


def ephemeral_storage():
    """True when the database sits on a disk that is wiped on restart (Streamlit Community Cloud).

    Community Cloud clones the repository under /mount/src; AHEAD_EPHEMERAL_STORAGE=1/0 overrides the guess.
    """
    override = os.environ.get("AHEAD_EPHEMERAL_STORAGE")
    if override in ("0", "1"):
        return override == "1"
    return str(_db_path().resolve()).startswith("/mount/src/")


def create_user(email, password, role, name, institution="", demo=False):
    if role not in (ROLE_PATIENT, ROLE_DOCTOR) or (len(password) < 12 and not demo):
        raise ValueError("Use a password of at least 12 characters.")
    with connection() as db:
        if db.execute("SELECT 1 FROM users WHERE email=?", (email.strip().lower(),)).fetchone():
            return False
    with connection() as db:
        try:
            db.execute("INSERT INTO users(email,password_hash,role,name,institution) VALUES(?,?,?,?,?)",
                       (email.strip().lower(), password_hash(password), role, name.strip(), institution.strip()))
        except sqlite3.IntegrityError:
            return False
    return True


def authenticate(email, password, role):
    with connection() as db:
        row = db.execute("SELECT * FROM users WHERE email=? AND role=?", (email.strip().lower(), role)).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    result = {k: row[k] for k in ("id", "email", "role", "name", "institution")}
    result["preferences"] = json.loads(row["preferences_json"] or "{}")
    return result


def update_preferences(user_id, preferences):
    with connection() as db:
        db.execute("UPDATE users SET preferences_json=? WHERE id=?",
                   (json.dumps(preferences), user_id))


def update_profile(user_id, name, email, institution):
    with connection() as db:
        db.execute("UPDATE users SET name=?,email=?,institution=? WHERE id=?",
                   (name.strip(), email.strip().lower(), institution.strip(), user_id))


def change_password(user_id, current, new):
    with connection() as db:
        row = db.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
        if not row or not verify_password(current, row["password_hash"]):
            return False
        db.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash(new), user_id))
    return True


def insert_records(owner_id, cohort, disease, rows):
    now = datetime.now(timezone.utc).isoformat()
    with connection() as db:
        for index, row in enumerate(rows, 1):
            db.execute("""INSERT INTO records(owner_id,cohort,disease,patient_id,source_row,values_json,dob,issues_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                       (owner_id, cohort, disease, row["patient_id"], index, json.dumps(row["values"]),
                        row.get("dob"), json.dumps(row["issues"]), now, now))


def list_records(owner_id, disease=None):
    query = "SELECT * FROM records WHERE owner_id=?"
    args = [owner_id]
    if disease:
        query += " AND disease=?"
        args.append(disease)
    query += " ORDER BY updated_at DESC, id DESC"
    with connection() as db:
        return [dict(r) for r in db.execute(query, args).fetchall()]


def update_record(owner_id, record_id, **fields):
    allowed = {"values_json", "dob", "issues_json", "status", "note", "model", "score", "threshold"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    assignments = ",".join(f"{key}=?" for key in fields)
    with connection() as db:
        db.execute(f"UPDATE records SET {assignments} WHERE id=? AND owner_id=?",
                   [*fields.values(), record_id, owner_id])


def delete_records(owner_id):
    with connection() as db:
        db.execute("DELETE FROM record_predictions WHERE owner_id=?", (owner_id,))
        db.execute("DELETE FROM records WHERE owner_id=?", (owner_id,))


def save_screening(owner_id, disease, model, score, prediction, threshold, values=None, source="patient", previous_id=None):
    with connection() as db:
        cursor = db.execute("""INSERT INTO screenings(owner_id,disease,model,score,prediction,threshold,created_at,values_json,source,previous_id)
                               VALUES(?,?,?,?,?,?,?,?,?,?)""",
                            (owner_id, disease, model, score, prediction, threshold,
                             datetime.now(timezone.utc).isoformat(), json.dumps(values or {}), source, previous_id))
        return cursor.lastrowid


def load_screenings(owner_id):
    with connection() as db:
        return [dict(r) for r in db.execute(
            "SELECT * FROM screenings WHERE owner_id=? ORDER BY created_at DESC,id DESC LIMIT 500", (owner_id,)).fetchall()]


def delete_screenings(owner_id):
    with connection() as db:
        db.execute("DELETE FROM screenings WHERE owner_id=?", (owner_id,))
