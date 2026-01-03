from __future__ import annotations
import os
import pandas as pd
import numpy as np
import hashlib
import sqlite3

def extract_sensor_data(file_path: str, days: int = 7) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return pd.DataFrame(columns=all)
    try:
        df = pd.read_csv(file_path, sep=";")
    except Exception:
        df = pd.read_csv(file_path)

    if "timestamp" not in df.columns:
        print("[ERROR] Missing required column: 'timestamp'")
        return pd.DataFrame(columns=all)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return pd.DataFrame(columns=all)

    max_ts = df["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].copy()
    return df.reset_index(drop=True)

df = extract_sensor_data(r"C:\Users\Dell\Desktop\ETL pipeline\sensor_data.csv", days=7)
#df.to_csv("sensor_last7days.csv", index=False, sep=";") 

def extract_quality_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return pd.DataFrame()
    df = None
    last_err = None
    for enc in ("utf-8", "ISO-8859-1"):
        try:
            df = pd.read_csv(file_path, encoding=enc, sep=";")
            break
        except Exception as e:
            last_err = e
    if df is None:
        print(f"[ERROR] Could not read file with UTF-8 or ISO-8859-1: {last_err}")
        return pd.DataFrame()

    # Create defect_flag (0 = no fault, 1 = fault present)
    if "Fault Label" in df.columns:
        df["defect_flag"] = (df["Fault Label"] != 0).astype("int64")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Timestamp"])
    return df.reset_index(drop=True)

qdf = extract_quality_data(r"C:\Users\Dell\Desktop\ETL pipeline\quality_data.csv")

# Task 2.1 — CLEAN 
def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Candidate sensor cols
    sensor_cols = [c for c in ["temperature", "pressure", "vibration", "power"] if c in df.columns]
    if not sensor_cols:
        df["data_quality"] = "good"
        return df.reset_index(drop=True)

    # Replace NULL-like + error codes, convert to numeric
    for c in sensor_cols:
        df[c] = df[c].replace({"NULL": np.nan, "null": np.nan, "": np.nan})
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([-999, -1], np.nan)

    if "temperature" in df.columns:
        df.loc[(df["temperature"] < 0) | (df["temperature"] > 150), "temperature"] = np.nan
    if "pressure" in df.columns:
        df.loc[(df["pressure"] < 0) | (df["pressure"] > 10), "pressure"] = np.nan
    if "vibration" in df.columns:
        df.loc[(df["vibration"] < 0) | (df["vibration"] > 100), "vibration"] = np.nan

    # Sort for forward fill (if timestamp exists)
    sort_cols = [c for c in ["machine_id", "timestamp"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    # Track NaN BEFORE fill (per col)
    before_nan_any = df[sensor_cols].isna().any(axis=1)

    # Forward fill per machine if possible
    if "machine_id" in df.columns:
        df[sensor_cols] = df.groupby("machine_id", dropna=False)[sensor_cols].ffill()
    else:
        df[sensor_cols] = df[sensor_cols].ffill()
    # Track NaN AFTER fill
    after_nan_any = df[sensor_cols].isna().any(axis=1)

    df["data_quality"] = "good"
    df.loc[before_nan_any & ~after_nan_any, "data_quality"] = "estimated"
    df.loc[after_nan_any, "data_quality"] = "invalid"
    return df.reset_index(drop=True)

# Task 2.2 — STANDARDIZE (for both sensor + quality)
def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    ) 
    for col in ["line_id", "machine_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # If quality has fault_label -> defect_flag (0=no fault, 1=fault)
    if "fault_label" in df.columns and "defect_flag" not in df.columns:
        fl = pd.to_numeric(df["fault_label"], errors="coerce").fillna(0)
        df["defect_flag"] = (fl != 0).astype("int64")

    # record_id (stable hash). Prefer timestamp+machine_id, else timestamp only, else index only.
    keys = []
    for k in ["timestamp", "machine_id"]:
        if k in df.columns:
            keys.append(k)

    if keys:
        base = df[keys].astype(str).agg("|".join, axis=1) + "|" + df.index.astype(str)
    else:
        base = df.index.astype(str)

    df["record_id"] = base.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())

    return df.reset_index(drop=True)

# Task 2.3 — left join 
def join_sensor_quality(sensor_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    if sensor_df is None or sensor_df.empty:
        return pd.DataFrame()
    if quality_df is None:
        quality_df = pd.DataFrame()

    s = sensor_df.copy()
    q = quality_df.copy()

    # Join keys
    join_keys = [k for k in ["timestamp", "line_id", "machine_id"] if k in s.columns and k in q.columns]
    if not join_keys:
        join_keys = [k for k in ["timestamp", "machine_id"] if k in s.columns and k in q.columns]

    if not join_keys:
        s["quality_status"] = "not_checked"
        return s.reset_index(drop=True)

    # Avoid row multiplication: one quality row per key
    q = q.sort_values(join_keys).drop_duplicates(subset=join_keys, keep="last")

    bring_cols = [c for c in ["defect_flag", "fault_label", "record_id"] if c in q.columns and c not in join_keys]
    merged = s.merge(q[join_keys + bring_cols], on=join_keys, how="left", suffixes=("", "_q"))

    if "defect_flag" in merged.columns:
        merged["quality_status"] = np.where(
            merged["defect_flag"].isna(), "not_checked",
            np.where(pd.to_numeric(merged["defect_flag"], errors="coerce").fillna(0) == 1, "failed", "passed")
        )
    else:
        merged["quality_status"] = "not_checked"
    return merged.reset_index(drop=True)

# Task 2.4 — HOURLY SUMMARY
def hourly_summaries(joined_df: pd.DataFrame) -> pd.DataFrame:
    if joined_df is None or joined_df.empty:
        return pd.DataFrame()

    df = joined_df.copy()

    df["hour"] = df["timestamp"].dt.floor("H")

    # ensure grouping keys (hour,line_id,machine_id) exist
    if "line_id" not in df.columns:
        df["line_id"] = "unknown"
    if "machine_id" not in df.columns:
        df["machine_id"] = "unknown"

    group_keys = ["hour", "line_id", "machine_id"]

    # sensor columns
    sensor_cols = [c for c in ["temperature", "pressure", "vibration", "power"] if c in df.columns]

    agg = {}
    for c in sensor_cols:
        agg[c] = ["mean", "min", "max", "std"]
        
    df["checked_flag"] = (df.get("quality_status", "not_checked") != "not_checked").astype(int)
    if "defect_flag" in df.columns:
        df["defects"] = (pd.to_numeric(df["defect_flag"], errors="coerce").fillna(0) == 1).astype(int)
    else:
        df["defects"] = 0

    base = df.groupby(group_keys).agg(agg)
    base.columns = [f"{col}_{stat}".replace("mean", "avg") for col, stat in base.columns]
    base = base.reset_index()

    counts = df.groupby(group_keys).agg(
        total_checks=("checked_flag", "sum"),
        defect_count=("defects", "sum"),
    ).reset_index()

    out = base.merge(counts, on=group_keys, how="left")
    out["defect_rate"] = np.where(
        out["total_checks"] > 0,
        (out["defect_count"] / out["total_checks"]) * 100.0,
        0.0
    )
    return out.reset_index(drop=True)

# Phase 3 — LOAD
def init_db(db_path: str = "production.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        line_id TEXT,
        machine_id TEXT,
        temperature REAL,
        pressure REAL,
        vibration REAL,
        power REAL,
        data_quality TEXT
    );
    CREATE TABLE IF NOT EXISTS quality_checks (
        check_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        line_id TEXT,
        machine_id TEXT,
        result TEXT,
        defect_type TEXT
    );
    CREATE TABLE IF NOT EXISTS hourly_summary (
        summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hour DATETIME,
        line_id TEXT,
        machine_id TEXT,
        avg_temperature REAL,
        min_temperature REAL,
        max_temperature REAL,
        avg_pressure REAL,
        avg_vibration REAL,
        total_checks INTEGER,
        defect_count INTEGER,
        defect_rate REAL
    );
    """)
    conn.commit()
    return conn

# Task 3.3
def load_table(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    columns: list[str],
    datetime_cols: list[str] | None = None,
    insert_sql: str | None = None
) -> None:
    if df is None or df.empty:
        print(f"[WARN] {table}: no data to load.")
        return
    d = df.copy()
    # Ensure columns exist
    for c in columns:
        if c not in d.columns:
            d[c] = np.nan
    # Convert datetimes to SQLite-friendly string
    if datetime_cols:
        for c in datetime_cols:
            d[c] = pd.to_datetime(d[c], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    # Default INSERT
    if insert_sql is None:
        placeholders = ",".join(["?"] * len(columns))
        cols_sql = ",".join(columns)
        insert_sql = f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})"

    rows = d[columns].values.tolist()
    conn.cursor().executemany(insert_sql, rows)
    conn.commit()
    print(f"[OK] Loaded {len(rows)} rows into {table}.")

# Build table-specific frames (small, clear)
def prepare_quality_for_db(qdf: pd.DataFrame) -> pd.DataFrame:
    if qdf is None or qdf.empty:
        return pd.DataFrame()
    d = qdf.copy()

    # Ensure defect_flag exists if fault_label exists
    if "defect_flag" not in d.columns and "fault_label" in d.columns:
        fl = pd.to_numeric(d["fault_label"], errors="coerce").fillna(0)
        d["defect_flag"] = (fl != 0).astype("int64")
    # result based on defect_flag
    d["result"] = np.where(pd.to_numeric(d.get("defect_flag", 0), errors="coerce").fillna(0) == 1, "fail", "pass")

    # defect_type based on fault_label
    fault_map = {0: 'No Fault', 1: "Bearing Fault", 2: "Overheating"}
    fl_int = pd.to_numeric(d.get("fault_label", 0), errors="coerce").fillna(0).astype(int)
    d["defect_type"] = fl_int.map(fault_map).where(fl_int != 0, "No Fault")
    return d

def prepare_summary_for_db(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    d = summary.copy()
    rename_map = {
        "temperature_avg": "avg_temperature",
        "temperature_min": "min_temperature",
        "temperature_max": "max_temperature",
        "pressure_avg": "avg_pressure",
        "vibration_avg": "avg_vibration",
    }
    d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})
    return d

# Run fonction 
def run_load_pipeline(df_sensor: pd.DataFrame, df_quality: pd.DataFrame, db_path: str = "production.db") -> None:
    sdf = clean_sensor_data(standardize_data(df_sensor))
    qdf2 = clean_sensor_data(standardize_data(df_quality))

    joined = join_sensor_quality(sdf, qdf2)
    summary = hourly_summaries(joined)

    conn = init_db(db_path)
    # 1) sensor_readings
    load_table(
        conn,
        "sensor_readings",
        sdf,
        columns=["timestamp", "line_id", "machine_id", "temperature", "pressure", "vibration", "power", "data_quality"],
        datetime_cols=["timestamp"],
        insert_sql="""
        INSERT OR REPLACE INTO sensor_readings
        (timestamp, line_id, machine_id, temperature, pressure, vibration, power, data_quality)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
    )
    # 2) quality_checks
    qdb = prepare_quality_for_db(qdf2)
    load_table(
        conn,
        "quality_checks",
        qdb,
        columns=["timestamp", "line_id", "machine_id", "result", "defect_type"],
        datetime_cols=["timestamp"]
    )
    # 3) hourly_summary
    sdb = prepare_summary_for_db(summary)
    load_table(
        conn,
        "hourly_summary",
        sdb,
        columns=["hour", "line_id", "machine_id", "avg_temperature", "min_temperature", "max_temperature",
                 "avg_pressure", "avg_vibration", "total_checks", "defect_count", "defect_rate"],
        datetime_cols=["hour"]
    )
    conn.close()
    print(f"[DONE] Database ready: {db_path}")

run_load_pipeline(df, qdf, db_path=r"C:\Users\Dell\Desktop\ETL pipeline\production.db") 