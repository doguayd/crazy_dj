"""
SQLite storage for song analysis results.
Schema is compatible with Dotify's existing Dotify.db3 extension plan.
"""

import sqlite3
import json
from pathlib import Path
from dataclasses import asdict
from analyzer.audio_analyzer import SongAnalysis


DB_PATH = Path(__file__).parent.parent / "crazy_dj.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS song_analysis (
                path        TEXT PRIMARY KEY,
                bpm         REAL,
                key         TEXT,
                energy      REAL,
                beat_times  TEXT,  -- JSON array
                chroma      TEXT,  -- JSON array
                duration    REAL,
                analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save(analysis: SongAnalysis):
    d = asdict(analysis)
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO song_analysis
                (path, bpm, key, energy, beat_times, chroma, duration)
            VALUES (:path, :bpm, :key, :energy, :beat_times, :chroma, :duration)
        """, {**d, "beat_times": json.dumps(d["beat_times"]),
                    "chroma": json.dumps(d["chroma"])})


def load(file_path: str) -> SongAnalysis | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM song_analysis WHERE path = ?", (file_path,)
        ).fetchone()
    if row is None:
        return None
    return SongAnalysis(
        path=row["path"],
        bpm=row["bpm"],
        key=row["key"],
        energy=row["energy"],
        beat_times=json.loads(row["beat_times"]),
        chroma=json.loads(row["chroma"]),
        duration=row["duration"],
    )


def load_all() -> list[SongAnalysis]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM song_analysis").fetchall()
    results = []
    for row in rows:
        results.append(SongAnalysis(
            path=row["path"],
            bpm=row["bpm"],
            key=row["key"],
            energy=row["energy"],
            beat_times=json.loads(row["beat_times"]),
            chroma=json.loads(row["chroma"]),
            duration=row["duration"],
        ))
    return results
