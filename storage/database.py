import json
import sqlite3
from config import DATABASE, STATUSES


def connect(path=DATABASE):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL)")
    con.commit()
    return con


def save_event(event, path=DATABASE):
    with connect(path) as con:
        con.execute("INSERT OR IGNORE INTO events VALUES (?, ?, ?)", (event["id"], json.dumps(event), STATUSES[0]))


def list_events(path=DATABASE):
    with connect(path) as con:
        rows = con.execute("SELECT payload,status FROM events ORDER BY rowid DESC").fetchall()
    return [dict(json.loads(payload), status=status) for payload,status in rows]


def update_status(event_id, status, path=DATABASE):
    if status not in STATUSES:
        raise ValueError("Invalid status")
    with connect(path) as con:
        result = con.execute("UPDATE events SET status=? WHERE id=?", (status,event_id))
        if result.rowcount != 1:
            raise KeyError("Report not found")
