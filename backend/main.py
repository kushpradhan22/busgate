"""
BusGate — Real-time Bus Boarding & Deboarding Detection System
Backend: FastAPI + WebSocket + YOLOv8 + ByteTrack + SQLite
"""

import asyncio
import base64
import json
import sqlite3
import time
import uuid
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Optional YOLO import (graceful fallback to motion detection) ──────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — falling back to motion detection")

app = FastAPI(title="BusGate API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path("busgate.db")

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id          TEXT PRIMARY KEY,
            bus_id      TEXT NOT NULL,
            route       TEXT NOT NULL,
            stop_name   TEXT NOT NULL,
            event_type  TEXT NOT NULL,   -- BOARD | DEBOARD
            count       INTEGER DEFAULT 1,
            timestamp   TEXT NOT NULL,
            hour        INTEGER NOT NULL,
            date        TEXT NOT NULL,
            confidence  REAL DEFAULT 1.0,
            track_id    INTEGER,
            frame_b64   TEXT
        );
        CREATE TABLE IF NOT EXISTS buses (
            bus_id      TEXT PRIMARY KEY,
            route       TEXT NOT NULL,
            capacity    INTEGER DEFAULT 80,
            occupancy   INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'active',
            current_stop TEXT DEFAULT 'Depot',
            last_seen   TEXT
        );
        CREATE TABLE IF NOT EXISTS stops (
            stop_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT UNIQUE NOT NULL,
            lat       REAL,
            lon       REAL
        );
        INSERT OR IGNORE INTO buses VALUES ('B-101','Route 12A',80,0,'active','Central Station',NULL);
        INSERT OR IGNORE INTO buses VALUES ('B-102','Route 7B', 80,0,'active','Market Square',NULL);
        INSERT OR IGNORE INTO buses VALUES ('B-103','Route 3C', 60,0,'active','University',NULL);
        INSERT OR IGNORE INTO buses VALUES ('B-104','Route 9D', 80,0,'active','Hospital',NULL);
        INSERT OR IGNORE INTO stops(name,lat,lon) VALUES
            ('Central Station',20.2961,85.8245),
            ('Market Square',20.3012,85.8178),
            ('University',20.2875,85.8301),
            ('Hospital',20.2934,85.8412),
            ('Park & Ride',20.3089,85.8134),
            ('Airport T1',20.2543,85.8178);
        """)

init_db()

# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class EventIn(BaseModel):
    bus_id: str
    route: str
    stop_name: str
    event_type: str
    count: int = 1
    confidence: float = 1.0
    track_id: Optional[int] = None

class BusUpdate(BaseModel):
    current_stop: Optional[str] = None
    status: Optional[str] = None

# ══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# ══════════════════════════════════════════════════════════════════════════════
#  PERSON TRACKER  (centroid + line-crossing)
# ══════════════════════════════════════════════════════════════════════════════

class PersonTracker:
    def __init__(self, line_ratio: float = 0.5):
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1
        self.line_ratio = line_ratio
        self.board_count = 0
        self.deboard_count = 0

    def update(self, detections: list, frame_h: int) -> list:
        """detections: list of (cx, cy, conf) tuples"""
        line_y = int(frame_h * self.line_ratio)
        events = []

        matched_ids = set()
        for cx, cy, conf in detections:
            best_id, best_dist = None, 80
            for tid, t in self.tracks.items():
                d = ((cx - t["cx"])**2 + (cy - t["cy"])**2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "cx": cx, "cy": cy,
                    "prev_cy": cy,
                    "crossed": False,
                    "ttl": 10,
                    "conf": conf,
                }
            else:
                tid = best_id
                matched_ids.add(tid)

            t = self.tracks[tid]
            prev_cy = t["cy"]
            t["cx"], t["cy"], t["ttl"], t["conf"] = cx, cy, 10, conf

            if not t["crossed"]:
                if prev_cy < line_y and cy >= line_y:
                    t["crossed"] = True
                    self.board_count += 1
                    events.append(("BOARD", tid, conf))
                elif prev_cy > line_y and cy <= line_y:
                    t["crossed"] = True
                    self.deboard_count += 1
                    events.append(("DEBOARD", tid, conf))

        # age out lost tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_ids:
                self.tracks[tid]["ttl"] -= 1
                if self.tracks[tid]["ttl"] <= 0:
                    del self.tracks[tid]

        return events

    def get_boxes(self, frame_h: int):
        line_y = int(frame_h * self.line_ratio)
        return [(t["cx"], t["cy"], t.get("conf", 1.0)) for t in self.tracks.values()], line_y


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class CameraProcessor:
    def __init__(self, bus_id: str, camera_index: int = 0, line_ratio: float = 0.5):
        self.bus_id = bus_id
        self.camera_index = camera_index
        self.tracker = PersonTracker(line_ratio)
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.model = YOLO("yolov8n.pt") if YOLO_AVAILABLE else None
        self.prev_gray = None
        self.fps = 0.0
        self.frame_count = 0

    def _detect_yolo(self, frame):
        results = self.model(frame, classes=[0], verbose=False, conf=0.4)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append((cx, cy, conf))
        return detections

    def _detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) > 1200:
                x, y, w, h = cv2.boundingRect(c)
                detections.append((x + w // 2, y + h // 2, 0.7))
        return detections

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        detections = self._detect_yolo(frame) if self.model else self._detect_motion(frame)
        events = self.tracker.update(detections, h)
        annotated = self._annotate(frame.copy(), detections, h)
        return events, annotated, detections

    def _annotate(self, frame, detections, h):
        line_y = int(h * self.tracker.line_ratio)
        # Detection line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 220, 180), 2)
        cv2.putText(frame, "DETECTION LINE", (10, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 180), 1)
        # Bounding boxes
        for i, (cx, cy, conf) in enumerate(detections):
            cv2.rectangle(frame, (cx - 20, cy - 35), (cx + 20, cy + 35), (50, 180, 255), 2)
            cv2.putText(frame, f"{conf:.0%}", (cx - 18, cy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 180, 255), 1)
        # Stats overlay
        cv2.rectangle(frame, (0, 0), (220, 56), (0, 0, 0), -1)
        cv2.putText(frame, f"Boarded:  {self.tracker.board_count}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 200, 120), 1)
        cv2.putText(frame, f"Deboarded: {self.tracker.deboard_count}", (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 100, 50), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (8, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        return frame

    def frame_to_b64(self, frame) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode()


processors: Dict[str, CameraProcessor] = {}

# ══════════════════════════════════════════════════════════════════════════════
#  REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"service": "BusGate API", "version": "1.0.0", "status": "running"}

@app.get("/api/buses")
def get_buses():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM buses").fetchall()
        return [dict(r) for r in rows]

@app.get("/api/stops")
def get_stops():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM stops").fetchall()
        return [dict(r) for r in rows]

@app.get("/api/events")
def get_events(bus_id: Optional[str] = None, date_filter: Optional[str] = None, limit: int = 100):
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if bus_id:
        query += " AND bus_id=?"; params.append(bus_id)
    if date_filter:
        query += " AND date=?"; params.append(date_filter)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

@app.get("/api/summary")
def get_summary(date_filter: Optional[str] = None):
    d = date_filter or str(date.today())
    with get_db() as conn:
        total_board = conn.execute(
            "SELECT COALESCE(SUM(count),0) FROM events WHERE event_type='BOARD' AND date=?", (d,)
        ).fetchone()[0]
        total_deboard = conn.execute(
            "SELECT COALESCE(SUM(count),0) FROM events WHERE event_type='DEBOARD' AND date=?", (d,)
        ).fetchone()[0]
        hourly = conn.execute("""
            SELECT hour,
                   SUM(CASE WHEN event_type='BOARD' THEN count ELSE 0 END) as boarded,
                   SUM(CASE WHEN event_type='DEBOARD' THEN count ELSE 0 END) as deboarded
            FROM events WHERE date=?
            GROUP BY hour ORDER BY hour
        """, (d,)).fetchall()
        by_stop = conn.execute("""
            SELECT stop_name,
                   SUM(CASE WHEN event_type='BOARD' THEN count ELSE 0 END) as boarded,
                   SUM(CASE WHEN event_type='DEBOARD' THEN count ELSE 0 END) as deboarded
            FROM events WHERE date=?
            GROUP BY stop_name ORDER BY (boarded+deboarded) DESC
        """, (d,)).fetchall()
        buses = conn.execute("SELECT * FROM buses").fetchall()
    return {
        "date": d,
        "total_board": total_board,
        "total_deboard": total_deboard,
        "net_occupancy": max(0, total_board - total_deboard),
        "hourly": [dict(r) for r in hourly],
        "by_stop": [dict(r) for r in by_stop],
        "buses": [dict(b) for b in buses],
    }

@app.post("/api/events")
async def post_event(ev: EventIn):
    now = datetime.now()
    eid = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute("""
            INSERT INTO events(id,bus_id,route,stop_name,event_type,count,timestamp,hour,date,confidence,track_id)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """, (eid, ev.bus_id, ev.route, ev.stop_name, ev.event_type,
              ev.count, now.isoformat(), now.hour, str(now.date()),
              ev.confidence, ev.track_id))
        delta = ev.count if ev.event_type == "BOARD" else -ev.count
        conn.execute(
            "UPDATE buses SET occupancy=MAX(0,occupancy+?),last_seen=? WHERE bus_id=?",
            (delta, now.isoformat(), ev.bus_id)
        )
    payload = {
        "type": "event",
        "id": eid,
        "bus_id": ev.bus_id,
        "event_type": ev.event_type,
        "stop_name": ev.stop_name,
        "count": ev.count,
        "timestamp": now.isoformat(),
        "confidence": ev.confidence,
    }
    await manager.broadcast(payload)
    return {"status": "ok", "event_id": eid}

@app.patch("/api/buses/{bus_id}")
async def update_bus(bus_id: str, upd: BusUpdate):
    fields, vals = [], []
    if upd.current_stop:
        fields.append("current_stop=?"); vals.append(upd.current_stop)
    if upd.status:
        fields.append("status=?"); vals.append(upd.status)
    if not fields:
        raise HTTPException(400, "Nothing to update")
    vals.append(bus_id)
    with get_db() as conn:
        conn.execute(f"UPDATE buses SET {','.join(fields)} WHERE bus_id=?", vals)
    await manager.broadcast({"type": "bus_update", "bus_id": bus_id, **upd.dict(exclude_none=True)})
    return {"status": "ok"}

@app.delete("/api/events/reset")
async def reset_today():
    d = str(date.today())
    with get_db() as conn:
        conn.execute("DELETE FROM events WHERE date=?", (d,))
        conn.execute("UPDATE buses SET occupancy=0")
    await manager.broadcast({"type": "reset", "date": d})
    return {"status": "ok", "date": d}

# ══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET — dashboard clients
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "connected", "msg": "BusGate live feed active"}))
        while True:
            await ws.receive_text()   # keep-alive ping
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET — camera stream (per bus)
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/camera/{bus_id}")
async def ws_camera(ws: WebSocket, bus_id: str, camera_index: int = 0, line_ratio: float = 0.5):
    await ws.accept()
    proc = CameraProcessor(bus_id, camera_index, line_ratio)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        await ws.send_text(json.dumps({"type": "error", "msg": "Camera not available"}))
        await ws.close()
        return

    processors[bus_id] = proc
    t_last = time.time()
    frame_count = 0

    try:
        with get_db() as conn:
            bus = dict(conn.execute("SELECT * FROM buses WHERE bus_id=?", (bus_id,)).fetchone() or {})
        stop_name = bus.get("current_stop", "Unknown")
        route = bus.get("route", "Unknown")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))
            events, annotated, _ = proc.process_frame(frame)
            frame_count += 1

            now = time.time()
            if now - t_last >= 1.0:
                proc.fps = frame_count / (now - t_last)
                frame_count = 0
                t_last = now

            b64 = proc.frame_to_b64(annotated)
            payload = {
                "type": "frame",
                "bus_id": bus_id,
                "frame": b64,
                "fps": round(proc.fps, 1),
                "board_count": proc.tracker.board_count,
                "deboard_count": proc.tracker.deboard_count,
                "active_tracks": len(proc.tracker.tracks),
                "events": [],
            }

            for ev_type, track_id, conf in events:
                ev = EventIn(
                    bus_id=bus_id, route=route, stop_name=stop_name,
                    event_type=ev_type, count=1, confidence=conf, track_id=track_id,
                )
                await post_event(ev)
                payload["events"].append({"type": ev_type, "track_id": track_id, "conf": round(conf, 2)})

            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.033)   # ~30 fps cap

    except (WebSocketDisconnect, Exception) as e:
        print(f"[ws/camera/{bus_id}] disconnected: {e}")
    finally:
        cap.release()
        processors.pop(bus_id, None)
        await ws.close()
