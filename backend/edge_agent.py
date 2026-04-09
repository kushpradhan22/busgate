#!/usr/bin/env python3
"""
BusGate — Edge Device Agent
Run this on Raspberry Pi 5 / Jetson Nano mounted inside the bus.
Captures from local camera, runs detection, POSTs events to central server.

Usage:
    python edge_agent.py --bus-id B-101 --server http://192.168.1.100:8000
"""

import argparse
import time
import requests
import cv2
import numpy as np
from datetime import datetime

YOLO_AVAILABLE = False
print("[INFO] Running in motion detection mode")

BUS_STOPS = [
    "Central Station", "Market Square", "University",
    "Hospital", "Park & Ride", "Airport T1",
]


class EdgeAgent:
    def __init__(self, bus_id: str, route: str, server_url: str,
                 camera_index: int = 0, line_ratio: float = 0.5):
        self.bus_id      = bus_id
        self.route       = route
        self.server_url  = server_url.rstrip("/")
        self.camera_index = camera_index
        self.line_ratio  = line_ratio
        self.current_stop = BUS_STOPS[0]
        self.tracks      = {}
        self.next_id     = 1
        self.model       = YOLO("yolov8n.pt") if YOLO_AVAILABLE else None
        self.prev_gray   = None
        self.session     = requests.Session()

    # ── Detection ──────────────────────────────────────────────────────────
    def detect(self, frame):
        if self.model:
            results = self.model(frame, classes=[0], verbose=False, conf=0.4)
            dets = []
            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    dets.append((int((x1+x2)/2), int((y1+y2)/2), conf))
            return dets
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(15,15),0)
            if self.prev_gray is None:
                self.prev_gray = gray; return []
            diff = cv2.absdiff(self.prev_gray, gray)
            self.prev_gray = gray
            _, thresh = cv2.threshold(diff,20,255,cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            dets = []
            for c in cnts:
                if cv2.contourArea(c) > 1200:
                    x,y,w,h = cv2.boundingRect(c)
                    dets.append((x+w//2, y+h//2, 0.7))
            return dets

    # ── Tracking & line crossing ────────────────────────────────────────────
    def update_tracks(self, dets, frame_h):
        line_y = int(frame_h * self.line_ratio)
        events = []
        matched = set()
        for cx,cy,conf in dets:
            best, bd = None, 80
            for tid,t in self.tracks.items():
                d = ((cx-t["cx"])**2+(cy-t["cy"])**2)**.5
                if d<bd: bd=d; best=tid
            if best is None:
                tid = self.next_id; self.next_id+=1
                self.tracks[tid] = {"cx":cx,"cy":cy,"crossed":False,"ttl":10,"conf":conf}
            else:
                tid = best; matched.add(tid)
            t = self.tracks[tid]
            prev_cy = t["cy"]
            t.update({"cx":cx,"cy":cy,"ttl":10,"conf":conf})
            if not t["crossed"]:
                if prev_cy < line_y <= cy:
                    t["crossed"]=True; events.append(("BOARD",tid,conf))
                elif prev_cy > line_y >= cy:
                    t["crossed"]=True; events.append(("DEBOARD",tid,conf))
        for tid in list(self.tracks):
            if tid not in matched:
                self.tracks[tid]["ttl"] -= 1
                if self.tracks[tid]["ttl"] <= 0:
                    del self.tracks[tid]
        return events

    # ── POST event to server ────────────────────────────────────────────────
    def post_event(self, event_type: str, track_id: int, conf: float):
        payload = {
            "bus_id":    self.bus_id,
            "route":     self.route,
            "stop_name": self.current_stop,
            "event_type": event_type,
            "count":     1,
            "confidence": round(conf, 3),
            "track_id":  track_id,
        }
        try:
            r = self.session.post(f"{self.server_url}/api/events", json=payload, timeout=3)
            status = "OK" if r.status_code == 200 else f"ERR {r.status_code}"
        except requests.exceptions.ConnectionError:
            status = "SERVER OFFLINE — event buffered"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {event_type:8s} | "
              f"track={track_id} | conf={conf:.0%} | stop={self.current_stop} | {status}")

    # ── Main loop ───────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[BusGate Edge] Bus={self.bus_id} | Route={self.route} | "
              f"Server={self.server_url} | YOLO={YOLO_AVAILABLE}")

        fps_t = time.time()
        frames = 0
        while True:
            ret, frame = cap.read()
            if not ret: continue
            h,w = frame.shape[:2]
            dets   = self.detect(frame)
            events = self.update_tracks(dets, h)
            for ev_type, tid, conf in events:
                self.post_event(ev_type, tid, conf)
            frames += 1
            if time.time() - fps_t >= 5:
                print(f"[FPS] {frames/5:.1f}")
                fps_t = time.time(); frames = 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BusGate Edge Agent")
    p.add_argument("--bus-id",       default="B-101",                      help="Bus ID")
    p.add_argument("--route",        default="Route 12A",                  help="Route name")
    p.add_argument("--server",       default="http://localhost:8000",       help="Backend URL")
    p.add_argument("--camera",       default=0,       type=int,            help="Camera index")
    p.add_argument("--line-ratio",   default=0.5,     type=float,          help="Detection line position (0–1)")
    p.add_argument("--stop",         default="Central Station",            help="Initial stop name")
    args = p.parse_args()

    agent = EdgeAgent(
        bus_id=args.bus_id,
        route=args.route,
        server_url=args.server,
        camera_index=args.camera,
        line_ratio=args.line_ratio,
    )
    agent.current_stop = args.stop
    agent.run()
