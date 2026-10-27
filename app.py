from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from videores import SafetyState, associate, assessable, load_zone, plausible_person, PPE_CONF

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "best_v2.pt"
VIDEO_PATH = BASE / "videos" / "construction.mp4"
ZONE_PATH = BASE / "zone.json"
DEVICE = "0"
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.5
JPEG_QUALITY = 70

app = FastAPI()
events_log: list = []
events_lock = threading.Lock()
last_frame = None
frame_lock = threading.Lock()


def add_event(event: dict):
    with events_lock:
        events_log.append(event)
        if len(events_log) > 50:
            del events_log[:-50]


def process_video():
    global last_frame
    model = YOLO(str(MODEL_PATH))
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    ids = {v: k for k, v in names.items()}
    expected = {"person": 0, "gloves": 9, "helmet": 10, "safety-vest": 16}
    if any(ids.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Expected SH17 classes, got {names}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (fps and fps > 0):
        fps = 25.0

    tracker = BYTETracker(SimpleNamespace(
        track_high_thresh=0.55,#было .35
        track_low_thresh=0.10,
        new_track_thresh=0.35, 
        track_buffer=30,
        match_thresh=0.8, 
        fuse_score=True,
    ))
    state = SafetyState()

    zone = None
    zone_size = None
    frame_no = 0

    print(f"[app.py] Модель загружена: {MODEL_PATH.name}", flush=True)
    print(f"[app.py] Видео: {VIDEO_PATH.name}", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            # При перемотке видео время начинается с 0, поэтому старое состояние людей сбрасываем.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_no = 0
            state = SafetyState()
            continue

        now = frame_no / fps
        frame_no += 1

        result = model.predict(
            frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=640,
            device=DEVICE, classes=list(expected.values()), verbose=False,
        )[0]
        boxes = result.boxes.cpu().numpy()

        person_boxes = boxes[boxes.cls == ids["person"]]
        tracks = tracker.update(person_boxes, frame)
        people = [{"id": int(t[4]), "box": tuple(map(float, t[:4]))} for t in tracks]

        equipment = [
            (names[int(b[5])], tuple(map(float, b[:4])), float(b[4]))
            for b in boxes.data if int(b[5]) != ids["person"]
        ]

        found, ambiguous = associate(people, equipment)
        height, width = frame.shape[:2]

        if ZONE_PATH.exists():
            if zone_size != (width, height):
                try:
                    zone = load_zone(ZONE_PATH, str(VIDEO_PATH.resolve()), width, height)
                    zone_size = (width, height)
                except Exception as e:
                    print(f"[app.py] Зона не загружена: {e}", flush=True)
                    zone = None

        if zone is not None:
            cv2.polylines(frame, [zone], True, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "ZONE DISABLED", (12, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        for name, box, confidence in equipment:
            if confidence >= PPE_CONF.get(name, 0.2):
                a, b, c, d = map(int, box)
                cv2.rectangle(frame, (a, b), (c, d), (255, 200, 0), 1)

        for p in people:
            pid, box = p["id"], p["box"]
            foot = ((box[0] + box[2]) / 2, box[3])
            # Мелкие ложные рамки не считаем человеком для алерта зоны.
            danger = (zone is not None and plausible_person(box, width, height)
                      and cv2.pointPolygonTest(zone, foot, False) >= 0)
            visible = {k: assessable(box, k, width, height) for k in PPE_CONF}

            statuses, events = state.update(
                pid, now, found[pid], ambiguous[pid], visible, danger,
            )
            for event in events:
                event["wall_time"] = time.strftime("%H:%M:%S")
                add_event(event)

            color = {"present": (0, 200, 0), "missing": (0, 0, 255),
                     "unknown": (0, 200, 255)}[statuses["helmet"]]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if danger:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 2)

            short = {"present": "OK", "missing": "MISS?", "unknown": "?"}
            label = (
                f'ID{pid} H:{short[statuses["helmet"]]} '
                f'V:{short[statuses["safety-vest"]]} '
                f'G:{short[statuses["gloves"]]}'
                + (" DANGER" if danger else "")
            )
            cv2.putText(frame, label, (max(0, x1), max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        state.prune(now)

        with events_lock:
            total = state.total
        cv2.rectangle(frame, (5, 5), (345, 48), (0, 0, 0), -1)
        cv2.putText(frame, f"Events: {total}  Time: {now:.1f}s", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with frame_lock:
                last_frame = jpg.tobytes()

    cap.release()


@app.on_event("startup")
async def startup():
    t = threading.Thread(target=process_video, daemon=True)
    t.start()


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><title>PPE Safety Control</title>
<style>
body{background:#0e1117;color:#e6edf3;font-family:sans-serif;margin:0;padding:20px}
h1{font-size:22px;margin:0 0 16px 0}
.layout{display:grid;grid-template-columns:2fr 1fr;gap:20px}
.video{border:1px solid #30363d;border-radius:8px;overflow:hidden;background:#000}
.video img{width:100%;display:block}
.panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.panel h2{font-size:14px;margin:0 0 12px 0;color:#8b949e;text-transform:uppercase}
#events{list-style:none;padding:0;margin:0;font-family:monospace;font-size:13px;max-height:600px;overflow-y:auto}
#events li{padding:8px 10px;border-left:3px solid #d29922;background:#1c2128;margin-bottom:4px;border-radius:4px}
#events li.danger{border-left-color:#f85149}
#events li.helmet{border-left-color:#f0883e}
#events li.vest{border-left-color:#a371f7}
.meta{font-size:12px;color:#8b949e;margin-top:12px}
</style></head>
<body><h1>PPE Safety Control</h1>
<div class="layout">
<div class="video"><img src="/video" alt="Видеопоток"></div>
<div class="panel"><h2>Последние события</h2>
<ul id="events"><li>Ожидание...</li></ul>
<div class="meta">Модель: YOLO11s / SH17<br>«Опасная зона» — человек непрерывно в зоне не менее 0,5 с. «Предположительно» — наблюдений пока мало.</div></div>
</div>
<script>
const ev=document.getElementById('events');
// Понятные названия для типов событий; сами type остаются английскими.
const names={danger_zone:'Опасная зона',no_helmet:'Нет каски','no_safety-vest':'Нет жилета',no_gloves:'Нет перчаток'};
const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
ws.onmessage=(e)=>{const d=JSON.parse(e.data);
if(!d.length){ev.innerHTML='<li>Пока нарушений нет</li>';return;}
ev.innerHTML=d.map(x=>{const c=x.type==='danger_zone'?'danger':x.type.includes('helmet')?'helmet':x.type.includes('vest')?'vest':'';
return `<li class="${c}"><b>${x.wall_time}</b> · ${names[x.type]||x.type}${x.suspected?' (предположительно)':''}<br><small>ID ${x.person_id} · ${x.time_s.toFixed(1)} с</small></li>`}).join('')};
ws.onclose=()=>{ev.innerHTML='<li>Соединение потеряно.</li>'};
</script></body></html>"""


def mjpeg_stream():
    while True:
        with frame_lock:
            frame = last_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.get("/video")
async def video_feed():
    return StreamingResponse(mjpeg_stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with events_lock:
                recent = list(events_log[-20:])[::-1]
            await ws.send_text(json.dumps(recent, ensure_ascii=False))
            await asyncio.sleep(1)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)