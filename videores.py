"""SH17: YOLO detections -> person tracking -> PPE association -> events.
Run: python videores.py --device 0
Labels use ASCII because OpenCV putText does not render Cyrillic.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import math
import time
from datetime import datetime
from uuid import uuid4

BASE = Path(__file__).resolve().parent
PPE_CONF = {'helmet': 0.20, 'safety-vest': 0.20, 'gloves': 0.20}
# Relative regions (x1, y1, x2, y2) within a person's box.
REGIONS = {'helmet': (-0.10, -0.12, 1.10, 0.30),
           'safety-vest': (0.05, 0.18, 0.95, 0.70),
           'gloves': (-0.20, 0.25, 1.20, 0.95)}
REQUIRED = ('helmet', 'safety-vest')  # Gloves are informational by default.
CONFIRM_SECONDS = 2.0
MIN_OBSERVATIONS = 5
HOLD_SECONDS = 0.8
DEBOUNCE_SECONDS = 5.0
# No default polygon: zones must be explicitly configured for each source.


def configure_zone(frame, path, source):
    """Select a convex floor polygon on a scaled preview; save normalized points."""
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    scale = min(1.0, 1200 / w, 750 / h)
    preview = cv2.resize(frame, (round(w*scale), round(h*scale)))
    ph, pw = preview.shape[:2]
    points = []
    window = 'Zone setup: LMB add | RMB undo | R reset | ENTER save | ESC cancel'

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((max(0, min(pw-1, x)), max(0, min(ph-1, y))))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            view = preview.copy()
            for i, point in enumerate(points):
                cv2.circle(view, point, 5, (0, 0, 255), -1)
                cv2.putText(view, str(i+1), point, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
            if len(points) > 1:
                cv2.polylines(view, [np.array(points, np.int32)],
                              len(points) >= 3, (0, 0, 255), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(20) & 0xff
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if key == ord('r'):
                points.clear()
            if key in (10, 13):
                polygon = np.array(points, np.int32)
                if (len(set(points)) < 3 or not cv2.isContourConvex(polygon)
                        or abs(cv2.contourArea(polygon)) < 25):
                    print('Choose at least 3 points around a convex polygon; no crossings.')
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    'version': 1, 'source': source, 'frame_size': [w, h],
                    'points': [[x/(pw-1), y/(ph-1)] for x, y in points],
                }, indent=2), encoding='utf-8')
                print(f'Zone saved: {path}')
                return True
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def load_zone(path, source, width, height):
    import cv2
    import numpy as np
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or data.get('source') != source:
        raise ValueError('Zone belongs to another source. Run --setup-zone again.')
    old_w, old_h = data['frame_size']
    if old_w <= 0 or old_h <= 0 or abs((width/height)/(old_w/old_h)-1) > 0.01:
        raise ValueError('Frame aspect ratio changed. Run --setup-zone again.')
    points = np.asarray(data['points'], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any()):
        raise ValueError('Invalid zone coordinates. Run --setup-zone again.')
    polygon = np.rint(points * [width-1, height-1]).astype(np.int32)
    if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 25:
        raise ValueError('Invalid polygon. Run --setup-zone again.')
    return polygon


def region(box, name):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a, b, c, d = REGIONS[name]
    return (x1 + a*w, y1 + b*h, x1 + c*w, y1 + d*h)


def area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def intersection(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]),
                 min(a[2], b[2]), min(a[3], b[3])))


def associate(people, equipment):
    """Each PPE detection gets at most one owner; ambiguous owners stay unknown.
    people: [{'id': int, 'box': xyxy}]; equipment: [(name, xyxy, confidence)].
    """
    found = {p['id']: set() for p in people}
    ambiguous = {p['id']: set() for p in people}
    for name, box, confidence in equipment:
        if name not in REGIONS or confidence < PPE_CONF[name]:
            continue
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        candidates = []
        for p in people:
            r = region(p['box'], name)
            coverage = intersection(box, r) / max(area(box), 1)
            if not (r[0] <= cx <= r[2] and r[1] <= cy <= r[3]) or coverage < 0.5:
                continue
            dx = (cx-(r[0]+r[2])/2) / max(r[2]-r[0], 1)
            dy = (cy-(r[1]+r[3])/2) / max(r[3]-r[1], 1)
            candidates.append((coverage - 0.3*math.hypot(dx, dy), p['id']))
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0]-candidates[1][0] < 0.15:
            for _, pid in candidates:
                ambiguous[pid].add(name)
        elif candidates:
            found[candidates[0][1]].add(name)
    return found, ambiguous


def assessable(box, name, width, height):
    # Coarse visibility gate; does NOT solve occlusions or bent postures.
    if box[3]-box[1] < 100 or box[2]-box[0] < 35:
        return False
    r = region(box, name)
    clipped = (max(0, r[0]), max(0, r[1]), min(width, r[2]), min(height, r[3]))
    return intersection(r, clipped) / max(area(r), 1) >= 0.95


class SafetyState:
    """Independent per stream; timestamps must increase (seconds)."""
    def __init__(self):
        self.people = {}
        self.last_event = {}
        self.total = 0

    def update(self, pid, now, seen, unknown, visible, danger):
        s = self.people.get(pid)
        if s is None:
            s = {'last': now, 'ppe': {}}
            self.people[pid] = s
        # Do not accumulate missing evidence during a tracking gap.
        if now-s['last'] > 0.5:
            s['ppe'].clear()
        s['last'] = now
        statuses = {}
        for name in PPE_CONF:
            v = s['ppe'].setdefault(name, {'hit': -math.inf, 'start': None, 'n': 0})
            if name in seen:
                v.update(hit=now, start=None, n=0)
                status = 'present'
            elif name in unknown or not visible[name]:
                v.update(start=None, n=0)
                status = 'unknown'
            elif now-v['hit'] <= HOLD_SECONDS:
                status = 'present'
            else:
                if v['start'] is None:
                    v['start'] = now
                v['n'] += 1
                status = ('missing' if now-v['start'] >= CONFIRM_SECONDS
                          and v['n'] >= MIN_OBSERVATIONS else 'unknown')
            statuses[name] = status
        kinds = [f'no_{k}' for k in REQUIRED if statuses[k] == 'missing']
        if danger:
            kinds.append('danger_zone')
        events = []
        for kind in kinds:
            key = (pid, kind)
            if now-self.last_event.get(key, -math.inf) >= DEBOUNCE_SECONDS:
                self.last_event[key] = now
                self.total += 1
                events.append({'event_id': self.total, 'time_s': round(now, 3),
                               'person_id': pid, 'type': kind,
                               'suspected': kind != 'danger_zone'})
        return statuses, events

    def prune(self, now):
        expired = {pid for pid, s in self.people.items() if now-s['last'] > 30}
        for pid in expired:
            del self.people[pid]
        self.last_event = {k: t for k, t in self.last_event.items() if k[0] not in expired}


class ScreenshotStore:
    """One annotated image per event-bearing frame plus its JSON metadata."""
    def __init__(self, root, source, model):
        self.folder = Path(root) / (datetime.now().strftime('%Y%m%d_%H%M%S')
                                    + '_' + uuid4().hex[:8])
        self.folder.mkdir(parents=True, exist_ok=False)
        self.source, self.model = source, model
        self.sequence = 0
        print(f'Screenshots: {self.folder.resolve()}')

    def save(self, frame, now, events, manual=False):
        import cv2
        self.sequence += 1
        kind = 'manual' if manual else 'alert'
        stem = f'{self.sequence:06d}_{kind}_{now:010.3f}s'
        image_path = self.folder / (stem + '.jpg')
        metadata_path = self.folder / (stem + '.json')
        try:
            ok, encoded = cv2.imencode('.jpg', frame)
            if not ok:
                raise OSError('JPEG encoding failed')
            # Python handles Windows paths with Cyrillic, unlike some cv2.imwrite builds.
            image_path.write_bytes(encoded.tobytes())
            metadata_path.write_text(json.dumps({
                'time_s': round(now, 3), 'saved_at': datetime.now().astimezone().isoformat(),
                'source': self.source, 'model': self.model,
                'manual': manual, 'events': events, 'image': image_path.name,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        except (OSError, cv2.error) as exc:
            print(f'SCREENSHOT ERROR: {exc}', flush=True)
            return None
        print(f'Saved: {image_path}', flush=True)
        return image_path


def hold_frame(frame, store, now, events, eof=False):
    """No reads, inference or automatic events while paused or at end of file."""
    import cv2
    view = frame.copy()
    text = 'END: S save | Q/ESC exit' if eof else 'PAUSED: SPACE resume | S save | Q exit'
    cv2.rectangle(view, (0, 80), (min(frame.shape[1]-1, 720), 115), (0, 0, 0), -1)
    cv2.putText(view, text, (12, 105), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    while True:
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xff
        if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
        if key == ord('s'):
            store.save(frame, now, events, manual=True)
        if key == ord(' ') and not eof:
            return True


WINDOW = 'SH17 PPE | SPACE pause | S screenshot | Q quit'


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(BASE/'best_v2.pt'))
    parser.add_argument('--source', default=str(BASE/'videos'/'construction.mp4'))
    parser.add_argument('--device', default='0', help='0 for CUDA; cpu for CPU')
    parser.add_argument('--live', action='store_true', help='Wall clock for camera/RTSP')
    parser.add_argument('--zone-file', default=None, help='Explicit saved zone JSON')
    parser.add_argument('--setup-zone', action='store_true', help='Draw zone and exit')
    parser.add_argument('--screenshots-dir', default=str(BASE/'screenshots'/'violations'))
    args = parser.parse_args()
    zone_path = Path(args.zone_file) if args.zone_file else BASE/'zone.json'
    source_key = args.source if (args.source.isdecimal() or '://' in args.source) else str(Path(args.source).resolve())
    source = int(args.source) if args.source.isdecimal() else args.source
    live = args.live or isinstance(source, int) or '://' in args.source
    if args.setup_zone:
        setup_cap = cv2.VideoCapture(source)
        try:
            ok, first = setup_cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read video: {source}')
            configure_zone(first, zone_path, source_key)
        finally:
            setup_cap.release()
        return
    model = YOLO(args.model)
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    ids = {v: k for k, v in names.items()}
    expected = {'person': 0, 'gloves': 9, 'helmet': 10, 'safety-vest': 16}
    if any(ids.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Expected SH17 classes, got {names}')
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
        print('FPS unavailable: assuming 25 FPS for file timestamps.')
    # track_buffer is in frames; PPE never goes through the tracker.
    tracker = BYTETracker(SimpleNamespace(track_high_thresh=0.35,
        track_low_thresh=0.10, new_track_thresh=0.35,
        track_buffer=30, match_thresh=0.8, fuse_score=True))
    state = SafetyState()
    zone = None
    zone_size = None
    if not args.zone_file:
        print('Zone DISABLED. Use --setup-zone, then --zone-file zone.json.')
    store = ScreenshotStore(args.screenshots_dir, args.source, args.model)
    started, frame_no = time.monotonic(), 0
    last_frame, last_events, now = None, [], 0.0
    try:
        while True:
            tick = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                if last_frame is not None and not live:
                    print('End of video. S saves the last frame; Q closes the window.')
                    hold_frame(last_frame, store, now, last_events, eof=True)
                elif live:
                    print('Stream read failed.')
                break
            frame_events = []
            now = time.monotonic()-started if live else frame_no/fps
            frame_no += 1
            result = model.predict(frame, conf=0.10, iou=0.5, imgsz=640,
                device=args.device, classes=list(expected.values()), verbose=False)[0]
            boxes = result.boxes.cpu().numpy()
            person_boxes = boxes[boxes.cls == ids['person']]
            tracks = tracker.update(person_boxes, frame)
            people = [{'id': int(t[4]), 'box': tuple(map(float, t[:4]))} for t in tracks]
            equipment = [(names[int(b[5])], tuple(map(float, b[:4])), float(b[4]))
                         for b in boxes.data if int(b[5]) != ids['person']]
            found, ambiguous = associate(people, equipment)
            height, width = frame.shape[:2]
            if args.zone_file and zone_size != (width, height):
                zone = load_zone(zone_path, source_key, width, height)
                zone_size = (width, height)
            if zone is not None:
                cv2.polylines(frame, [zone], True, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'ZONE DISABLED', (12, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            for name, box, confidence in equipment:
                if confidence >= PPE_CONF[name]:
                    a, b, c, d = map(int, box)
                    cv2.rectangle(frame, (a, b), (c, d), (255, 200, 0), 1)
            for p in people:
                pid, box = p['id'], p['box']
                foot = ((box[0]+box[2])/2, box[3])
                danger = zone is not None and cv2.pointPolygonTest(zone, foot, False) >= 0
                visible = {k: assessable(box, k, width, height) for k in PPE_CONF}
                statuses, events = state.update(pid, now, found[pid], ambiguous[pid], visible, danger)
                frame_events.extend(events)
                for event in events:
                    print(json.dumps(event, ensure_ascii=False), flush=True)
               
                color = {'present': (0, 200, 0), 'missing': (0, 0, 255),
                         'unknown': (0, 200, 255)}[statuses['helmet']]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if danger:
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 2)
                short = {'present': 'OK', 'missing': 'MISS?', 'unknown': '?'}
                label = (f'ID{pid} H:{short[statuses["helmet"]]} '
                         f'V:{short[statuses["safety-vest"]]} G:{short[statuses["gloves"]]}'
                         + (' DANGER' if danger else ''))
                cv2.putText(frame, label, (max(0, x1), max(18, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            state.prune(now)
            cv2.rectangle(frame, (5, 5), (345, 48), (0, 0, 0), -1)
            cv2.putText(frame, f'Events: {state.total}  Time: {now:.1f}s', (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if frame_events:
                store.save(frame, now, frame_events)
            last_frame, last_events = frame.copy(), frame_events
            cv2.imshow(WINDOW, frame)
            delay = 1 if live else max(1, math.ceil((1/fps - (time.monotonic()-tick))*1000))
            key = cv2.waitKey(delay) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('s'):
                store.save(frame, now, frame_events, manual=True)
            if key == ord(' '):
                if not hold_frame(frame, store, now, frame_events):
                    break
                if live:
                    state.people.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f'Total events: {state.total}')


if __name__ == '__main__':
    main()
