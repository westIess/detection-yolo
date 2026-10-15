import cv2
import numpy as np
import time
from ultralytics import YOLO


MODEL_PATH = "runs/detect/train-16/weights/best.pt"
VIDEO_PATH = "videos/construction.mp4"

DANGER_ZONE = np.array([
    [300, 400],
    [900, 400],
    [900, 700],
    [300, 700],
], dtype=np.int32)

VIOLATION_CLASSES = ["no_helmet", "no_goggle", "no_gloves", "no_boots"]
DEBOUNCE_SECONDS = 5


if __name__ == '__main__':
    print(f"Загружаю модель: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Классы модели: {list(model.names.values())}")

    print(f"Открываю видео: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        exit()

    violation_count = 0
    active_violations = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Видео закончилось.")
            break

        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=0.4, verbose=False,
        )
        annotated = results[0].plot()

        cv2.polylines(annotated, [DANGER_ZONE], True, (0, 0, 255), 3)
        cv2.putText(annotated, "DANGER ZONE",
                    (DANGER_ZONE[0][0], DANGER_ZONE[0][1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        now = time.time()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "Person":
                foot = ((x1 + x2) // 2, y2)
                if cv2.pointPolygonTest(DANGER_ZONE, foot, False) >= 0:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated, f"ID{track_id} IN DANGER",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                    key = f"danger_{track_id}"
                    if now - active_violations.get(key, 0) > DEBOUNCE_SECONDS:
                        active_violations[key] = now
                        violation_count += 1
                        print(f"[{time.strftime('%H:%M:%S')}] ID{track_id} в опасной зоне")

            if cls_name in VIOLATION_CLASSES:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(annotated, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                key = f"{cls_name}_{track_id}"
                if now - active_violations.get(key, 0) > DEBOUNCE_SECONDS:
                    active_violations[key] = now
                    violation_count += 1
                    print(f"[{time.strftime('%H:%M:%S')}] ID{track_id}: {cls_name}")

        cv2.rectangle(annotated, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.putText(annotated, f"Violations: {violation_count}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("PPE Safety Control", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nИтого нарушений: {violation_count}")