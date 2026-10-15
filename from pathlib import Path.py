from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    print("Запущен файл:", Path(__file__).resolve(), flush=True)

    model = YOLO("yolo11n.pt")

    model.train(
        data="C:/Users/wlkvc/datasets/construction-ppe/data.yaml",
        epochs=3,           # короткая проверка
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        cache=False,
        amp=True,
        plots=False,
        patience=15,
        name="ram_check",
    )