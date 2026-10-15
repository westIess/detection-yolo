from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    model.train(
        data="C:/Users/wlkvc/PyCharmMiscProject/datasets/construction-ppe/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=8,
        patience=15,
    )