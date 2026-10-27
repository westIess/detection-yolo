import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Путь к датасету — тот, где реально лежат картинки
DATASET_ROOT = Path("C:/Users/wlkvc/PyCharmMiscProject/datasets/construction-ppe")

# Ищем первую картинку в test/val/train
img_path = None
for split in ["test", "val", "train"]:
    folder = DATASET_ROOT / "images" / split
    if folder.exists():
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
        if images:
            img_path = images[0]
            break

if img_path is None:
    print(f"Ошибка: не нашёл картинок в {DATASET_ROOT}")
    exit()

print(f"Загружаю: {img_path}")
img = cv2.imread(str(img_path))

if img is None:
    print(f"Ошибка чтения картинки: {img_path}")
    exit()

print(f"Размер картинки: {img.shape}")

# ONNX
session = ort.InferenceSession("best_v1.onnx", providers=["CPUExecutionProvider"])
print(f"Вход модели: {session.get_inputs()[0].shape}")
print(f"Выход модели: {session.get_outputs()[0].shape}")

# Предобработка
img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0
img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

# Инференс
outputs = session.run(None, {session.get_inputs()[0].name: img_input})
print(f"\nФорма выхода: {outputs[0].shape}")
print(f"Тип выхода: {outputs[0].dtype}")
print(f"Максимальная уверенность: {outputs[0][0, 4:, :].max():.4f}")
print("\n ONNX-модель работает")