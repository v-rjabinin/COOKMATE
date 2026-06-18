from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import os
import numpy as np
from typing import List, Dict
import torch

app = FastAPI(title="CookMate Food Detector")

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
MODEL_NANO_PATH = os.path.join(MODEL_DIR, "yolo11n_best.pt")
MODEL_SMALL_PATH = os.path.join(MODEL_DIR, "yolo11s_best.pt")

CONF_NANO = float(os.getenv("CONF_NANO", "0.30"))
CONF_SMALL = float(os.getenv("CONF_SMALL", "0.75"))
IOU_NMS = float(os.getenv("IOU_NMS", "0.45"))

if not os.path.exists(MODEL_NANO_PATH):
    raise RuntimeError(f"Модель nano не найдена: {MODEL_NANO_PATH}")
if not os.path.exists(MODEL_SMALL_PATH):
    raise RuntimeError(f"Модель small не найдена: {MODEL_SMALL_PATH}")

print(f"Загрузка моделей из {MODEL_DIR}...")
model_nano = YOLO(MODEL_NANO_PATH)
model_small = YOLO(MODEL_SMALL_PATH)
print(f"Модели загружены! | Nano conf={CONF_NANO}, Small conf={CONF_SMALL}")


def run_prediction(model, img, conf: float) -> List[Dict]:
    """Запускает предсказание одной моделью"""
    results = model.predict(img, conf=conf, iou=IOU_NMS, verbose=False, device='cpu')
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf_val = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls]
        detections.append({
            "class": class_name,
            "confidence": round(conf_val, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "source": "small" if model is model_small else "nano"
        })
    return detections


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Вычисляет IoU между двумя bounding box"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def merge_detections(small_dets: List[Dict], nano_dets: List[Dict]) -> List[Dict]:
    """
    Объединяет детекции: small как основа + nano как дополнение.
    Удаляет дубликаты по IoU.
    """
    merged = list(small_dets)  # начинаем со всех детекций small

    for nano_det in nano_dets:
        is_duplicate = False
        for existing_det in merged:
            # если тот же класс & высокий IoU => дубликат
            if nano_det["class"] == existing_det["class"] and compute_iou(nano_det["bbox"], existing_det["bbox"]) > IOU_NMS:
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(nano_det)

    return merged


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Основной endpoint — ансамбль двух моделей"""
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

        # запускаем обе модели (параллельно)
        small_dets = run_prediction(model_small, img, CONF_SMALL)
        nano_dets = run_prediction(model_nano, img, CONF_NANO)

        # объединяем результаты
        merged_dets = merge_detections(small_dets, nano_dets)

        # сортируем по уверенности (сначала самые уверенные)
        merged_dets.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "detections": merged_dets,
            "count": len(merged_dets),
            "stats": {
                "from_small": len(small_dets),
                "from_nano_added": len(merged_dets) - len(small_dets),
                "total_nano": len(nano_dets)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/small_only")
async def predict_small(file: UploadFile = File(...)):
    """Только small модель (для сравнения)"""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    dets = run_prediction(model_small, img, CONF_SMALL)
    return {"detections": dets, "count": len(dets)}


@app.post("/predict/nano_only")
async def predict_nano(file: UploadFile = File(...)):
    """Только nano модель (для сравнения)"""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    dets = run_prediction(model_nano, img, CONF_NANO)
    return {"detections": dets, "count": len(dets)}


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
