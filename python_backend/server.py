from fastapi import FastAPI
from multiprocessing import shared_memory
from loguru import logger
import numpy as np
import cv2
import json
import os
from ultralytics import YOLO

app = FastAPI()

# Configura o loguru (opcional: envia para arquivo também)
logger.add("/var/log/server.log", level="INFO")

REGISTERED_MODEL = None

@app.get("/register")
def register(height: int, width: int, channels: int):
    global REGISTERED_MODEL
    if REGISTERED_MODEL is None:
        REGISTERED_MODEL = {
            "weights":'/project/resources/models/yolov8n.pt',
            "shape": (height, width, channels)
        }

@app.get("/infer")
def infer(frame_id: str):
    logger.info(f"Received inference request for frame_id: {frame_id}")
    if not REGISTERED_MODEL:
        return {"status":"fail"}

    shm_name = f"frame_{frame_id}"
    shape = REGISTERED_MODEL["shape"]

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        logger.info(f"Successfully connected to shared memory: {shm_name}")
    except FileNotFoundError:
        logger.error(f"Shared memory {shm_name} not found.")
        return {"status": "error", "message": "Shared memory not found"}

    try:
        image = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        img_copy = image.copy()
        cv2.imwrite("/project/results/img.png", img_copy)
        logger.info(f"Frame shape: {img_copy.shape}")
    except Exception as e:
        logger.error(f"Error reading frame from SHM: {e}")
        shm.close()
        return {"status": "error", "message": "Failed to read image"}

    shm.close()

    # Dummy detection logic
    h, w = img_copy.shape[:2]
    model = YOLO(REGISTERED_MODEL["weights"])    
    
    results = model.predict(img_copy, imgsz=640)
    result = results[0] 
    boxes = result.boxes
    detections = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0])
            label = model.names[class_id]

            detections.append({
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "conf": conf,
                "classId": class_id,
                "label": label  
            })

    print(detections)
    # Save result JSON
    os.makedirs("/tmp/results", exist_ok=True)
    result_path = f"/tmp/results/{frame_id}.json"
    with open(result_path, "w") as f:
        json.dump({"detections": detections}, f)

    logger.info(f"Saved detection result to: {result_path}")

    return {"status": "success"}