from fastapi import FastAPI
from multiprocessing import shared_memory
from loguru import logger
import numpy as np
import cv2
import json
import os

app = FastAPI()

# Configura o loguru (opcional: envia para arquivo também)
logger.add("/var/log/server.log", level="INFO")

@app.get("/infer")
def infer(frame_id: str):
    logger.info(f"Received inference request for frame_id: {frame_id}")

    shm_name = f"frame_{frame_id}"
    shape = (720, 1280, 3)

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        logger.info(f"Successfully connected to shared memory: {shm_name}")
    except FileNotFoundError:
        logger.error(f"Shared memory {shm_name} not found.")
        return {"status": "error", "message": "Shared memory not found"}

    try:
        image = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        img_copy = image.copy()
        logger.info(f"Frame shape: {img_copy.shape}")
    except Exception as e:
        logger.error(f"Error reading frame from SHM: {e}")
        shm.close()
        return {"status": "error", "message": "Failed to read image"}

    shm.close()

    # Dummy detection logic
    h, w = img_copy.shape[:2]
    detection = {"x": int(w / 4), "y": int(h / 4), "w": int(w / 2), "h": int(h / 2)}
    logger.info(f"Generated dummy detection: {detection}")

    # Save result JSON
    os.makedirs("/tmp/results", exist_ok=True)
    result_path = f"/tmp/results/{frame_id}.json"
    with open(result_path, "w") as f:
        json.dump({"detections": [detection]}, f)

    logger.info(f"Saved detection result to: {result_path}")

    return {"status": "success"}