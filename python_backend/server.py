from fastapi import FastAPI
from multiprocessing import shared_memory
import numpy as np
import cv2
import json
import os

app = FastAPI()

@app.get("/infer")
def infer(frame_id: str):
    shm_name = f"frame_{frame_id}"
    shape = (720, 1280, 3)
    shm = shared_memory.SharedMemory(name=shm_name)
    image = np.ndarray(shape, dtype=np.uint8, buffer = shm.buf)
    img_copy = image.copy()
    shm.close()

    #dummy detection
    h, w = img_copy.shape[:2]
    detection = {"X": int(w / 4), "y": int(h / 4), "w": int(w / 2), "h": int(h / 2)}
    os.makedirs("/tmp/result", exist_ok=True)
    with open(f"/tmp/results/{frame_id}.json", "w") as f:
        json.dump({"detections": [detection]}, f)

    

    return {"status": "success"}