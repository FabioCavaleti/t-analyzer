from fastapi import FastAPI
from multiprocessing import shared_memory
from loguru import logger
import numpy as np
import cv2
import json
import os
from ultralytics import YOLO
from models.CourtDetector.CourtDetector import CourtDetector
import torch

app = FastAPI()

# Configura o loguru (opcional: envia para arquivo também)
logger.add("/var/log/server.log", level="INFO")

REGISTERED_MODELS = {}
REGISTERED_BALL_DETECTOR = False
REGISTERED_COURT_DETECTOR = False
REGISTERED_PLAYER_DETECTOR = False
SHAPE = None


def get_ball_detections(img: np.ndarray):
    results = REGISTERED_MODELS["ball_detector"].predict(img, imgsz=640)
    result = results[0]
    boxes = result.boxes
    ball_detections = []
    if boxes is not None and len(boxes) > 0:

        max_conf_idx = boxes.conf.argmax().item()

        box = boxes[max_conf_idx]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        class_id = int(box.cls[0])
        label = REGISTERED_MODELS["ball_detector"].names[class_id]

        ball_detections.append(
            {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "conf": conf,
                "classId": class_id,
                "label": label,
            }
        )
    return ball_detections


def get_court_keypoints(img: np.ndarray):
    keypoints = REGISTERED_MODELS["court_detector"].detect(img)
    return keypoints


def bbox_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_player_detections(img: np.ndarray, keypoints: list):
    if len(keypoints) < 14:
        return []
    results = REGISTERED_MODELS["player_detector"].predict(img, imgsz=640)
    result = results[0]
    boxes = result.boxes

    kps = np.array(keypoints).reshape(14, 2)
    sideA = kps[:7]
    sideB = kps[7:]
    centroidA = sideA.mean(axis=0)
    centroidB = sideB.mean(axis=0)

    player_detections = []
    if boxes is not None and len(boxes) > 0:
        candidates = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0])
            label = "Player"

            # center = bbox_center(x1, y1, x2, y2)
            # distA = np.linalg.norm(center - centroidA)
            # distB = np.linalg.norm(center - centroidB)

            candidates.append(
                {
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(x2 - x1),
                    "h": int(y2 - y1),
                    "conf": conf,
                    "classId": class_id,
                    "label": label,
                }
            )
        player_detections.append(
            min(
                candidates,
                key=lambda x: np.linalg.norm(
                    bbox_center(x["x"], x["y"], x["x"] + x["w"], x["y"] + x["h"])
                    - centroidA
                ),
            )
        )
        player_detections.append(
            min(
                candidates,
                key=lambda x: np.linalg.norm(
                    bbox_center(x["x"], x["y"], x["x"] + x["w"], x["y"] + x["h"])
                    - centroidB
                ),
            )
        )
    return player_detections


@app.get("/register_ball_detector")
def registerBallDetector():
    global REGISTERED_BALL_DETECTOR
    if REGISTERED_BALL_DETECTOR is False:
        weights = "/project/resources/models/ball_detector.pt"
        REGISTERED_MODELS["ball_detector"] = YOLO(weights)
        REGISTERED_BALL_DETECTOR = True
        return {"status": "success", "message": "Ball detector registered successfully"}
    return {"status": "fail", "message": "Ball detector already registered"}


@app.get("/register_court_detector")
def registerCourtDetector():
    global REGISTERED_COURT_DETECTOR
    if REGISTERED_COURT_DETECTOR is False:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        REGISTERED_MODELS["court_detector"] = CourtDetector(device)
        REGISTERED_COURT_DETECTOR = True
        return {
            "status": "success",
            "message": "Court detector registered successfully",
        }
    return {"status": "fail", "message": "Court detector already registered"}


@app.get("/register_player_detector")
def registerPlayerDetector():
    global REGISTERED_PLAYER_DETECTOR
    if REGISTERED_PLAYER_DETECTOR is False:
        weights = "/project/resources/models/yolov8n.pt"
        model = YOLO(weights)
        # print(model.classes)
        # model.classes = [0]  # Assuming class 0 is for players
        # model.names = {0: 'Player'}
        REGISTERED_MODELS["player_detector"] = model
        REGISTERED_PLAYER_DETECTOR = True
        return {
            "status": "success",
            "message": "Player detector registered successfully",
        }
    return {"status": "fail", "message": "Player detector already registered"}


@app.get("/register_shape")
def registerShape(height: int, width: int):
    global SHAPE
    if SHAPE is None:
        SHAPE = (height, width, 3)
        return {"status": "success", "message": f"Shape registered: {SHAPE}"}
    return {"status": "fail", "message": f"Shape already registered: {SHAPE}"}


@app.get("/infer")
def infer(frame_id: str):
    logger.info(f"Received inference request for frame_id: {frame_id}")
    if (
        not REGISTERED_BALL_DETECTOR
        or not REGISTERED_COURT_DETECTOR
        or not REGISTERED_PLAYER_DETECTOR
        or SHAPE is None
    ):
        logger.error("Models not registered or shape not set.")
        return {
            "status": "fail",
            "message": "Models not registered. Please register all models before inference.",
        }

    shm_name = f"frame_{frame_id}"

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        logger.info(f"Successfully connected to shared memory: {shm_name}")
    except FileNotFoundError:
        logger.error(f"Shared memory {shm_name} not found.")
        return {"status": "error", "message": "Shared memory not found"}

    try:
        image = np.ndarray(SHAPE, dtype=np.uint8, buffer=shm.buf)
        img_copy = image.copy()
        cv2.imwrite("/project/results/img.png", img_copy)
        logger.info(f"Frame shape: {img_copy.shape}")
    except Exception as e:
        logger.error(f"Error reading frame from SHM: {e}")
        shm.close()
        return {"status": "error", "message": "Failed to read image"}

    shm.close()

    ball_detections = get_ball_detections(img_copy)

    keypoints = get_court_keypoints(img_copy)

    player_detections = get_player_detections(img_copy, keypoints)

    print(ball_detections)
    print(player_detections)
    print(keypoints)
    # Save result JSON
    os.makedirs("/tmp/results", exist_ok=True)
    result_path = f"/tmp/results/{frame_id}.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "detections": ball_detections + player_detections,
                "keypoints": keypoints,
            },
            f,
        )

    logger.info(f"Saved detection result to: {result_path}")

    return {"status": "success"}
