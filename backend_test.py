import cv2
import requests
import numpy as np
import json
from multiprocessing import shared_memory

if __name__ == "__main__":


    BACKEND_URL = "http://localhost:8000"
    path = 'resources/teste.png'

    img = cv2.imread(path)
    shape = img.shape
    h, w, _= img.shape
    requests.get(f"{BACKEND_URL}/register_ball_detector")
    requests.get(f"{BACKEND_URL}/register_court_detector")
    requests.get(f"{BACKEND_URL}/register_player_detector")
    requests.get(f"{BACKEND_URL}/register_shape?height={h}&width={w}")



    shm = shared_memory.SharedMemory(create=True, size=img.nbytes, name = f"frame_{0}")
    shm_buf = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    np.copyto(shm_buf, img)

    r = requests.get(f"{BACKEND_URL}/infer?frame_id={0}")

    print(r)

    results_path = ""
    if r.json()["status"] == "success":
        results_path = f"/tmp/results/{0}.json"
    
    if not results_path:
        print("Erro")

    with open(results_path, "r") as f:
        data = json.load(f)

    print(data)
    
  
    for det in data.get("ball_detections", []):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for det in data.get("player_detections", []):
        x, y, w, h = det["x"], det['y'], det['w'], det['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    kps = data.get("keypoints")
    for i in range(0, len(kps), 2):
        x, y = int(kps[i]), int(kps[i + 1])
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

    cv2.imwrite("/project/results/backend_test.png", img)
    

    shm.close()
    shm.unlink()



