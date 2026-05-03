import cv2
import numpy as np

def draw_boxes(frame, detections, color_map=None):
    if color_map is None:
        color_map = {0: (0,0,255), 1: (255,0,0), 2: (0,255,0), 3: (255,255,0)}
    for d in detections:
        bbox = d["bbox"].astype(int)
        conf = d["confidence"]
        cls = d["class"]
        color = color_map.get(cls, (255,255,255))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = f"Tool_{cls}: {conf:.2f}"
        cv2.putText(frame, label, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def calculate_fps(frame_count, elapsed):
    return frame_count / elapsed if elapsed > 0 else 0.0