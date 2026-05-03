#!/usr/bin/env python3
"""Real‑time surgical tool detection using YOLOv8."""
import os
import cv2
import logging
from dotenv import load_dotenv
import yaml
from src.detector import YOLODetector
from src.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    load_dotenv()
    config = load_config()
    model_path = os.getenv("MODEL_PATH", "models/best.pt")
    video_source = os.getenv("VIDEO_SOURCE", "0")
    conf_thresh = float(os.getenv("CONF_THRESHOLD", config["inference"]["conf"]))
    iou_thresh = float(os.getenv("IOU_THRESHOLD", config["inference"]["iou"]))

    detector = YOLODetector(model_path, device=os.getenv("DEVICE", "cpu"))
    processor = VideoProcessor(detector, conf_thresh, iou_thresh, video_source)
    processor.run()

if __name__ == "__main__":
    main()