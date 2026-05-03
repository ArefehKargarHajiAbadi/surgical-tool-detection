from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)

    def detect(self, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45):
        """Return detections with bounding boxes, confidence, class."""
        results = self.model(frame, conf=conf, iou=iou, device=self.device, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else []
            confs = r.boxes.conf.cpu().numpy() if r.boxes else []
            clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else []
            for box, c, cls in zip(boxes, confs, clss):
                detections.append({
                    "bbox": box,      # [x1, y1, x2, y2]
                    "confidence": float(c),
                    "class": int(cls)
                })
        return detections